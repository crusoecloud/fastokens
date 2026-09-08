use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fmt,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};
use serde::Deserialize;
use serde_json::Value;

use super::Result;
use crate::pre_tokenizers::BYTE_TO_CHAR;

type TokenId = u32;
type ParsedMergeMap = HashMap<(u32, u32), (u32, u32)>;
type Vocab = HashMap<String, u32>;

const INVALID_TOKEN: u32 = u32::MAX;

/// Pretokens with at most this many initial (per-byte) symbols use the
/// stack-resident linear-scan merge instead of the heap. Sized so the stack
/// arrays fit in registers/L1 and `u8` linked-list indices stay in range.
const SMALL_MERGE_MAX: usize = 32;

/// Open-addressing hash table for merge lookups.
#[derive(Clone, PartialEq)]
struct MergeMap {
    mask: usize,
    keys: Vec<u64>,
    vals: Vec<u32>,
}

const EMPTY_KEY: u64 = u64::MAX;

impl MergeMap {
    fn new() -> Self {
        Self {
            mask: 0,
            keys: Vec::new(),
            vals: Vec::new(),
        }
    }

    fn from_parsed(parsed: &ParsedMergeMap) -> Self {
        if parsed.is_empty() {
            return Self::new();
        }
        // ~50% load factor.
        let capacity = (parsed.len() * 2).next_power_of_two();
        let mask = capacity - 1;
        let mut keys = vec![EMPTY_KEY; capacity];
        let mut vals = vec![0u32; capacity];

        for (&(t1, t2), &(_rank, merged_id)) in parsed {
            let key = pack_pair(t1, t2);
            let mut idx = fx_hash(key) as usize & mask;
            loop {
                if keys[idx] == EMPTY_KEY {
                    keys[idx] = key;
                    vals[idx] = merged_id;
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self { mask, keys, vals }
    }

    /// Look up the merged token ID for a pair.
    #[inline(always)]
    fn get(&self, t1: u32, t2: u32) -> Option<u32> {
        if self.keys.is_empty() {
            return None;
        }
        let key = pack_pair(t1, t2);
        let mut idx = fx_hash(key) as usize & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == key {
                return Some(unsafe { *self.vals.get_unchecked(idx) });
            }
            if k == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    fn len(&self) -> usize {
        self.keys.iter().filter(|&&k| k != EMPTY_KEY).count()
    }
}

/// Bigram bridgeability table for vocab-aware safe splitting.
///
/// For each of 256×256 possible byte pairs, records whether that pair
/// appears in any vocabulary token. Used to identify split points that
/// cannot be crossed by BPE merges.
#[derive(Clone, PartialEq)]
pub struct BigramBridgeTable {
    /// Flat array: bridgeable[prev * 256 + cur] == true if some vocab
    /// token contains adjacent bytes (prev, cur).
    bridgeable: Box<[bool; 65536]>,
}

impl BigramBridgeTable {
    /// Check if a byte pair can be bridged by some vocab token.
    #[inline(always)]
    pub fn is_bridgeable(&self, prev: u8, cur: u8) -> bool {
        self.bridgeable[prev as usize * 256 + cur as usize]
    }
}

/// Build a bigram bridge table by scanning all vocab tokens.
fn build_bigram_bridge_table(id_to_token: &[String]) -> BigramBridgeTable {
    let mut bridgeable = Box::new([false; 65536]);

    for token_str in id_to_token {
        let bytes = token_str.as_bytes();
        // Mark all adjacent byte pairs in this token as bridgeable
        for window in bytes.windows(2) {
            let prev = window[0] as usize;
            let cur = window[1] as usize;
            bridgeable[prev * 256 + cur] = true;
        }
    }

    BigramBridgeTable { bridgeable }
}

#[inline(always)]
fn pack_pair(t1: u32, t2: u32) -> u64 {
    (t1 as u64) << 32 | t2 as u64
}

#[inline(always)]
fn fx_hash(key: u64) -> u64 {
    key.wrapping_mul(0x517cc1b727220a95)
}

/// FxHash-based [`BuildHasher`] for the token cache.
#[derive(Clone, Default)]
struct FxBuildHasher;

impl std::hash::BuildHasher for FxBuildHasher {
    type Hasher = FxStrHasher;
    fn build_hasher(&self) -> FxStrHasher {
        FxStrHasher(0)
    }
}

struct FxStrHasher(u64);

impl std::hash::Hasher for FxStrHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut state = self.0;
        let mut i = 0;
        while i + 8 <= bytes.len() {
            let word = u64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
            state = state.wrapping_add(word).wrapping_mul(0x517cc1b727220a95);
            i += 8;
        }
        while i < bytes.len() {
            state = state
                .wrapping_add(bytes[i] as u64)
                .wrapping_mul(0x517cc1b727220a95);
            i += 1;
        }
        self.0 = state;
    }
}

type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

const FLAT_CACHE_BITS: usize = 16;
const EMPTY_SLOT: u64 = 0;

#[derive(Clone, Copy)]
#[repr(C)]
struct CacheSlot {
    hash: u64,
    offset: u32,
    len: u16,
    key_len: u16,
    key_offset: u32,
}

struct FlatCache {
    bpe_id: usize,
    mask: usize,
    max_load: usize,
    slots: Vec<CacheSlot>,
    pool: Vec<u32>,
    key_pool: Vec<u8>,
    count: usize,
}

impl FlatCache {
    fn new() -> Self {
        Self::with_bits(FLAT_CACHE_BITS)
    }

    /// A flat cache with `1 << bits` slots. The thread-local L1 uses
    /// [`FLAT_CACHE_BITS`]; the shared-cache shards use fewer bits each.
    fn with_bits(bits: usize) -> Self {
        let size = 1usize << bits;
        Self {
            bpe_id: 0,
            mask: size - 1,
            max_load: size * 3 / 4,
            slots: vec![
                CacheSlot {
                    hash: EMPTY_SLOT,
                    offset: 0,
                    len: 0,
                    key_len: 0,
                    key_offset: 0,
                };
                size
            ],
            pool: Vec::new(),
            key_pool: Vec::new(),
            count: 0,
        }
    }

    fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.hash = EMPTY_SLOT;
        }
        self.pool.clear();
        self.key_pool.clear();
        self.count = 0;
    }

    #[inline(always)]
    fn hash_str(s: &str) -> u64 {
        let bytes = s.as_bytes();
        let mut h: u64 = bytes.len() as u64;
        let mut i = 0;
        while i + 8 <= bytes.len() {
            let word = u64::from_ne_bytes(bytes[i..i + 8].try_into().unwrap());
            h = h.wrapping_add(word).wrapping_mul(0x517cc1b727220a95);
            i += 8;
        }
        while i < bytes.len() {
            h = h
                .wrapping_add(bytes[i] as u64)
                .wrapping_mul(0x517cc1b727220a95);
            i += 1;
        }
        if h == EMPTY_SLOT {
            h = 1;
        }
        h
    }

    #[inline(always)]
    fn get(&self, key: &str, out: &mut Vec<u32>) -> bool {
        let hash = Self::hash_str(key);
        let key_bytes = key.as_bytes();
        let mut idx = hash as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            if slot.hash == hash {
                let ks = slot.key_offset as usize;
                let ke = ks + slot.key_len as usize;
                if unsafe { self.key_pool.get_unchecked(ks..ke) } == key_bytes {
                    let start = slot.offset as usize;
                    let end = start + slot.len as usize;
                    out.extend_from_slice(unsafe { self.pool.get_unchecked(start..end) });
                    return true;
                }
            }
            if slot.hash == EMPTY_SLOT {
                return false;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn insert(&mut self, key: &str, ids: &[u32]) {
        if self.count >= self.max_load {
            self.clear();
        }
        let hash = Self::hash_str(key);
        let key_bytes = key.as_bytes();
        let mut idx = hash as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            let h = slot.hash;
            if h == EMPTY_SLOT {
                let Ok(len) = u16::try_from(ids.len()) else {
                    return;
                };
                let Ok(key_len) = u16::try_from(key_bytes.len()) else {
                    return;
                };
                self.count += 1;
                let offset = self.pool.len() as u32;
                self.pool.extend_from_slice(ids);
                let key_offset = self.key_pool.len() as u32;
                self.key_pool.extend_from_slice(key_bytes);
                let slot = unsafe { self.slots.get_unchecked_mut(idx) };
                slot.hash = hash;
                slot.offset = offset;
                slot.len = len;
                slot.key_offset = key_offset;
                slot.key_len = key_len;
                return;
            }
            if h == hash {
                let ks = slot.key_offset as usize;
                let ke = ks + slot.key_len as usize;
                if unsafe { self.key_pool.get_unchecked(ks..ke) } == key_bytes {
                    let Ok(len) = u16::try_from(ids.len()) else {
                        return;
                    };
                    let offset = self.pool.len() as u32;
                    self.pool.extend_from_slice(ids);
                    let slot = unsafe { self.slots.get_unchecked_mut(idx) };
                    slot.offset = offset;
                    slot.len = len;
                    return;
                }
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

// ── Pretoken cache (fused scanner path) ──────────────────────────────────────
//
// On natural-text corpora the fused encode loop is overwhelmingly a cache hit
// (>90%), and the working set (unique pretokens) is far larger than L2/L3, so a
// lookup is a near-random memory access. The design (after gigatoken's
// `ShortPretokenCache`) makes each hit as cheap as possible:
//
// - The key is the pretoken's bytes packed into a `u128` (≤15 bytes, length in
//   the top byte), so a match is a single 128-bit integer compare — no separate
//   hashing of bytes plus a `memcmp` against a side pool.
// - Each entry is exactly 32 bytes and holds up to 3 token ids INLINE (≈98% of
//   pretokens encode to ≤2 tokens), so a hit reads one cache line and copies the
//   ids straight out — no second load into an id arena. Longer id sequences
//   spill to a pool (`v[0]` = offset).
// - The table GROWS (doubling) instead of clearing at load, so the hot set
//   survives on long-tail streaming — the previous clear-at-¾-load table threw
//   away frequent entries and re-ran BPE for them. It starts small (a single
//   short document keeps it tiny) and is capped, degrading to a clear only for
//   pathologically diverse input beyond the cap.
//
// Pretokens longer than 15 bytes (rare in natural text) are not cached here;
// they take the merge path directly.
const PRETOKEN_CACHE_MIN_BITS: usize = 12;
const PRETOKEN_CACHE_MAX_BITS: usize = 21; // ~2M entries × 32 B = 64 MiB cap
const PT_INLINE: usize = 3;

/// Prefetch the cache line at `p` into L1 (read hint). No memory effects, so
/// any address is safe; a no-op on architectures without a prefetch intrinsic.
#[inline(always)]
fn prefetch_read(p: *const u8) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: prefetch has no memory effects and reads nothing.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) p,
            options(nostack, preserves_flags, readonly),
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch has no memory effects and reads nothing.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}

#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct PtEntry {
    /// Packed pretoken bytes + length; `0` marks an empty slot.
    key: u128,
    /// Token count. `≤ PT_INLINE` → tokens in `v`; otherwise `v[0]` is the
    /// offset of `len` ids in the spill pool.
    len: u32,
    v: [u32; PT_INLINE],
}

const _: () = assert!(std::mem::size_of::<PtEntry>() == 32);

impl PtEntry {
    #[inline(always)]
    const fn empty() -> Self {
        Self {
            key: 0,
            len: 0,
            v: [0; PT_INLINE],
        }
    }
}

struct PretokenCache {
    bpe_id: usize,
    mask: usize,
    cap: usize,
    len: usize,
    slots: Vec<PtEntry>,
    spill: Vec<u32>,
}

impl PretokenCache {
    fn new() -> Self {
        let cap = 1usize << PRETOKEN_CACHE_MIN_BITS;
        Self {
            bpe_id: 0,
            mask: cap - 1,
            cap,
            len: 0,
            slots: vec![PtEntry::empty(); cap],
            spill: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for e in &mut self.slots {
            e.key = 0;
        }
        self.spill.clear();
        self.len = 0;
    }

    /// Pack ≤15 pretoken bytes + length into a nonzero `u128`, or `None` if the
    /// pretoken is empty or too long to cache inline.
    #[inline(always)]
    fn pack_key(bytes: &[u8]) -> Option<u128> {
        let n = bytes.len();
        if n == 0 || n > 15 {
            return None;
        }
        let mut buf = [0u8; 16];
        buf[..n].copy_from_slice(bytes);
        buf[15] = n as u8; // length tag in the top byte → key is never 0
        Some(u128::from_le_bytes(buf))
    }

    /// Hot-path packer for a pretoken at `buf[start..start+len]`, `1 ≤ len ≤ 15`.
    /// When ≥16 bytes remain it reads one unaligned `u128` and masks off the
    /// surplus bytes — avoiding the per-token variable-length `memmove` that
    /// `copy_from_slice` compiles to (measured ~30% of scan time). Near the
    /// buffer end it falls back to the byte copy. Result is identical to
    /// `pack_key(&buf[start..start+len]).unwrap()`.
    #[inline(always)]
    fn pack_key_at(buf: &[u8], start: usize, len: usize) -> u128 {
        debug_assert!((1..=15).contains(&len));
        if start + 16 <= buf.len() {
            // SAFETY: start + 16 <= len, so 16 bytes from `start` are in bounds.
            let raw = unsafe { (buf.as_ptr().add(start) as *const u128).read_unaligned() };
            let keep = (1u128 << (len * 8)) - 1; // low `len` bytes (len ≤ 15 → shift ≤ 120)
            (raw & keep) | ((len as u128) << 120)
        } else {
            let mut b = [0u8; 16];
            b[..len].copy_from_slice(&buf[start..start + len]);
            b[15] = len as u8;
            u128::from_le_bytes(b)
        }
    }

    #[inline(always)]
    fn hash(key: u128) -> u64 {
        // Fold the 128-bit key to 64 bits and mix with a single multiply. The
        // high half (high bytes + length tag) is rotated in so short keys —
        // whose bytes all sit in the low half — still spread across all bits.
        let lo = key as u64;
        let hi = (key >> 64) as u64;
        let mut h = (lo ^ hi.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
        h
    }

    /// Prefetch the cache line holding `hash`'s home slot. The fused encode
    /// loop issues this many spans before probing, so the (near-random, often
    /// DRAM-resident) line is resident by the time it is read.
    #[inline(always)]
    fn prefetch(&self, hash: u64) {
        let idx = hash as usize & self.mask;
        // SAFETY: idx <= mask < cap; prefetch reads nothing, any address is ok.
        prefetch_read(unsafe { self.slots.as_ptr().add(idx) } as *const u8);
    }

    #[inline(always)]
    fn get(&self, key: &str, out: &mut Vec<u32>) -> bool {
        match Self::pack_key(key.as_bytes()) {
            Some(k) => self.get_by_key(k, Self::hash(k), out),
            None => false,
        }
    }

    #[inline(always)]
    fn get_by_key(&self, k: u128, hash: u64, out: &mut Vec<u32>) -> bool {
        let mut idx = hash as usize & self.mask;
        loop {
            let e = unsafe { self.slots.get_unchecked(idx) };
            if e.key == k {
                let len = e.len as usize;
                if len <= PT_INLINE {
                    out.extend_from_slice(unsafe { e.v.get_unchecked(..len) });
                } else {
                    let o = e.v[0] as usize;
                    out.extend_from_slice(unsafe { self.spill.get_unchecked(o..o + len) });
                }
                return true;
            }
            if e.key == 0 {
                return false;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Branchless fast probe of the home slot: unconditionally copy that slot's
    /// inline ids into `out`'s spare capacity, then advance the length by the
    /// matched token count (a `cmov` — `0` when the home slot doesn't hold `k`
    /// with an inline value). The copy is not gated on the key compare, so the
    /// hit path has no branch-dependent load; a dead copy on a miss is simply
    /// overwritten. Returns whether it emitted the token; a `false` (miss,
    /// displaced key, or spilled value) falls to [`Self::get_by_key`].
    ///
    /// The caller must guarantee `out` has at least [`PT_INLINE`] spare slots.
    #[inline(always)]
    fn probe_emit_fast(&self, k: u128, hash: u64, out: &mut Vec<u32>) -> bool {
        let idx = hash as usize & self.mask;
        let e = unsafe { self.slots.get_unchecked(idx) };
        let hit = e.key == k && (e.len as usize) <= PT_INLINE;
        let w = out.len();
        // SAFETY: caller reserved >= PT_INLINE spare, so [w, w+PT_INLINE) is in
        // the allocation; the advance keeps len <= capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(e.v.as_ptr(), out.as_mut_ptr().add(w), PT_INLINE);
            out.set_len(w + if hit { e.len as usize } else { 0 });
        }
        hit
    }

    #[inline(always)]
    fn insert(&mut self, key: &str, ids: &[u32]) {
        if let Some(k) = Self::pack_key(key.as_bytes()) {
            self.insert_by_key(k, Self::hash(k), ids);
        }
    }

    #[inline(always)]
    fn insert_by_key(&mut self, k: u128, hash: u64, ids: &[u32]) {
        if (self.len + 1) * 4 > self.cap * 3 {
            self.grow_or_clear();
        }
        let e = Self::build_entry(&mut self.spill, k, ids);
        self.place_hashed(e, hash);
    }

    /// Build an entry, spilling ids past the inline capacity into `spill`.
    #[inline(always)]
    fn build_entry(spill: &mut Vec<u32>, k: u128, ids: &[u32]) -> PtEntry {
        let mut e = PtEntry {
            key: k,
            len: ids.len() as u32,
            v: [0; PT_INLINE],
        };
        if ids.len() <= PT_INLINE {
            e.v[..ids.len()].copy_from_slice(ids);
        } else {
            e.v[0] = spill.len() as u32;
            spill.extend_from_slice(ids);
        }
        e
    }

    /// Insert an already-built entry into its first empty slot (caller ensures
    /// the key is absent and there is room).
    #[inline(always)]
    fn place(&mut self, e: PtEntry) {
        let hash = Self::hash(e.key);
        self.place_hashed(e, hash);
    }

    #[inline(always)]
    fn place_hashed(&mut self, e: PtEntry, hash: u64) {
        let mut idx = hash as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked_mut(idx) };
            if slot.key == 0 {
                *slot = e;
                self.len += 1;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[cold]
    fn grow_or_clear(&mut self) {
        if self.cap >= (1usize << PRETOKEN_CACHE_MAX_BITS) {
            self.clear();
            return;
        }
        let new_cap = self.cap * 2;
        let old = std::mem::replace(&mut self.slots, vec![PtEntry::empty(); new_cap]);
        self.cap = new_cap;
        self.mask = new_cap - 1;
        self.len = 0;
        // Spill offsets are preserved across a grow (the pool is untouched).
        for e in old {
            if e.key != 0 {
                self.place(e);
            }
        }
    }
}

#[inline(always)]
fn append_encoded_symbol(
    id: u32,
    small_ids: &mut [u32; SMALL_MERGE_MAX],
    small_n: &mut usize,
    heap_mode: &mut bool,
    scratch: &mut MergeScratch,
) {
    if !*heap_mode {
        if *small_n < SMALL_MERGE_MAX {
            small_ids[*small_n] = id;
            *small_n += 1;
            return;
        }

        // The 33rd symbol crosses the stack branch's bound. Transfer the
        // already validated prefix once, then keep collecting in the existing
        // linked-list representation used by the heap merger.
        for (i, &c) in small_ids[..*small_n].iter().enumerate() {
            scratch.symbols.push(MergeSymbol {
                c,
                prev: if i == 0 { -1 } else { (i - 1) as i32 },
                next: -1,
            });
            if i > 0 {
                scratch.symbols[i - 1].next = i as i32;
            }
        }
        *heap_mode = true;
    }

    let i = scratch.symbols.len();
    scratch.symbols.push(MergeSymbol {
        c: id,
        prev: if i == 0 { -1 } else { (i - 1) as i32 },
        next: -1,
    });
    if i > 0 {
        scratch.symbols[i - 1].next = i as i32;
    }
}

thread_local! {
    static TL_BPE_CACHE: RefCell<FlatCache> = RefCell::new(FlatCache::new());
    static TL_FUSED_CACHE: RefCell<PretokenCache> = RefCell::new(PretokenCache::new());
}

const CACHE_SHARDS: usize = 64;

/// Slot bits per shared-cache shard: 64 shards x 4096 slots = 256k entries.
const SHARED_SHARD_BITS: usize = 12;

/// Cross-thread token cache: [`CACHE_SHARDS`] mutex-guarded [`FlatCache`]
/// shards. Versus the previous `HashMap<String, Vec<u32>>` shards this is
/// allocation-free per insert (keys and ids are copied into per-shard pools
/// that retain capacity across clears) and bounded (a shard clears at 3/4 load
/// rather than growing forever) — removing the two heap allocations per cold
/// pretoken and fixing unbounded growth on diverse long-running traffic.
struct SharedCache {
    shards: Vec<Mutex<FlatCache>>,
}

impl SharedCache {
    fn new() -> Self {
        Self {
            shards: (0..CACHE_SHARDS)
                .map(|_| Mutex::new(FlatCache::with_bits(SHARED_SHARD_BITS)))
                .collect(),
        }
    }

    /// Shard selector using the TOP bits of the same hash a [`FlatCache`] uses
    /// (low bits) to index slots, so the two are independent.
    #[inline]
    fn shard_index(key: &str) -> usize {
        (FlatCache::hash_str(key) >> (64 - 6)) as usize & (CACHE_SHARDS - 1)
    }

    #[inline]
    fn get_into(&self, key: &str, out: &mut Vec<u32>) -> bool {
        self.shards[Self::shard_index(key)]
            .lock()
            .unwrap()
            .get(key, out)
    }

    #[inline]
    fn insert(&self, key: &str, ids: &[u32]) {
        self.shards[Self::shard_index(key)]
            .lock()
            .unwrap()
            .insert(key, ids);
    }
}

/// Raw deserialization helper.
#[derive(Deserialize)]
struct RawBpe {
    #[serde(default)]
    vocab: Vocab,
    #[serde(default)]
    merges: Vec<Value>,
    #[allow(dead_code)]
    dropout: Option<f64>,
    #[allow(dead_code)]
    unk_token: Option<String>,
    #[allow(dead_code)]
    continuing_subword_prefix: Option<String>,
    #[allow(dead_code)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    fuse_unk: bool,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
}

/// Monotonic counter for unique Bpe instance IDs.
static BPE_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// Entry in the BPE merge priority queue.
/// `key = (rank << 32) | pos`, `val = (left_c << 32) | right_c`.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(C)]
struct MergeEntry {
    key: u64,
    val: u64,
}

impl MergeEntry {
    #[inline(always)]
    fn new(rank: u32, pos: u32, left_c: u32, right_c: u32) -> Self {
        Self {
            key: (rank as u64) << 32 | pos as u64,
            val: (left_c as u64) << 32 | right_c as u64,
        }
    }

    #[inline(always)]
    fn pos(&self) -> u32 {
        self.key as u32
    }

    #[inline(always)]
    fn left_c(&self) -> u32 {
        (self.val >> 32) as u32
    }

    #[inline(always)]
    fn right_c(&self) -> u32 {
        self.val as u32
    }
}

impl Ord for MergeEntry {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for MergeEntry {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Symbol in the merge linked list.
#[derive(Clone, Copy)]
struct MergeSymbol {
    c: u32,
    prev: i32,
    next: i32,
}

struct MergeScratch {
    symbols: Vec<MergeSymbol>,
    heap: BinaryHeap<Reverse<MergeEntry>>,
    heap_buf: Vec<Reverse<MergeEntry>>,
}

impl MergeScratch {
    fn new() -> Self {
        Self {
            symbols: Vec::new(),
            heap: BinaryHeap::new(),
            heap_buf: Vec::new(),
        }
    }
}

thread_local! {
    static TL_MERGE_SCRATCH: RefCell<MergeScratch> = RefCell::new(MergeScratch::new());
}

/// Interleaved slot for the ranked merge map (16 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct RankedMergeSlot {
    key: u64,
    rank: u32,
    id: u32,
}

/// Open-addressing hash table storing `(left_id, right_id) → (rank, merged_id)`.
#[derive(Clone)]
struct RankedMergeMap {
    mask: usize,
    slots: Vec<RankedMergeSlot>,
}

impl RankedMergeMap {
    fn from_parsed(parsed: &ParsedMergeMap) -> Self {
        if parsed.is_empty() {
            return Self {
                mask: 0,
                slots: Vec::new(),
            };
        }
        let capacity = (parsed.len() * 2).next_power_of_two();
        let mask = capacity - 1;
        let mut slots = vec![
            RankedMergeSlot {
                key: EMPTY_KEY,
                rank: 0,
                id: 0
            };
            capacity
        ];

        for (&(t1, t2), &(rank, merged_id)) in parsed {
            let key = pack_pair(t1, t2);
            let mut idx = fx_hash(key) as usize & mask;
            loop {
                if slots[idx].key == EMPTY_KEY {
                    slots[idx] = RankedMergeSlot {
                        key,
                        rank,
                        id: merged_id,
                    };
                    break;
                }
                idx = (idx + 1) & mask;
            }
        }

        Self { mask, slots }
    }

    /// Look up the rank and merged token ID for a pair.
    #[inline(always)]
    fn get(&self, t1: u32, t2: u32) -> Option<(u32, u32)> {
        if self.slots.is_empty() {
            return None;
        }
        let key = pack_pair(t1, t2);
        let mut idx = fx_hash(key) as usize & self.mask;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            if slot.key == key {
                return Some((slot.rank, slot.id));
            }
            if slot.key == EMPTY_KEY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

/// CSR adjacency structure for merge pair discovery.
#[derive(Clone)]
struct MergeAdjacency {
    offsets: Vec<u32>,
    data: Vec<(u32, u32, u32)>, // (neighbor, rank, new_id)
}

impl MergeAdjacency {
    fn from_parsed(parsed: &ParsedMergeMap, vocab_size: usize) -> Self {
        let mut counts = vec![0u32; vocab_size];
        for &(left, _right) in parsed.keys() {
            counts[left as usize] += 1;
        }

        let mut offsets = Vec::with_capacity(vocab_size + 1);
        offsets.push(0u32);
        let mut running = 0u32;
        for &c in &counts {
            running += c;
            offsets.push(running);
        }

        let mut data = vec![(0u32, 0u32, 0u32); running as usize];
        let mut write_pos = offsets[..vocab_size].to_vec();
        for (&(left, right), &(rank, merged_id)) in parsed {
            let idx = write_pos[left as usize] as usize;
            data[idx] = (right, rank, merged_id);
            write_pos[left as usize] += 1;
        }

        for i in 0..vocab_size {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            data[start..end].sort_unstable_by_key(|&(neighbor, _, _)| neighbor);
        }

        Self { offsets, data }
    }

    #[inline(always)]
    fn get(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let start = unsafe { *self.offsets.get_unchecked(left as usize) } as usize;
        let end = unsafe { *self.offsets.get_unchecked(left as usize + 1) } as usize;
        let slice = unsafe { self.data.get_unchecked(start..end) };
        match slice.binary_search_by_key(&right, |&(n, _, _)| n) {
            Ok(idx) => {
                let entry = unsafe { slice.get_unchecked(idx) };
                Some((entry.1, entry.2))
            }
            Err(_) => None,
        }
    }
}

#[derive(Deserialize)]
#[serde(try_from = "RawBpe")]
pub struct Bpe {
    #[serde(skip)]
    id: usize,
    daac: DoubleArrayAhoCorasick<TokenId>,
    merge_map: MergeMap,
    unmerge_map: Vec<(TokenId, TokenId)>,
    next_prefix_map: Vec<TokenId>,
    token_lens: Vec<u16>,
    shared_cache: SharedCache,
    id_to_token: Vec<String>,
    token_to_id: FxHashMap<String, u32>,
    byte_to_initial_token: [u32; 256],
    byte_fallback_token_ids: [u32; 256],
    /// Token id for each single ASCII-character string (`INVALID_TOKEN` when
    /// absent). Fast path for the char-based merge engine, avoiding a HashMap
    /// probe per character.
    single_char_token: [u32; 128],
    ranked_merge_map: RankedMergeMap,
    byte_pair_initial: Vec<(u32, u32)>,
    merge_adj: MergeAdjacency,
    ignore_merges: bool,
    byte_fallback: bool,
    pub bigram_bridge_table: BigramBridgeTable,
}

impl TryFrom<RawBpe> for Bpe {
    type Error = String;

    fn try_from(raw: RawBpe) -> Result<Self> {
        let merge_map = parse_merges(&raw.vocab, &raw.merges)?;
        let mut bpe = Self::new(&raw.vocab, merge_map)?;
        bpe.ignore_merges = raw.ignore_merges;
        bpe.byte_fallback = raw.byte_fallback;
        Ok(bpe)
    }
}

enum Decomposition {
    Pair(TokenId, TokenId),
    CharsNotInVocab,
    Stuck,
}

fn encoding_decomposition(text: &str, vocab: &Vocab, merge_map: &ParsedMergeMap) -> Decomposition {
    let mut tokens: Vec<TokenId> = Vec::new();
    for ch in text.chars() {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        match vocab.get(s) {
            Some(&tid) => tokens.push(tid),
            None => return Decomposition::CharsNotInVocab,
        }
    }

    if tokens.len() < 2 {
        return Decomposition::CharsNotInVocab;
    }

    while tokens.len() > 2 {
        let mut best_rank = u32::MAX;
        let mut best_pos = usize::MAX;
        let mut best_new = 0;
        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i], tokens[i + 1]);
            if let Some(&(rank, new_id)) = merge_map.get(&pair)
                && rank < best_rank
            {
                best_rank = rank;
                best_pos = i;
                best_new = new_id;
            }
        }
        if best_pos == usize::MAX {
            return Decomposition::Stuck;
        }
        tokens[best_pos] = best_new;
        tokens.remove(best_pos + 1);
    }

    Decomposition::Pair(tokens[0], tokens[1])
}

fn parse_merges(vocab: &Vocab, merges: &[Value]) -> Result<ParsedMergeMap> {
    let mut merge_map = ParsedMergeMap::new();
    for (rank, entry) in merges.iter().enumerate() {
        let (left, right) = parse_merge_entry(entry)?;
        let &left_id = vocab
            .get(left)
            .ok_or_else(|| format!("merge token not in vocab: {left:?}"))?;
        let &right_id = vocab
            .get(right)
            .ok_or_else(|| format!("merge token not in vocab: {right:?}"))?;
        let merged = format!("{left}{right}");
        let &merged_id = vocab
            .get(&merged)
            .ok_or_else(|| format!("merged token not in vocab: {merged:?}"))?;
        merge_map.insert((left_id, right_id), (rank as u32, merged_id));
    }
    Ok(merge_map)
}

fn parse_merge_entry(entry: &Value) -> Result<(&str, &str)> {
    match entry {
        Value::String(s) => {
            let (left, right) = s
                .split_once(' ')
                .ok_or_else(|| format!("invalid merge entry (no space): {s:?}"))?;
            Ok((left, right))
        }
        Value::Array(arr) if arr.len() == 2 => {
            let left = arr[0]
                .as_str()
                .ok_or_else(|| format!("merge element not a string: {:?}", arr[0]))?;
            let right = arr[1]
                .as_str()
                .ok_or_else(|| format!("merge element not a string: {:?}", arr[1]))?;
            Ok((left, right))
        }
        _ => Err(format!("unrecognized merge entry format: {entry:?}")),
    }
}

/// Split a token's bytes into the two lower-ranked pieces that merge to form
/// it, using the tiktoken byte-pair-merge algorithm capped at `max_rank`.
///
/// `boundaries` is scratch space reused across calls. Returns the byte offset
/// of the split point, i.e. the pieces are `bytes[..mid]` and `bytes[mid..]`.
fn tiktoken_split(
    byte_ranks: &HashMap<&[u8], u32>,
    bytes: &[u8],
    max_rank: u32,
    boundaries: &mut Vec<usize>,
) -> Result<usize> {
    boundaries.clear();
    boundaries.extend(0..=bytes.len());

    // Repeatedly merge the lowest-ranked adjacent pair whose rank is below
    // this token's own rank, exactly as tiktoken's `_byte_pair_merge` does.
    loop {
        let mut best_rank = u32::MAX;
        let mut best = usize::MAX;
        for i in 0..boundaries.len().saturating_sub(2) {
            let pair = &bytes[boundaries[i]..boundaries[i + 2]];
            if let Some(&rank) = byte_ranks.get(pair)
                && rank < max_rank
                && rank < best_rank
            {
                best_rank = rank;
                best = i;
            }
        }
        if best == usize::MAX {
            break;
        }
        boundaries.remove(best + 1);
    }

    if boundaries.len() != 3 {
        return Err(format!(
            "tiktoken token did not decompose into 2 pieces (got {}): {bytes:?}",
            boundaries.len() - 1
        ));
    }
    Ok(boundaries[1])
}

impl Bpe {
    pub fn new(vocab: &Vocab, merge_map: ParsedMergeMap) -> Result<Self> {
        if vocab.is_empty() {
            return Err("cannot build Bpe with empty vocabulary".into());
        }

        let vocab_r: std::collections::BTreeMap<u32, &str> =
            vocab.iter().map(|(s, &id)| (id, s.as_str())).collect();

        let id_to_token: Vec<String> = (0..=*vocab_r.keys().max().unwrap())
            .map(|t| {
                vocab_r
                    .get(&t)
                    .ok_or_else(|| format!("non-contiguous tokens - token {t} is missing"))
                    .map(|s| s.to_string())
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let max_token = vocab_r.keys().max().copied().unwrap();

        let mut unmerge_map = (0..=max_token).map(|t| (t, t)).collect::<Vec<_>>();
        let mut is_orphan = vec![false; (max_token + 1) as usize];
        for (&tid, text) in &vocab_r {
            if text.chars().count() < 2 {
                continue;
            }
            match encoding_decomposition(text, vocab, &merge_map) {
                Decomposition::Pair(left, right) => {
                    unmerge_map[tid as usize] = (left, right);
                }
                Decomposition::Stuck => {
                    is_orphan[tid as usize] = true;
                }
                Decomposition::CharsNotInVocab => {}
            }
        }

        let daac = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::LeftmostLongest)
            .build_with_values(vocab_r.iter().filter_map(|(&token, pattern)| {
                (!is_orphan[token as usize]).then_some((pattern, token))
            }))
            .map_err(|e| format!("error building DAAC: {e}"))?;

        let token_lens: Vec<u16> = (0..=max_token)
            .map(|t| {
                u16::try_from(vocab_r[&t].len())
                    .map_err(|_| format!("token {t} length {} exceeds u16::MAX", vocab_r[&t].len()))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let next_prefix_map: Vec<TokenId> = (0..=max_token)
            .map(|token| {
                let token_str = &vocab_r[&token];
                let Some((last_char_start, _)) = token_str.char_indices().next_back() else {
                    return INVALID_TOKEN;
                };
                if last_char_start == 0 {
                    return INVALID_TOKEN;
                }
                daac.leftmost_find_iter(&token_str[..last_char_start])
                    .next()
                    .map_or(INVALID_TOKEN, |m| m.value())
            })
            .collect();

        let flat_merge_map = MergeMap::from_parsed(&merge_map);
        let ranked_merge_map = RankedMergeMap::from_parsed(&merge_map);

        let mut byte_to_initial_token = [INVALID_TOKEN; 256];
        for byte_val in 0u16..256 {
            let ch = BYTE_TO_CHAR[byte_val as usize];
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            if let Some(&id) = vocab.get(s) {
                byte_to_initial_token[byte_val as usize] = id;
            }
        }

        let mut byte_fallback_token_ids = [INVALID_TOKEN; 256];
        for byte_val in 0u16..256 {
            let token = format!("<0x{byte_val:02X}>");
            if let Some(&id) = vocab.get(token.as_str()) {
                byte_fallback_token_ids[byte_val as usize] = id;
            }
        }

        // Pre-compute initial byte-pair merges (256×256 table).
        let mut byte_pair_initial = vec![(u32::MAX, 0u32); 65536];
        for b1 in 0u16..256 {
            let t1 = byte_to_initial_token[b1 as usize];
            if t1 == INVALID_TOKEN {
                continue;
            }
            for b2 in 0u16..256 {
                let t2 = byte_to_initial_token[b2 as usize];
                if t2 == INVALID_TOKEN {
                    continue;
                }
                if let Some((rank, new_id)) = ranked_merge_map.get(t1, t2) {
                    byte_pair_initial[b1 as usize * 256 + b2 as usize] = (rank, new_id);
                }
            }
        }

        let mut single_char_token = [INVALID_TOKEN; 128];
        for (byte, slot) in single_char_token.iter_mut().enumerate() {
            let ch = byte as u8 as char;
            let mut buf = [0u8; 1];
            if let Some(&id) = vocab.get(ch.encode_utf8(&mut buf) as &str) {
                *slot = id;
            }
        }

        let vocab_size = id_to_token.len();
        let merge_adj = MergeAdjacency::from_parsed(&merge_map, vocab_size);

        let bigram_bridge_table = build_bigram_bridge_table(&id_to_token);

        Ok(Self {
            id: BPE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            daac,
            merge_map: flat_merge_map,
            unmerge_map,
            next_prefix_map,
            token_lens,
            shared_cache: SharedCache::new(),
            id_to_token,
            token_to_id: {
                let mut m = HashMap::with_capacity_and_hasher(vocab.len(), FxBuildHasher);
                m.extend(vocab.iter().map(|(k, v)| (k.clone(), *v)));
                m
            },
            byte_to_initial_token,
            byte_fallback_token_ids,
            single_char_token,
            ranked_merge_map,
            byte_pair_initial,
            merge_adj,
            ignore_merges: false,
            byte_fallback: false,
            bigram_bridge_table,
        })
    }

    /// Build a [`Bpe`] from tiktoken mergeable ranks (`token_bytes -> rank`).
    ///
    /// The ranks are converted into the byte-level BPE representation used
    /// internally: each token's bytes are mapped through the GPT-2
    /// byte-to-unicode table to form the vocab key, and the merge list is
    /// regenerated from the ranks (splitting each multi-byte token into the two
    /// lower-ranked pieces that form it, exactly as tiktoken does). The rank
    /// serves as both the token id and the merge priority.
    pub fn from_tiktoken_ranks(ranks: &[(Vec<u8>, u32)]) -> Result<Self> {
        if ranks.is_empty() {
            return Err("cannot build Bpe from empty tiktoken ranks".into());
        }

        // Fast raw-byte-sequence -> rank lookup for merge generation.
        let mut byte_ranks: HashMap<&[u8], u32> = HashMap::with_capacity(ranks.len());
        for (bytes, rank) in ranks {
            byte_ranks.insert(bytes.as_slice(), *rank);
        }

        // Byte-level vocab: map each token's bytes through the GPT-2 table so
        // the representation matches HuggingFace byte-level BPE tokenizers.
        let mut vocab: Vocab = HashMap::with_capacity(ranks.len());
        for (bytes, rank) in ranks {
            let mut key = String::with_capacity(bytes.len());
            for &b in bytes {
                key.push(BYTE_TO_CHAR[b as usize]);
            }
            vocab.insert(key, *rank);
        }

        // Regenerate the merge list from the ranks.
        let mut merge_map = ParsedMergeMap::with_capacity(ranks.len());
        let mut boundaries: Vec<usize> = Vec::new();
        for (bytes, rank) in ranks {
            if bytes.len() < 2 {
                continue;
            }
            let mid = tiktoken_split(&byte_ranks, bytes, *rank, &mut boundaries)?;
            let (left, right) = bytes.split_at(mid);
            let (Some(&left_id), Some(&right_id)) = (byte_ranks.get(left), byte_ranks.get(right))
            else {
                return Err(format!(
                    "tiktoken token {bytes:?} split into pieces not present in the vocabulary"
                ));
            };
            merge_map.insert((left_id, right_id), (*rank, *rank));
        }

        Self::new(&vocab, merge_map)
    }

    pub fn is_compatible_token_pair(&self, mut t1: TokenId, mut t2: TokenId) -> bool {
        if t1 == INVALID_TOKEN {
            return false;
        }

        let mut limit = u32::MAX;
        loop {
            if let Some(t) = self.merge_map.get(t1, t2)
                && t < limit
            {
                return false;
            }

            if t1 > t2 {
                limit = t1;
                t1 = self.unmerge_map[t1 as usize].1;
                if t1 == limit {
                    limit = t2 + 1;
                    t2 = self.unmerge_map[t2 as usize].0;
                    if t2 + 1 == limit {
                        return true;
                    }
                }
            } else {
                limit = t2 + 1;
                t2 = self.unmerge_map[t2 as usize].0;
                if t2 + 1 == limit {
                    limit = t1;
                    t1 = self.unmerge_map[t1 as usize].1;
                    if t1 == limit {
                        return true;
                    }
                }
            }
        }
    }

    fn next_match(&self, input: &str) -> Option<TokenId> {
        let m = self.daac.leftmost_find_iter(input).next()?;
        (m.start() == 0).then(|| m.value())
    }

    pub fn tokenize(&self, input: &str) -> Result<Vec<TokenId>> {
        let mut out = Vec::new();
        self.tokenize_into(input, &mut out)?;
        Ok(out)
    }

    #[inline(always)]
    pub fn tokenize_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        if let Some(token) = self.next_match(input)
            && self.token_lens[token as usize] as usize == input.len()
        {
            out.push(token);
            return Ok(());
        }

        let bpe_id = self.id;
        let hit = TL_BPE_CACHE.with(|c| {
            let c = c.borrow();
            if c.bpe_id != bpe_id {
                return false;
            }
            c.get(input, out)
        });
        if hit {
            return Ok(());
        }

        let start = out.len();
        if self.shared_cache.get_into(input, out) {
            TL_BPE_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.bpe_id != bpe_id {
                    c.bpe_id = bpe_id;
                    c.clear();
                }
                c.insert(input, &out[start..]);
            });
            return Ok(());
        }

        self.merge_all_encoded_into(input, out)?;

        let ids = &out[start..];
        TL_BPE_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                c.bpe_id = bpe_id;
                c.clear();
            }
            c.insert(input, ids);
        });
        self.shared_cache.insert(input, ids);

        Ok(())
    }

    #[inline(always)]
    fn encoded_small_candidate(input: &str) -> bool {
        input.len() <= SMALL_MERGE_MAX
    }

    /// BPE merge on already-encoded (ByteLevel) text. Dispatches short
    /// candidates to the stack merger and retains the priority-queue merger for
    /// the other inputs.
    #[inline(always)]
    fn merge_all_encoded_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }
        if Self::encoded_small_candidate(input) {
            self.merge_all_encoded_small_into(input, out)
        } else {
            self.merge_all_encoded_heap_into(input, out)
        }
    }

    /// Stack merger for encoded inputs whose initial-symbol count can fit in
    /// the bounded representation. The collector promotes byte-fallback
    /// expansions that cross the bound to the existing heap representation.
    fn merge_all_encoded_small_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();

            let mut small_ids = [0u32; SMALL_MERGE_MAX];
            let mut small_n = 0usize;
            let mut heap_mode = false;

            for ch in input.chars() {
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                let found = if ch.is_ascii() {
                    let id = self.single_char_token[ch as usize];
                    (id != INVALID_TOKEN).then_some(id)
                } else {
                    self.token_to_id.get(encoded).copied()
                };
                if let Some(id) = found {
                    append_encoded_symbol(
                        id,
                        &mut small_ids,
                        &mut small_n,
                        &mut heap_mode,
                        &mut scratch,
                    );
                    continue;
                }

                if !self.byte_fallback {
                    return Err(format!("character {ch:?} not in vocabulary"));
                }

                for &byte in encoded.as_bytes() {
                    let id = self.byte_fallback_token_ids[byte as usize];
                    if id == INVALID_TOKEN {
                        return Err(format!(
                            "byte fallback token <0x{byte:02X}> not in vocabulary"
                        ));
                    }
                    append_encoded_symbol(
                        id,
                        &mut small_ids,
                        &mut small_n,
                        &mut heap_mode,
                        &mut scratch,
                    );
                }
            }

            if !heap_mode {
                if small_n == 1 {
                    out.push(small_ids[0]);
                } else {
                    self.merge_small_encoded(&mut small_ids, small_n, out);
                }
                return Ok(());
            }

            let n = scratch.symbols.len();
            self.init_merge_heap(&mut scratch, n);
            self.run_merge_loop(&mut scratch, out);
            Ok(())
        })
    }

    /// Reference priority-queue BPE merge on already-encoded (ByteLevel) text.
    /// It remains the fallback for long inputs and the correctness oracle for
    /// the stack merger's tests.
    fn merge_all_encoded_heap_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();

            let mut n = 0usize;
            for ch in input.chars() {
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                let found = if ch.is_ascii() {
                    let id = self.single_char_token[ch as usize];
                    (id != INVALID_TOKEN).then_some(id)
                } else {
                    self.token_to_id.get(encoded).copied()
                };
                if let Some(id) = found {
                    scratch.symbols.push(MergeSymbol {
                        c: id,
                        prev: if n == 0 { -1 } else { (n - 1) as i32 },
                        next: -1,
                    });
                    if n > 0 {
                        scratch.symbols[n - 1].next = n as i32;
                    }
                    n += 1;
                    continue;
                }

                if !self.byte_fallback {
                    return Err(format!("character {ch:?} not in vocabulary"));
                }

                for &byte in encoded.as_bytes() {
                    let id = self.byte_fallback_token_ids[byte as usize];
                    if id == INVALID_TOKEN {
                        return Err(format!(
                            "byte fallback token <0x{byte:02X}> not in vocabulary"
                        ));
                    }
                    scratch.symbols.push(MergeSymbol {
                        c: id,
                        prev: if n == 0 { -1 } else { (n - 1) as i32 },
                        next: -1,
                    });
                    if n > 0 {
                        scratch.symbols[n - 1].next = n as i32;
                    }
                    n += 1;
                }
            }

            if n == 1 {
                out.push(scratch.symbols[0].c);
                return Ok(());
            }

            self.init_merge_heap(&mut scratch, n);
            self.run_merge_loop(&mut scratch, out);
            Ok(())
        })
    }

    /// Linear-scan BPE merge for short encoded pretokens (`n <=
    /// SMALL_MERGE_MAX` initial symbols). Avoids the `BinaryHeap` entirely: a
    /// stack-resident doubly-linked list plus a per-position rank array,
    /// find-min by a short scan over stack `u32`s, merge (O(1) pointer update),
    /// then refresh only the two neighbor pairs.
    ///
    /// Produces the identical token sequence as [`Self::run_merge_loop`]: both
    /// process the globally lowest-`(rank, pos)` active pair each step (the
    /// heap's `MergeEntry` key is `(rank << 32) | pos`; the scan's strict `<`
    /// keeps the leftmost/lowest-`pos` position on ties). Enforced by the
    /// `merge_small_matches_heap` differential test. `ids[..n]` are the
    /// initial encoded-symbol token ids.
    fn merge_small_encoded(&self, ids: &mut [u32; SMALL_MERGE_MAX], n: usize, out: &mut Vec<u32>) {
        let mut next = [0u8; SMALL_MERGE_MAX];
        let mut prev = [0u8; SMALL_MERGE_MAX];
        let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
        let mut new_ids = [0u32; SMALL_MERGE_MAX];
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = (i as u8).wrapping_sub(1); // prev[0] = 255 (>= n): sentinel
        }
        for i in 0..n - 1 {
            if let Some((rank, new_id)) = self.merge_adj.get(ids[i], ids[i + 1]) {
                ranks[i] = rank;
                new_ids[i] = new_id;
            }
        }
        loop {
            let mut best = u32::MAX;
            let mut best_i = 0usize;
            for (i, &rank) in ranks[..n - 1].iter().enumerate() {
                if rank < best {
                    best = rank;
                    best_i = i;
                }
            }
            if best == u32::MAX {
                break;
            }
            let i = best_i;
            ids[i] = new_ids[i];
            let dead = next[i] as usize;
            let new_right = next[dead] as usize;
            next[i] = new_right as u8;
            ranks[dead] = u32::MAX;
            if new_right < n {
                prev[new_right] = i as u8;
                match self.merge_adj.get(ids[i], ids[new_right]) {
                    Some((rank, new_id)) => {
                        ranks[i] = rank;
                        new_ids[i] = new_id;
                    }
                    None => ranks[i] = u32::MAX,
                }
            } else {
                ranks[i] = u32::MAX;
            }
            let left = prev[i] as usize;
            if left < n {
                match self.merge_adj.get(ids[left], ids[i]) {
                    Some((rank, new_id)) => {
                        ranks[left] = rank;
                        new_ids[left] = new_id;
                    }
                    None => ranks[left] = u32::MAX,
                }
            }
        }
        let mut i = 0usize;
        while i < n {
            out.push(ids[i]);
            i = next[i] as usize;
        }
    }

    /// Linear-scan BPE merge for short raw pretokens (`n <= SMALL_MERGE_MAX`
    /// bytes). `ids[..n]` are the per-byte initial token ids.
    fn merge_small_raw(
        &self,
        bytes: &[u8],
        ids: &mut [u32; SMALL_MERGE_MAX],
        n: usize,
        out: &mut Vec<u32>,
    ) {
        let mut next = [0u8; SMALL_MERGE_MAX];
        let mut prev = [0u8; SMALL_MERGE_MAX];
        let mut ranks = [u32::MAX; SMALL_MERGE_MAX];
        let mut new_ids = [0u32; SMALL_MERGE_MAX];
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = (i as u8).wrapping_sub(1); // prev[0] = 255 (>= n): sentinel
        }
        // Round-1 ranks via the dense byte-pair table: one direct-indexed load
        // per pair instead of a CSR neighbor scan.
        for i in 0..n - 1 {
            let (rank, new_id) =
                self.byte_pair_initial[bytes[i] as usize * 256 + bytes[i + 1] as usize];
            if rank != u32::MAX {
                ranks[i] = rank;
                new_ids[i] = new_id;
            }
        }
        loop {
            let mut best = u32::MAX;
            let mut best_i = 0usize;
            for (i, &rank) in ranks[..n - 1].iter().enumerate() {
                if rank < best {
                    best = rank;
                    best_i = i;
                }
            }
            if best == u32::MAX {
                break;
            }
            let i = best_i;
            ids[i] = new_ids[i];
            let dead = next[i] as usize;
            let new_right = next[dead] as usize;
            next[i] = new_right as u8;
            ranks[dead] = u32::MAX;
            if new_right < n {
                prev[new_right] = i as u8;
                match self.merge_adj.get(ids[i], ids[new_right]) {
                    Some((rank, new_id)) => {
                        ranks[i] = rank;
                        new_ids[i] = new_id;
                    }
                    None => ranks[i] = u32::MAX,
                }
            } else {
                ranks[i] = u32::MAX;
            }
            let left = prev[i] as usize;
            if left < n {
                match self.merge_adj.get(ids[left], ids[i]) {
                    Some((rank, new_id)) => {
                        ranks[left] = rank;
                        new_ids[left] = new_id;
                    }
                    None => ranks[left] = u32::MAX,
                }
            }
        }
        let mut i = 0usize;
        while i < n {
            out.push(ids[i]);
            i = next[i] as usize;
        }
    }

    /// `ignore_merges` whole-pretoken lookup: is the ByteLevel-encoded form of
    /// the entire pretoken a single vocab token? Encodes into a stack buffer
    /// for short pretokens (`BYTE_TO_CHAR` codepoints are <= U+0143, so <= 2
    /// UTF-8 bytes per input byte) instead of allocating a `String` per cold
    /// pretoken; falls back to a heap `String` only for long ones.
    #[inline]
    fn whole_pretoken_id(&self, raw: &str) -> Option<u32> {
        const STACK: usize = 128;
        let bytes = raw.as_bytes();
        if bytes.len() * 2 <= STACK {
            let mut buf = [0u8; STACK];
            let mut n = 0;
            for &b in bytes {
                n += BYTE_TO_CHAR[b as usize].encode_utf8(&mut buf[n..]).len();
            }
            // SAFETY: buf[..n] is a concatenation of `char::encode_utf8`
            // outputs, hence valid UTF-8.
            let encoded = unsafe { std::str::from_utf8_unchecked(&buf[..n]) };
            self.token_to_id.get(encoded).copied()
        } else {
            let mut encoded = String::with_capacity(bytes.len() * 2);
            for &b in bytes {
                encoded.push(BYTE_TO_CHAR[b as usize]);
            }
            self.token_to_id.get(encoded.as_str()).copied()
        }
    }

    /// BPE merge on raw (pre-ByteLevel) bytes. Short pretokens (the common
    /// case) use the stack-resident linear scan; longer ones use the heap.
    fn merge_all_raw_into(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        if raw_input.is_empty() {
            return Ok(());
        }

        let bytes = raw_input.as_bytes();
        if bytes.len() <= SMALL_MERGE_MAX {
            let n = bytes.len();
            let mut ids = [0u32; SMALL_MERGE_MAX];
            for (i, &byte) in bytes.iter().enumerate() {
                let id = self.byte_to_initial_token[byte as usize];
                if id == INVALID_TOKEN {
                    return Err(format!("byte 0x{byte:02x} has no token in vocabulary"));
                }
                ids[i] = id;
            }
            if n == 1 {
                out.push(ids[0]);
            } else {
                self.merge_small_raw(bytes, &mut ids, n, out);
            }
            return Ok(());
        }

        self.merge_all_raw_heap_into(raw_input, out)
    }

    /// Reference priority-queue BPE merge on raw (pre-ByteLevel) bytes, used for
    /// long pretokens and as the correctness oracle for [`Self::merge_small_raw`].
    fn merge_all_raw_heap_into(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();
            scratch.heap_buf.clear();

            let bytes = raw_input.as_bytes();
            let n = bytes.len();
            let mut prev_byte = 0u8;
            for (i, &byte) in bytes.iter().enumerate() {
                let id = self.byte_to_initial_token[byte as usize];
                if id == INVALID_TOKEN {
                    return Err(format!("byte 0x{byte:02x} has no token in vocabulary"));
                }
                scratch.symbols.push(MergeSymbol {
                    c: id,
                    prev: if i == 0 { -1 } else { (i - 1) as i32 },
                    next: if i == n - 1 { -1 } else { (i + 1) as i32 },
                });
                // Check pair with previous byte via pre-computed table.
                if i > 0 {
                    let (rank, _new_id) =
                        self.byte_pair_initial[prev_byte as usize * 256 + byte as usize];
                    if rank != u32::MAX {
                        scratch.heap_buf.push(Reverse(MergeEntry::new(
                            rank,
                            (i - 1) as u32,
                            self.byte_to_initial_token[prev_byte as usize],
                            id,
                        )));
                    }
                }
                prev_byte = byte;
            }

            if n == 1 {
                out.push(scratch.symbols[0].c);
                return Ok(());
            }

            // Bulk heapify.
            let mut tmp = std::mem::take(&mut scratch.heap_buf);
            scratch.heap.extend(tmp.drain(..));
            scratch.heap_buf = tmp;

            self.run_merge_loop(&mut scratch, out);

            Ok(())
        })
    }

    /// Seed the priority queue with all initial adjacent pairs.
    #[inline(always)]
    fn init_merge_heap(&self, scratch: &mut MergeScratch, n: usize) {
        let symbols = &scratch.symbols;
        scratch.heap.extend((0..n - 1).filter_map(|i| {
            let left = symbols[i].c;
            let right = symbols[i + 1].c;
            self.merge_adj
                .get(left, right)
                .map(|(rank, _new_id)| Reverse(MergeEntry::new(rank, i as u32, left, right)))
        }));
    }

    #[inline(always)]
    fn run_merge_loop(&self, scratch: &mut MergeScratch, out: &mut Vec<u32>) {
        let symbols = &mut scratch.symbols;
        let heap = &mut scratch.heap;

        while let Some(Reverse(entry)) = heap.pop() {
            let pos = entry.pos() as usize;
            let sym = symbols[pos];

            // Stale-entry check.
            let left_c = entry.left_c();
            let right_c = entry.right_c();
            if sym.c != left_c {
                continue;
            }
            let next_idx = sym.next;
            if next_idx < 0 {
                continue;
            }
            let next_idx = next_idx as usize;
            let next_sym = symbols[next_idx];
            if next_sym.c != right_c {
                continue;
            }

            // Derive new_id from adjacency list.
            let new_id = match self.merge_adj.get(left_c, right_c) {
                Some((_, nid)) => nid,
                None => continue,
            };

            // Merge: left symbol absorbs right.
            symbols[pos].c = new_id;
            symbols[pos].next = next_sym.next;
            if next_sym.next >= 0 {
                symbols[next_sym.next as usize].prev = pos as i32;
            }
            symbols[next_idx].c = INVALID_TOKEN;

            // Discover new adjacent pairs.
            if sym.prev >= 0 {
                let prev_c = symbols[sym.prev as usize].c;
                if let Some((rank, _)) = self.merge_adj.get(prev_c, new_id) {
                    heap.push(Reverse(MergeEntry::new(
                        rank,
                        sym.prev as u32,
                        prev_c,
                        new_id,
                    )));
                }
            }
            let new_next = symbols[pos].next;
            if new_next >= 0 {
                let next_c = symbols[new_next as usize].c;
                if let Some((rank, _)) = self.merge_adj.get(new_id, next_c) {
                    heap.push(Reverse(MergeEntry::new(rank, pos as u32, new_id, next_c)));
                }
            }
        }

        let mut i: i32 = 0;
        while i >= 0 {
            let sym = symbols[i as usize];
            out.push(sym.c);
            i = sym.next;
        }
    }

    #[inline(always)]
    pub fn tokenize_into_fused(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        if raw_input.is_empty() {
            return Ok(());
        }

        let bpe_id = self.id;
        let hit = TL_FUSED_CACHE.with(|c| {
            let c = c.borrow();
            if c.bpe_id != bpe_id {
                return false;
            }
            c.get(raw_input, out)
        });
        if hit {
            return Ok(());
        }

        let start = out.len();
        if self.ignore_merges
            && let Some(id) = self.whole_pretoken_id(raw_input)
        {
            out.push(id);
        } else {
            self.merge_all_raw_into(raw_input, out)?;
        }

        let ids = &out[start..];
        TL_FUSED_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                c.bpe_id = bpe_id;
                c.clear();
            }
            c.insert(raw_input, ids);
        });

        Ok(())
    }

    /// Fused tokenization of one already-sliced raw-text piece, consulting and
    /// populating the given thread-local cache plus the shared cache. Shared by
    /// the split-based and range-based batch entry points.
    #[inline]
    fn fused_one(&self, text: &str, cache: &mut PretokenCache, out: &mut Vec<u32>) -> Result<()> {
        if cache.get(text, out) {
            return Ok(());
        }

        let start = out.len();
        if self.ignore_merges
            && let Some(id) = self.whole_pretoken_id(text)
        {
            out.push(id);
        } else {
            self.merge_all_raw_into(text, out)?;
        }
        cache.insert(text, &out[start..]);
        Ok(())
    }

    /// Encode one already-harvested pretoken span using a precomputed cache key
    /// and hash (`key == 0` means the span is not inline-cacheable). Its cache
    /// line was prefetched `PREFETCH_DISTANCE` spans earlier by the driver.
    #[inline(always)]
    fn process_span(
        &self,
        cache: &mut PretokenCache,
        text: &str,
        key: u128,
        hash: u64,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        if key != 0 {
            out.reserve(PT_INLINE);
            if cache.probe_emit_fast(key, hash, out) {
                return Ok(());
            }
            if cache.get_by_key(key, hash, out) {
                return Ok(());
            }
        }
        let start = out.len();
        if self.ignore_merges
            && let Some(id) = self.whole_pretoken_id(text)
        {
            out.push(id);
        } else {
            self.merge_all_raw_into(text, out)?;
        }
        if key != 0 {
            cache.insert_by_key(key, hash, &out[start..]);
        }
        Ok(())
    }

    pub fn tokenize_batch_fused(
        &self,
        buffer: &str,
        splits: &[crate::pre_tokenized::Split],
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let bpe_id = self.id;
        TL_FUSED_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            if cache.bpe_id != bpe_id {
                cache.bpe_id = bpe_id;
                cache.clear();
            }

            for split in splits {
                if let Some(id) = split.token_id {
                    out.push(id);
                } else if !split.range.is_empty() {
                    let text = &buffer[split.range.clone()];
                    if !text.is_empty() {
                        self.fused_one(text, &mut cache, out)?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Like [`Self::tokenize_batch_fused`] but over pre-computed `(start, end)`
    /// byte ranges (all text; no pre-assigned token IDs), as produced by the
    /// scanner fast path.
    /// Fused scan+BPE of one segment under a single thread-local cache borrow.
    ///
    /// Drives [`scan_core`](crate::pre_tokenizers::scan::scan_core) as a
    /// software pipeline: the scan harvests each pretoken's cache key/hash and
    /// prefetches its cache line, but the actual probe+encode of that span is
    /// deferred [`PREFETCH_DISTANCE`] spans (held in a small ring), by which
    /// time the (often DRAM-resident) line has arrived. This hides the pretoken
    /// cache's memory latency, which dominates the fused loop on long-tail text.
    pub fn tokenize_scanned_segment(
        &self,
        kind: crate::pre_tokenizers::scan::ScanKind,
        seg: &str,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        /// Spans of prefetch-ahead. Enough to cover DRAM latency; the ring is
        /// tiny so it stays in registers/L1.
        const PREFETCH_DISTANCE: usize = 16;

        #[derive(Clone, Copy, Default)]
        struct Pending {
            start: u32,
            end: u32,
            key: u128,
            hash: u64,
        }

        let bpe_id = self.id;
        let sb = seg.as_bytes();
        TL_FUSED_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            if cache.bpe_id != bpe_id {
                cache.bpe_id = bpe_id;
                cache.clear();
            }

            let mut ring = [Pending::default(); PREFETCH_DISTANCE];
            let mut n: usize = 0; // spans harvested so far

            crate::pre_tokenizers::scan::scan_core(kind, seg, |start, end| {
                if start == end {
                    return Ok(());
                }
                // Harvest: pack the key, prefetch its line, park the span.
                let len = end - start;
                let (key, hash) = if (1..=15).contains(&len) {
                    let k = PretokenCache::pack_key_at(sb, start, len);
                    let h = PretokenCache::hash(k);
                    cache.prefetch(h);
                    (k, h)
                } else {
                    (0u128, 0u64)
                };
                // The span `n - PREFETCH_DISTANCE` (living in this ring slot,
                // prefetched that many spans ago) is now due — encode it before
                // its slot is overwritten.
                if n >= PREFETCH_DISTANCE {
                    let p = ring[n % PREFETCH_DISTANCE];
                    self.process_span(
                        &mut cache,
                        &seg[p.start as usize..p.end as usize],
                        p.key,
                        p.hash,
                        out,
                    )?;
                }
                ring[n % PREFETCH_DISTANCE] = Pending {
                    start: start as u32,
                    end: end as u32,
                    key,
                    hash,
                };
                n += 1;
                Ok(())
            })?;

            // Drain the last (up to) PREFETCH_DISTANCE parked spans, in order.
            let first = n.saturating_sub(PREFETCH_DISTANCE);
            for i in first..n {
                let p = ring[i % PREFETCH_DISTANCE];
                self.process_span(
                    &mut cache,
                    &seg[p.start as usize..p.end as usize],
                    p.key,
                    p.hash,
                    out,
                )?;
            }
            Ok(())
        })
    }

    /// Like [`tokenize_scanned_segment`], but also appends fine-grained reuse
    /// boundaries to `bounds`: for every pretoken that begins at a hard boundary
    /// (preceded by a `\r`/`\n`, followed by an ASCII non-whitespace byte), the
    /// `(local_byte_offset, local_token_index)` at that point — i.e.
    /// `out[..token_index]` is exactly the encoding of `seg[..byte_offset]`.
    /// Used only by the prefix cache's (cold) miss path, so the extra per-
    /// pretoken check stays out of the hot [`tokenize_scanned_segment`].
    pub fn tokenize_scanned_segment_rec(
        &self,
        kind: crate::pre_tokenizers::scan::ScanKind,
        seg: &str,
        out: &mut Vec<u32>,
        bounds: &mut Vec<(u32, u32)>,
    ) -> Result<()> {
        let bpe_id = self.id;
        let b = seg.as_bytes();
        TL_FUSED_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            if cache.bpe_id != bpe_id {
                cache.bpe_id = bpe_id;
                cache.clear();
            }
            crate::pre_tokenizers::scan::scan_core(kind, seg, |start, end| {
                if start > 0
                    && (b[start - 1] == b'\n' || b[start - 1] == b'\r')
                    && b[start] < 0x80
                    && !b[start].is_ascii_whitespace()
                {
                    bounds.push((start as u32, out.len() as u32));
                }
                if start != end {
                    self.fused_one(&seg[start..end], &mut cache, out)?;
                }
                Ok(())
            })
        })
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

impl Clone for Bpe {
    fn clone(&self) -> Self {
        Self {
            id: BPE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            daac: self.daac.clone(),
            merge_map: self.merge_map.clone(),
            unmerge_map: self.unmerge_map.clone(),
            next_prefix_map: self.next_prefix_map.clone(),
            token_lens: self.token_lens.clone(),
            shared_cache: SharedCache::new(),
            id_to_token: self.id_to_token.clone(),
            token_to_id: self.token_to_id.clone(),
            byte_to_initial_token: self.byte_to_initial_token,
            byte_fallback_token_ids: self.byte_fallback_token_ids,
            single_char_token: self.single_char_token,
            ranked_merge_map: self.ranked_merge_map.clone(),
            byte_pair_initial: self.byte_pair_initial.clone(),
            merge_adj: self.merge_adj.clone(),
            ignore_merges: self.ignore_merges,
            byte_fallback: self.byte_fallback,
            bigram_bridge_table: self.bigram_bridge_table.clone(),
        }
    }
}

impl fmt::Debug for Bpe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bpe")
            .field("vocab_size", &self.token_lens.len())
            .field("merges", &self.merge_map.len())
            .finish()
    }
}

impl PartialEq for Bpe {
    fn eq(&self, other: &Self) -> bool {
        self.daac == other.daac
            && self.merge_map == other.merge_map
            && self.unmerge_map == other.unmerge_map
            && self.next_prefix_map == other.next_prefix_map
            && self.token_lens == other.token_lens
            && self.ignore_merges == other.ignore_merges
            && self.byte_fallback == other.byte_fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_structs::ModelConfig;

    fn test_bpe() -> Bpe {
        let vocab: Vocab = [
            ("a", 0),
            ("b", 1),
            ("c", 2),
            ("d", 3),
            ("ab", 4),
            ("cd", 5),
            ("abcd", 6),
        ]
        .into_iter()
        .map(|(s, id)| (s.to_string(), id))
        .collect();

        let merges: Vec<Value> = vec![
            Value::String("a b".into()),
            Value::String("c d".into()),
            Value::String("ab cd".into()),
        ];

        let merge_map = parse_merges(&vocab, &merges).unwrap();
        Bpe::new(&vocab, merge_map).unwrap()
    }

    #[test]
    fn empty_input() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn single_char() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("a").unwrap(), vec![0]);
        assert_eq!(bpe.tokenize("d").unwrap(), vec![3]);
    }

    #[test]
    fn simple_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("ab").unwrap(), vec![4]);
        assert_eq!(bpe.tokenize("cd").unwrap(), vec![5]);
    }

    #[test]
    fn chained_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abcd").unwrap(), vec![6]);
    }

    #[test]
    fn partial_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abc").unwrap(), vec![4, 2]);
    }

    #[test]
    fn repeated_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abab").unwrap(), vec![4, 4]);
    }

    #[test]
    fn merge_small_matches_heap() {
        let bpe = test_bpe();
        let alphabet = [b'a', b'b', b'c', b'd'];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;

        for len in 1..=SMALL_MERGE_MAX {
            for _ in 0..512 {
                let mut bytes = Vec::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    bytes.push(alphabet[state as usize & 3]);
                }
                let input = std::str::from_utf8(&bytes).unwrap();
                let mut small = Vec::new();
                let mut heap = Vec::new();
                bpe.merge_all_raw_into(input, &mut small).unwrap();
                bpe.merge_all_raw_heap_into(input, &mut heap).unwrap();
                assert_eq!(small, heap, "short merge mismatch for {input:?}");
            }
        }
    }

    fn assert_encoded_matches_heap(bpe: &Bpe, input: &str) {
        let mut optimized = vec![0xdead_beefu32, 0xcafe_babe];
        let mut heap = optimized.clone();
        let optimized_result = bpe.merge_all_encoded_into(input, &mut optimized);
        let heap_result = bpe.merge_all_encoded_heap_into(input, &mut heap);
        assert_eq!(
            optimized_result, heap_result,
            "result mismatch for {input:?}"
        );
        assert_eq!(optimized, heap, "output mismatch for {input:?}");
    }

    #[test]
    fn encoded_small_matches_heap_for_all_symbol_counts() {
        let bpe = test_bpe();
        let alphabet = [b'a', b'b', b'c', b'd'];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;

        for len in 1..=SMALL_MERGE_MAX + 1 {
            for _ in 0..128 {
                let mut input = String::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    input.push(alphabet[state as usize & 3] as char);
                }
                assert_encoded_matches_heap(&bpe, &input);
            }
        }
    }

    #[test]
    fn encoded_small_preserves_leftmost_equal_rank_ties() {
        let vocab: Vocab = [("a", 0), ("b", 1), ("c", 2), ("ab", 3), ("bc", 4)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let merge_map = [((0, 1), (7, 3)), ((1, 2), (7, 4))].into_iter().collect();
        let bpe = Bpe::new(&vocab, merge_map).unwrap();

        assert_encoded_matches_heap(&bpe, "abc");
        let mut out = Vec::new();
        bpe.merge_all_encoded_into("abc", &mut out).unwrap();
        assert_eq!(out, vec![3, 2]);
    }

    #[test]
    fn encoded_small_handles_bytelevel_chars_and_combining_marks() {
        let left = BYTE_TO_CHAR[0];
        let right = BYTE_TO_CHAR[1];
        let left_right = format!("{left}{right}");
        let vocab: Vocab = [
            (left.to_string(), 0),
            (right.to_string(), 1),
            (left_right, 2),
        ]
        .into_iter()
        .map(|(s, id)| (s, id))
        .collect();
        let merge_map = [((0, 1), (0, 2))].into_iter().collect();
        let bytelevel_bpe = Bpe::new(&vocab, merge_map).unwrap();
        assert_encoded_matches_heap(&bytelevel_bpe, &format!("{left}{right}"));
        assert_encoded_matches_heap(&bytelevel_bpe, &left.to_string().repeat(32));
        assert_encoded_matches_heap(&bytelevel_bpe, &left.to_string().repeat(33));

        let combining = '\u{301}';
        let combined = format!("e{combining}");
        let vocab: Vocab = [("e".into(), 0), (combining.to_string(), 1), (combined, 2)]
            .into_iter()
            .collect();
        let merge_map = [((0, 1), (0, 2))].into_iter().collect();
        let combining_bpe = Bpe::new(&vocab, merge_map).unwrap();
        assert_encoded_matches_heap(&combining_bpe, &format!("e{combining}"));
    }

    fn fallback_bpe(include_second_byte: bool) -> Bpe {
        let mut vocab: Vocab = [("a".into(), 0), ("<0xC3>".into(), 1)]
            .into_iter()
            .collect();
        if include_second_byte {
            vocab.insert("<0xA9>".into(), 2);
            vocab.insert("<0xC3><0xA9>".into(), 3);
        }
        let merge_map = if include_second_byte {
            [((1, 2), (0, 3))].into_iter().collect()
        } else {
            ParsedMergeMap::new()
        };
        let mut bpe = Bpe::new(&vocab, merge_map).unwrap();
        bpe.byte_fallback = true;
        bpe
    }

    #[test]
    fn encoded_small_handles_byte_fallback_expansion_and_boundary() {
        let bpe = fallback_bpe(true);
        assert_encoded_matches_heap(&bpe, "é");
        assert_encoded_matches_heap(&bpe, &format!("{}é", "a".repeat(30)));
        assert_encoded_matches_heap(&bpe, &format!("{}é", "a".repeat(31)));
    }

    #[test]
    fn encoded_errors_are_equal_and_leave_prefilled_output_untouched() {
        let bpe = fallback_bpe(false);
        let input = "aé";
        let mut optimized = vec![17, 19];
        let mut heap = optimized.clone();
        let optimized_result = bpe.merge_all_encoded_into(input, &mut optimized);
        let heap_result = bpe.merge_all_encoded_heap_into(input, &mut heap);
        assert_eq!(optimized_result, heap_result);
        assert_eq!(
            optimized_result.unwrap_err(),
            "byte fallback token <0xA9> not in vocabulary"
        );
        assert_eq!(optimized, vec![17, 19]);
        assert_eq!(heap, optimized);

        let mut optimized = vec![23];
        let mut heap = optimized.clone();
        let optimized_result = bpe.merge_all_encoded_into("?", &mut optimized);
        let heap_result = bpe.merge_all_encoded_heap_into("?", &mut heap);
        assert_eq!(optimized_result, heap_result);
        assert_eq!(
            optimized_result.unwrap_err(),
            "byte fallback token <0x3F> not in vocabulary"
        );
        assert_eq!(optimized, vec![23]);
        assert_eq!(heap, optimized);
    }

    #[test]
    fn deserialize_from_json() {
        let json = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": ["a b"]
        });
        let config: ModelConfig = serde_json::from_value(json).unwrap();
        assert!(matches!(config, ModelConfig::Bpe(_)));
    }

    #[test]
    fn deserialize_array_merges() {
        let json = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": [["a", "b"]]
        });
        let config: ModelConfig = serde_json::from_value(json).unwrap();
        let ModelConfig::Bpe(bpe) = config else {
            panic!("expected Bpe variant");
        };
        assert_eq!(bpe.tokenize("ab").unwrap(), vec![2]);
    }

    #[test]
    fn cache_returns_same_result() {
        let vocab: Vocab = [("a", 0), ("b", 1), ("ab", 2)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let merges = vec![Value::String("a b".into())];
        let merge_map = parse_merges(&vocab, &merges).unwrap();
        let bpe = Bpe::new(&vocab, merge_map).unwrap();

        let first = bpe.tokenize("ab").unwrap();
        let second = bpe.tokenize("ab").unwrap();
        assert_eq!(first, second);
        assert_eq!(first, vec![2]);
    }
}
