use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fmt,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};
use serde::Deserialize;
use serde_json::Value;

use super::Result;
use crate::huge::HugeTable;
use crate::pre_tokenizers::BYTE_TO_CHAR;

type TokenId = u32;
type ParsedMergeMap = HashMap<(u32, u32), (u32, u32)>;
type Vocab = HashMap<String, u32>;

const INVALID_TOKEN: u32 = u32::MAX;

/// Pretokens with at most this many initial (per-byte) symbols use the
/// stack-resident linear-scan merge instead of the heap. Sized so the stack
/// arrays fit in registers/L1 and `u8` linked-list indices stay in range.
const SMALL_MERGE_MAX: usize = 128;
/// Packed merge values (`merge_grid`, `merge_adj`, `pair_initial`):
/// `rank << 32 | SAFE << 30 | product`, product a 30-bit compact id. Numeric order
/// is rank order (ranks are unique), so `min` picks the highest-priority merge.
/// `SAFE`: every occurrence of the pair may merge in one multipass sweep — its
/// product can't take part in any cheaper merge that would be due first.
const PV_ID_MASK: u64 = (1 << 30) - 1;
const PV_SAFE: u64 = 1 << 30;

/// Position bits of a packed small-merge key (`rank << SMALL_KEY_BITS | pos`).
const SMALL_KEY_BITS: u32 = 7;
const SMALL_KEY_MASK: u32 = (1 << SMALL_KEY_BITS) - 1;
const _: () = assert!(
    SMALL_MERGE_MAX <= 1 << SMALL_KEY_BITS
        && SMALL_MERGE_MAX.is_multiple_of(8)
        && SMALL_MERGE_MAX < 255
);

/// Side length of the dense merge grid (see [`Bpe::merge_grid`]). Pairs whose
/// two compact ids are both `< MERGE_GRID_DIM` are answered by one direct load.
/// 512² × 8 B = 2 MiB; frequent merges cluster in the low compact ids, so this
/// covers the overwhelming majority of hot-loop lookups.
const MERGE_GRID_DIM: u32 = 512;

/// Side length of [`Bpe::code_grid`], the rank-code grid used instead of
/// `merge_grid` when [`rank_codes_valid`]: 1024² codes × 4 B = 4 MiB.
const CODE_GRID_DIM: u32 = 1024;

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
const PRETOKEN_CACHE_MIN_BITS: usize = 16;
const PRETOKEN_CACHE_MAX_BITS: usize = 21; // ~2M entries × 32 B = 64 MiB cap
const PT_INLINE: usize = 3;

/// Ids of a spilled cache value copied at once on a hit (see
/// [`PretokenCache::get_or_vacancy_promote`]); the spill keeps this much room.
const SPILL_COPY: usize = 16;

/// Whether a long pretoken (> 15 bytes) starting with byte `b0` bypasses the
/// long-pretoken cache: one led by a CJK ideograph or kana (U+3000–U+9FFF) is a
/// run of them — unique as a whole in running text, which has no spaces there
/// (~16% hits measured), so a hash, a probe and an insert per span cost more
/// than the rare hit saves; its segmented merge ([`Bpe::merge_with_atoms`]) is
/// cheap. (Cyrillic or Latin words repeat: those stay cached.)
#[inline(always)]
fn long_uncached(b0: u8) -> bool {
    (0xE3..=0xE9).contains(&b0)
}

/// Low bits of a [`CharAtoms`] BMP entry: the atom's compact id.
const ATOM_ID_BITS: u32 = 18;
const ATOM_ID_MASK: u32 = (1 << ATOM_ID_BITS) - 1;

/// `KEEP_LOW_BYTES[n]` keeps the low `n` bytes of a `u128`.
const KEEP_LOW_BYTES: [u128; 16] = {
    let mut m = [0u128; 16];
    let mut i = 1;
    while i < 16 {
        m[i] = (1u128 << (i * 8)) - 1;
        i += 1;
    }
    m
};

/// [`KEEP_LOW_BYTES`] split into its low and high 64-bit halves (the key packer
/// works in general-purpose registers).
const KEEP_LOW_HALVES: ([u64; 16], [u64; 16]) = {
    let (mut lo, mut hi) = ([0u64; 16], [0u64; 16]);
    let mut i = 0;
    while i < 16 {
        lo[i] = KEEP_LOW_BYTES[i] as u64;
        hi[i] = (KEEP_LOW_BYTES[i] >> 64) as u64;
        i += 1;
    }
    (lo, hi)
};

/// Prefetch the cache line at `p` into L2 only (not L1): for a line needed a
/// whole chunk later, where an L1 prefetch would evict the working set. No
/// memory effects; a no-op off x86-64.
#[inline(always)]
fn prefetch_l2(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch has no memory effects and reads nothing.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T1);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

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

/// Exact cache for pretokens too long to pack into a `u128` key (> 15 bytes).
///
/// These are rare as *spans* (~2% of pretokens on prose) but they are long
/// (whitespace/indentation runs, long words, punctuation rules), so they carry
/// ~10% of the input bytes — and each costs a full O(n²)/heap BPE merge. Before
/// this cache they were re-merged on every occurrence. `tk-encode` caches words of
/// any length too, but trusts a 127-bit hash; this one is exact: the pretoken
/// bytes live in an arena and a hit is confirmed with a `memcmp`.
///
/// Kept apart from the 32-byte inline table so the hot short-pretoken probe is
/// untouched; it is only consulted for spans that could not be packed.
struct LongCache {
    mask: usize,
    len: usize,
    slots: HugeTable<LongSlot>,
    /// Pretoken bytes, referenced by `LongSlot::boff/blen`.
    bytes: Vec<u8>,
    /// Token ids, referenced by `LongSlot::ioff/ilen`.
    ids: Vec<u32>,
}

#[derive(Clone, Copy)]
struct LongSlot {
    /// Nonzero 64-bit hash of the bytes; `0` marks an empty slot.
    hash: u64,
    boff: u32,
    blen: u32,
    ioff: u32,
    ilen: u32,
}

const LONG_CACHE_MIN_BITS: usize = 10;
const LONG_CACHE_MAX_BITS: usize = 18; // 256k entries
/// Arena budget: past this many pretoken bytes the cache clears (diverse input).
const LONG_CACHE_MAX_BYTES: usize = 32 << 20;

impl LongCache {
    fn new() -> Self {
        let cap = 1usize << LONG_CACHE_MIN_BITS;
        Self {
            mask: cap - 1,
            len: 0,
            slots: HugeTable::new(cap, LongSlot::EMPTY),
            bytes: Vec::new(),
            ids: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for s in self.slots.iter_mut() {
            s.hash = 0;
        }
        self.bytes.clear();
        self.ids.clear();
        self.len = 0;
    }

    /// Hash of a long pretoken: each 16-byte chunk (and the last 16 bytes, read
    /// overlapping) folded through one 64×64→128 multiply with keys that depend
    /// on its position, the products summed — independent multiplies, not a
    /// chain, and no copy of a ragged tail. Never returns 0 (the empty-slot marker).
    #[inline]
    fn hash(b: &[u8]) -> u64 {
        const K0: u64 = 0xA076_1D64_78BD_642F;
        const K1: u64 = 0xE703_7ED1_A0B4_28DB;
        const K2: u64 = 0x8EBC_6AF0_9C88_C6E3;
        const K3: u64 = 0x5899_65CC_7537_4CC3;
        #[inline(always)]
        fn fold(a: u64, b: u64) -> u64 {
            let p = a as u128 * b as u128;
            p as u64 ^ (p >> 64) as u64
        }
        let n = b.len();
        let rd = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let mut acc = (n as u64).wrapping_mul(K2);
        if n >= 16 {
            let (mut i, mut k) = (0usize, 0u64);
            while i + 16 <= n {
                acc = acc.wrapping_add(fold(
                    rd(i) ^ K0.wrapping_add(k),
                    rd(i + 8) ^ K1.wrapping_add(k),
                ));
                i += 16;
                k = k.wrapping_add(K2);
            }
            acc = acc.wrapping_add(fold(rd(n - 16) ^ K3, rd(n - 8) ^ K2));
        } else {
            // (Never on the cache's own paths: long pretokens have > 15 bytes.)
            let mut buf = [0u8; 16];
            buf[..n].copy_from_slice(b);
            let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
            let hi = u64::from_le_bytes(buf[8..].try_into().unwrap());
            acc = acc.wrapping_add(fold(lo ^ K0, hi ^ K1));
        }
        fold(acc ^ K3, (n as u64) ^ K1) | 1
    }

    /// Prefetch the slot a lookup of `hash` starts at.
    #[inline(always)]
    fn prefetch(&self, hash: u64) {
        // SAFETY: `hash & mask < slots.len()`; a prefetch reads nothing.
        prefetch_read(unsafe { self.slots.as_ptr().add(hash as usize & self.mask) } as *const u8);
    }

    #[inline]
    fn get(&self, text: &[u8], hash: u64, out: &mut Vec<u32>) -> bool {
        let mut idx = hash as usize & self.mask;
        loop {
            // SAFETY: `idx <= mask < slots.len()`.
            let s = unsafe { *self.slots.get_unchecked(idx) };
            if s.hash == 0 {
                return false;
            }
            if s.hash == hash && s.blen as usize == text.len() {
                let (bo, bl) = (s.boff as usize, s.blen as usize);
                if &self.bytes[bo..bo + bl] == text {
                    let (io, il) = (s.ioff as usize, s.ilen as usize);
                    out.extend_from_slice(&self.ids[io..io + il]);
                    return true;
                }
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Insert a key known to be absent.
    #[inline]
    fn insert(&mut self, text: &[u8], hash: u64, ids: &[u32]) {
        if self.bytes.len() + text.len() > LONG_CACHE_MAX_BYTES {
            self.clear();
        }
        if (self.len + 1) * 4 > self.slots.len() * 3 {
            self.grow_or_clear();
        }
        let s = LongSlot {
            hash,
            boff: self.bytes.len() as u32,
            blen: text.len() as u32,
            ioff: self.ids.len() as u32,
            ilen: ids.len() as u32,
        };
        self.bytes.extend_from_slice(text);
        self.ids.extend_from_slice(ids);
        self.place(s);
    }

    #[inline]
    fn place(&mut self, s: LongSlot) {
        let mut idx = s.hash as usize & self.mask;
        loop {
            // SAFETY: `idx <= mask < slots.len()`.
            let slot = unsafe { self.slots.get_unchecked_mut(idx) };
            if slot.hash == 0 {
                *slot = s;
                self.len += 1;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[cold]
    fn grow_or_clear(&mut self) {
        if self.slots.len() >= (1usize << LONG_CACHE_MAX_BITS) {
            self.clear();
            return;
        }
        let new_cap = self.slots.len() * 2;
        let old = std::mem::replace(&mut self.slots, HugeTable::new(new_cap, LongSlot::EMPTY));
        self.mask = new_cap - 1;
        self.len = 0;
        for &s in old.iter() {
            if s.hash != 0 {
                self.place(s);
            }
        }
    }
}

impl LongSlot {
    const EMPTY: Self = Self {
        hash: 0,
        boff: 0,
        blen: 0,
        ioff: 0,
        ilen: 0,
    };
}

/// A [`PretokenCache`] image pre-filled with every short vocab entry, each
/// valued with exactly what a cold miss on its bytes computes
/// ([`Bpe::miss_value_into`]). A thread's cache starts from it when it binds to
/// the model outside a parallel job ([`Bpe::pt_seed`]), so a pretoken that is a
/// whole vocab word — most first-seen words in running text — is a hit on the
/// hot path instead of a miss.
struct PtSeed {
    cap: usize,
    len: usize,
    slots: HugeTable<PtEntry>,
    spill: Vec<u32>,
}

struct PretokenCache {
    bpe_id: usize,
    mask: usize,
    cap: usize,
    len: usize,
    slots: HugeTable<PtEntry>,
    spill: Vec<u32>,
    /// Exact cache for pretokens longer than the 15-byte inline key.
    long: LongCache,
    /// [`Bpe::tokenize_scanned_segment`]'s span buffer, kept across calls: a
    /// fresh one is ~9 KB of zero fill, the bulk of a short input's fixed cost.
    span_chunk: Option<Box<SpanChunk>>,
    /// Holds the model's [`PtSeed`] (every whole short vocab word), so a miss is
    /// known not to be one: the `ignore_merges` fold probe is skipped. Cleared
    /// with the table.
    seeded: bool,
}

impl PretokenCache {
    fn new() -> Self {
        let cap = 1usize << PRETOKEN_CACHE_MIN_BITS;
        Self {
            bpe_id: 0,
            mask: cap - 1,
            cap,
            len: 0,
            slots: HugeTable::new(cap, PtEntry::empty()),
            spill: Vec::new(),
            long: LongCache::new(),
            span_chunk: None,
            seeded: false,
        }
    }

    fn clear(&mut self) {
        for e in self.slots.iter_mut() {
            e.key = 0;
        }
        self.spill.clear();
        self.len = 0;
        self.long.clear();
        self.seeded = false;
    }

    /// Switch this thread's cache to another tokenizer's model: its entries are
    /// meaningless now. With a `seed` the table becomes a copy of it; otherwise a
    /// grown table is dropped for a fresh minimum-size one — zeroing up to 64 MiB
    /// would cost the new model's first encode more than a lazily-zeroed
    /// allocation, and the new model starts from a compact table.
    fn reset_for(&mut self, bpe_id: usize, seed: Option<&PtSeed>) {
        match seed {
            Some(seed) => {
                if self.cap == seed.cap {
                    self.slots.copy_from_slice(&seed.slots);
                } else {
                    self.slots = HugeTable::from_slice(&seed.slots);
                }
                self.cap = seed.cap;
                self.mask = seed.cap - 1;
                self.len = seed.len;
                self.spill.clear();
                self.spill.extend_from_slice(&seed.spill);
                self.spill.reserve(SPILL_COPY);
                self.seeded = true;
                if self.long.slots.len() > (1usize << LONG_CACHE_MIN_BITS) {
                    self.long = LongCache::new();
                } else {
                    self.long.clear();
                }
            }
            None if self.cap > (1usize << PRETOKEN_CACHE_MIN_BITS)
                || self.long.slots.len() > (1usize << LONG_CACHE_MIN_BITS) =>
            {
                *self = Self::new();
            }
            None => self.clear(),
        }
        self.bpe_id = bpe_id;
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
            let (lo, hi) = unsafe {
                let p = buf.as_ptr().add(start);
                (
                    (p as *const u64).read_unaligned(),
                    (p.add(8) as *const u64).read_unaligned(),
                )
            };
            // The low `len` bytes, masked per 64-bit half from a table: a variable
            // `u128` shift is a branchy multi-instruction sequence on the hottest
            // path, and one 16-byte vector load costs lane moves into registers.
            let lo = lo & KEEP_LOW_HALVES.0[len & 15];
            let hi = (hi & KEEP_LOW_HALVES.1[len & 15]) | ((len as u64) << 56);
            (lo as u128) | ((hi as u128) << 64)
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

    #[inline(always)]
    fn get(&self, key: &str, out: &mut Vec<u32>) -> bool {
        let b = key.as_bytes();
        match Self::pack_key(b) {
            Some(k) => self.get_by_key(k, Self::hash(k), out),
            None if b.len() > 15 => self.long.get(b, LongCache::hash(b), out),
            None => false,
        }
    }

    #[inline(always)]
    fn get_by_key(&self, k: u128, hash: u64, out: &mut Vec<u32>) -> bool {
        self.get_or_vacancy(k, hash, out).is_ok()
    }

    /// Like [`Self::get_by_key`], but a miss returns the empty slot that ends the
    /// probe chain — exactly where [`Self::insert_at`] can place the key, so an
    /// insert after a miss need not walk the chain again.
    #[inline(always)]
    fn get_or_vacancy(
        &self,
        k: u128,
        hash: u64,
        out: &mut Vec<u32>,
    ) -> std::result::Result<(), usize> {
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
                return Ok(());
            }
            if e.key == 0 {
                return Err(idx);
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// [`Self::get_or_vacancy`] for the hot loop's slow path: a hit found away from
    /// its home slot is swapped into it, so repeat hits take the one-compare
    /// probe of [`Bpe::probe_emit`] instead of walking the chain again. Linear probing
    /// keeps every key findable across the swap: the home occupant only moves
    /// further along a run of occupied slots that already covered it.
    #[inline(always)]
    fn get_or_vacancy_promote(
        &mut self,
        k: u128,
        hash: u64,
        out: &mut Vec<u32>,
    ) -> std::result::Result<(), usize> {
        let home = hash as usize & self.mask;
        let mut idx = home;
        loop {
            let e = unsafe { *self.slots.get_unchecked(idx) };
            if e.key == k {
                let len = e.len as usize;
                if len <= PT_INLINE {
                    // All `PT_INLINE` lanes stored, the length advanced by `len`: a
                    // fixed-size copy where `extend_from_slice` of 1-3 ids is a
                    // `memmove` call.
                    out.reserve(PT_INLINE);
                    // SAFETY: `PT_INLINE` ids fit past `out.len()` (reserved), and
                    // the first `len` are initialized before the length covers them.
                    unsafe {
                        let dst = out.as_mut_ptr().add(out.len());
                        std::ptr::copy_nonoverlapping(e.v.as_ptr(), dst, PT_INLINE);
                        out.set_len(out.len() + len);
                    }
                } else if len <= SPILL_COPY {
                    // Like the inline lanes: a fixed-size copy (of up to 16 ids;
                    // a word of a script tokenized per char is often 4 to 10).
                    let o = e.v[0] as usize;
                    out.reserve(SPILL_COPY);
                    // SAFETY: `SPILL_COPY` ids fit past `out.len()` (reserved); the
                    // spill keeps `SPILL_COPY` ids of room past its length (see
                    // `build_entry`), so the source stays in its allocation, and
                    // only its first `len` ids (initialized) are covered by the length.
                    unsafe {
                        let dst = out.as_mut_ptr().add(out.len());
                        std::ptr::copy_nonoverlapping(self.spill.as_ptr().add(o), dst, SPILL_COPY);
                        out.set_len(out.len() + len);
                    }
                } else {
                    let o = e.v[0] as usize;
                    out.extend_from_slice(unsafe { self.spill.get_unchecked(o..o + len) });
                }
                if idx != home {
                    self.slots.swap(idx, home);
                }
                return Ok(());
            }
            if e.key == 0 {
                return Err(idx);
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Insert a key known to be absent at `vacancy`, the empty slot its probe
    /// chain ended on (from [`Self::get_or_vacancy`], with no insert in between).
    #[inline(always)]
    fn insert_at(&mut self, k: u128, hash: u64, vacancy: usize, ids: &[u32]) {
        if (self.len + 1) * 4 > self.cap * 3 {
            // Growing rehashes everything; the vacancy is stale.
            self.insert_by_key(k, hash, ids);
            return;
        }
        let e = Self::build_entry(&mut self.spill, k, ids);
        // SAFETY: `vacancy <= mask < slots.len()`.
        unsafe { *self.slots.get_unchecked_mut(vacancy) = e };
        self.len += 1;
    }

    #[inline(always)]
    fn insert(&mut self, key: &str, ids: &[u32]) {
        let b = key.as_bytes();
        if let Some(k) = Self::pack_key(b) {
            self.insert_by_key(k, Self::hash(k), ids);
        } else if b.len() > 15 {
            self.long.insert(b, LongCache::hash(b), ids);
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
            // Room for the fixed-size copy of a hit (see `get_or_vacancy_promote`).
            spill.reserve(SPILL_COPY);
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
        #[cfg(feature = "diag")]
        let _g = GrowTimer(std::time::Instant::now());
        if self.cap >= (1usize << PRETOKEN_CACHE_MAX_BITS) {
            self.clear();
            return;
        }
        let new_cap = self.cap * 2;
        let old = std::mem::replace(&mut self.slots, HugeTable::new(new_cap, PtEntry::empty()));
        self.cap = new_cap;
        self.mask = new_cap - 1;
        self.len = 0;
        // Spill offsets are preserved across a grow (the pool is untouched).
        for &e in old.iter() {
            if e.key != 0 {
                self.place(e);
            }
        }
    }
}

#[cfg(feature = "diag")]
struct GrowTimer(std::time::Instant);
#[cfg(feature = "diag")]
impl Drop for GrowTimer {
    fn drop(&mut self) {
        let ns = self.0.elapsed().as_nanos() as u64;
        DIAG_T.with(|d| {
            let mut a = d.get();
            a[6] += ns;
            d.set(a);
        });
    }
}

/// Slot bits of [`SharedPretokenCache`]: 2^19 slots of one cache line each
/// (32 MiB of address space, zero-filled lazily, so only the lines a workload
/// touches are ever committed).
const SHARED_PT_BITS: usize = 19;

/// One [`SharedPretokenCache`] slot: a seqlock (`seq` is odd while a writer is
/// storing) over an inline pretoken entry. Every field is atomic, so a reader
/// racing a writer can at worst see a torn entry, which the `seq` recheck
/// rejects. All zeros is an empty slot: packed keys are never 0.
#[repr(align(64))]
struct SharedPtSlot {
    seq: AtomicU32,
    len: AtomicU32,
    key: [AtomicU64; 2],
    ids: [AtomicU32; PT_INLINE],
}

/// Cross-thread second level behind every thread's [`PretokenCache`], for
/// pretokens with an inline key and at most [`PT_INLINE`] ids.
///
/// Each thread's cache only warms on the text that thread encodes, so when one
/// input is split across many threads each sees a fraction of it and misses far
/// more often — at 32 threads four times as many pretokens were merged from
/// scratch as by one thread over the same documents. Consulting what the other
/// threads already merged before running a full BPE merge recovers most of that.
/// Direct-mapped and lossy: a collision overwrites, and a writer that finds its
/// slot busy simply skips the insert.
struct SharedPretokenCache {
    slots: Box<[SharedPtSlot]>,
    mask: usize,
}

impl SharedPretokenCache {
    fn new() -> Self {
        let n = 1usize << SHARED_PT_BITS;
        let layout = std::alloc::Layout::array::<SharedPtSlot>(n).expect("shared cache layout");
        // SAFETY: an all-zero `SharedPtSlot` is valid (its fields are atomics of
        // integers); the allocation has the array's layout and exactly `n` slots.
        let slots = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout) as *mut SharedPtSlot;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, n))
        };
        Self { slots, mask: n - 1 }
    }

    /// Append the cached ids of `key` to `out`; `false` (and `out` untouched) if
    /// absent or the slot is being written.
    #[inline]
    fn get(&self, key: u128, hash: u64, out: &mut Vec<u32>) -> bool {
        // SAFETY: `hash & mask < slots.len()`.
        let s = unsafe { self.slots.get_unchecked(hash as usize & self.mask) };
        let seq = s.seq.load(Ordering::Acquire);
        if seq & 1 != 0
            || s.key[0].load(Ordering::Relaxed) != key as u64
            || s.key[1].load(Ordering::Relaxed) != (key >> 64) as u64
        {
            return false;
        }
        let len = s.len.load(Ordering::Relaxed) as usize;
        let ids = [
            s.ids[0].load(Ordering::Relaxed),
            s.ids[1].load(Ordering::Relaxed),
            s.ids[2].load(Ordering::Relaxed),
        ];
        std::sync::atomic::fence(Ordering::Acquire);
        if s.seq.load(Ordering::Relaxed) != seq || len == 0 || len > PT_INLINE {
            return false;
        }
        out.extend_from_slice(&ids[..len]);
        true
    }

    /// Record `key -> ids` (ignored if `ids` does not fit inline or another
    /// writer holds the slot).
    #[inline]
    fn insert(&self, key: u128, hash: u64, ids: &[u32]) {
        if ids.is_empty() || ids.len() > PT_INLINE {
            return;
        }
        // SAFETY: `hash & mask < slots.len()`.
        let s = unsafe { self.slots.get_unchecked(hash as usize & self.mask) };
        let seq = s.seq.load(Ordering::Relaxed);
        if seq & 1 != 0
            || s.seq
                .compare_exchange(
                    seq,
                    seq.wrapping_add(1),
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
                .is_err()
        {
            return;
        }
        s.key[0].store(key as u64, Ordering::Relaxed);
        s.key[1].store((key >> 64) as u64, Ordering::Relaxed);
        s.len.store(ids.len() as u32, Ordering::Relaxed);
        for (slot, &id) in s.ids.iter().zip(ids) {
            slot.store(id, Ordering::Relaxed);
        }
        s.seq.store(seq.wrapping_add(2), Ordering::Release);
    }
}

thread_local! {
    pub static DIAG: std::cell::Cell<[u64; 6]> = const { std::cell::Cell::new([0; 6]) };
    pub static DIAG_T: std::cell::Cell<[u64; 8]> = const { std::cell::Cell::new([0; 8]) };
    static TL_BPE_CACHE: RefCell<FlatCache> = RefCell::new(FlatCache::new());
    static TL_FUSED_CACHE: RefCell<PretokenCache> = RefCell::new(PretokenCache::new());
    /// Seeded caches of models this thread used before the one bound in
    /// `TL_FUSED_CACHE`, most recently used first (see [`Bpe::bind_cache`]).
    static TL_PARKED_CACHES: RefCell<Vec<PretokenCache>> = const { RefCell::new(Vec::new()) };
}

/// Seeded pretoken caches a thread keeps for models besides its current one.
const PARKED_CACHES: usize = 3;

/// Benchmark-only: replace this thread's pretoken caches with fresh (cold) ones.
#[doc(hidden)]
pub fn __bench_reset_thread_caches() {
    TL_FUSED_CACHE.with(|c| *c.borrow_mut() = PretokenCache::new());
    TL_PARKED_CACHES.with(|p| p.borrow_mut().clear());
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

/// Multibyte chars that BPE provably forms as a whole before anything else can
/// touch their bytes, so a merge may start from them as single symbols instead of
/// re-deriving each from its 2–4 bytes. On CJK text that removes about two thirds
/// of the merges of every cache-missed word.
///
/// A char `c` (a vocab token spelling exactly one char) is an *atom* when
/// 1. BPE over `c`'s bytes alone yields `[c]`, and
/// 2. every merge with `c` as an operand ranks after the merges that form it,
///
/// and pre-forming one occurrence is exact when also
/// 3. no vocab token occurs across either of its boundaries while covering only
///    part of it — checked against the actual neighbouring bytes, via the
///    *fragment* tokens (a token starting with continuation bytes, or ending in an
///    incomplete char; few, e.g. 374 in GLM-5.3, like `\x84件`).
///
/// Then no merge crosses `c`'s boundaries before `c` is whole (any such product
/// would be a vocab token covering part of `c`, by 3), so its internal merges
/// commute with everything else; and none involving `c` can fire before they
/// would have anyway (by 2) — so the merge sequence, hence the result, is the same.
#[derive(Clone)]
struct CharAtoms {
    /// Codepoint (BMP) → `rule << ATOM_ID_BITS | compact id` of its atom token,
    /// or `u32::MAX`. `rule` indexes `rules`: the fragment tokens that can cover
    /// part of the char (0: none — pre-formable anywhere). One load per char.
    bmp: Box<[u32]>,
    /// Astral codepoints (4-byte chars) → (compact id, rule index).
    astral: HashMap<u32, (u32, u32)>,
    /// Per-char fragment contexts, deduplicated; `rules[0]` is empty.
    rules: Vec<AtomRule>,
    /// Every pair of consecutive complete chars inside some vocab token, as bits
    /// of a one-hash filter over `first << 21 | second`. Two adjacent atoms whose
    /// pair is absent cannot merge — the product would be a token holding both —
    /// so a pretoken can be cut between them. A false positive only forgoes a
    /// cut, so the filter can be small enough to stay in L2.
    bigrams: Box<[u64]>,
    /// The merges whose operands are both made of whole non-ASCII chars — every
    /// merge an all-atom segment can make — in the value format of `merge_adj`:
    /// a table a fraction of the full one's size, so CJK merging stays in L2.
    merges: MergeAdjacency,
    /// Bits per 64-codepoint block of the BMP (see [`Self::pays_off`]): `good`,
    /// at least 8 atoms whose fragment checks are cheap; `fair`, checks up to 4
    /// times as costly; `some`, any atom.
    good: [u64; 16],
    fair: [u64; 16],
    some: [u64; 16],
}

/// The fragment tokens that could cover part of an atom char: `heads` are the
/// byte strings that, right before the char, complete a token ending inside it;
/// `tails` those that, right after it, complete a token starting inside it. The
/// byte sets hold their last / first bytes, so almost every occurrence is cleared
/// by one bit test. The rest compare the 8 bytes around the char against each
/// fragment as one masked word (`head_km` / `tail_km`: `(key, mask)`) — no
/// `memcmp` per fragment, of which a script like Hebrew has ~70 per letter.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct AtomRule {
    before: [u64; 4],
    after: [u64; 4],
    heads: Vec<Box<[u8]>>,
    tails: Vec<Box<[u8]>>,
    /// Heads of up to 8 bytes over the 8 bytes before the char (last byte
    /// highest): the head's bytes in the top of the word.
    head_km: Vec<(u64, u64)>,
    /// Tails of up to 8 bytes over the 8 bytes after the char (first byte lowest).
    tail_km: Vec<(u64, u64)>,
    /// The longer fragments (few), compared as slices.
    long_heads: Vec<Box<[u8]>>,
    long_tails: Vec<Box<[u8]>>,
}

impl CharAtoms {
    fn empty() -> Self {
        Self {
            bmp: Box::new([]),
            astral: HashMap::new(),
            rules: vec![AtomRule::default()],
            bigrams: vec![0; 1].into_boxed_slice(),
            merges: MergeAdjacency::empty(),
            good: [0; 16],
            fair: [0; 16],
            some: [0; 16],
        }
    }

    /// Whether `bytes` (not all ASCII) merges faster from its atoms than from its
    /// bytes, by the block of its first non-ASCII char: a CJK or Hangul char (lead
    /// byte E3..ED: the frequent ones are atoms, however sparse in their block),
    /// or a block of cheap atoms — or, the longer the run, the costlier the atoms
    /// may be (cut into segments, a long run merges in short pieces): fairly cheap
    /// ones past 16 bytes, any past LONG_ATOM_RUN.
    #[inline(always)]
    fn pays_off(&self, bytes: &[u8]) -> bool {
        let Some(k) = bytes.iter().position(|&b| b >= 0x80) else {
            return false;
        };
        let b0 = bytes[k] as u32;
        let at = |j: usize| bytes.get(k + j).map_or(0, |&b| b as u32 & 0x3F);
        let cp = match b0 {
            0xC0..=0xDF => ((b0 & 0x1F) << 6) | at(1),
            0xE3..=0xED => return true,
            0xE0..=0xEF => ((b0 & 0x0F) << 12) | (at(1) << 6) | at(2),
            _ => return bytes.len() > LONG_ATOM_RUN,
        };
        let (w, bit) = ((cp >> 12) as usize, (cp >> 6) & 63);
        self.good[w] >> bit & 1 != 0
            || (bytes.len() > 16 && self.fair[w] >> bit & 1 != 0)
            || (bytes.len() > LONG_ATOM_RUN && self.some[w] >> bit & 1 != 0)
    }

    /// Bit index of pair `key` in a filter of `bits` (a power of two) bits.
    #[inline(always)]
    fn bigram_bit(key: u64, bits: usize) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & (bits - 1)
    }

    /// Whether some vocab token may hold char `a` immediately followed by char
    /// `b` (never `false` when one does).
    #[inline(always)]
    fn joinable(&self, a: u32, b: u32) -> bool {
        let bit = Self::bigram_bit((a as u64) << 21 | b as u64, self.bigrams.len() * 64);
        // SAFETY: `bit < bigrams.len() * 64`.
        unsafe { *self.bigrams.get_unchecked(bit >> 6) >> (bit & 63) & 1 != 0 }
    }

    #[inline(always)]
    fn bit(set: &[u64; 4], k: u8) -> bool {
        set[(k >> 6) as usize] >> (k & 63) & 1 != 0
    }

    /// Longest fragment a masked word covers (longer ones decline the char's atom).
    const FRAG_MAX: usize = 8;

    /// Whether a head of `r` may end at `b[..i]`. Bytes before the text read as
    /// zero: a spurious match only declines the atom (the char then merges from
    /// its bytes), which is always exact.
    #[inline(always)]
    fn head_at(r: &AtomRule, b: &[u8], i: usize) -> bool {
        let w = if i >= 8 {
            u64::from_le_bytes(b[i - 8..i].try_into().unwrap())
        } else {
            let mut x = [0u8; 8];
            x[8 - i..].copy_from_slice(&b[..i]);
            u64::from_le_bytes(x)
        };
        r.head_km
            .iter()
            .fold(false, |hit, &(k, m)| hit | (w & m == k))
            || r.long_heads
                .iter()
                .any(|h| h.len() <= i && b[i - h.len()..i] == h[..])
    }

    /// Whether a tail of `r` may start at `b[j..]` (bytes past the text read as
    /// zero, as in [`Self::head_at`]).
    #[inline(always)]
    fn tail_at(r: &AtomRule, b: &[u8], j: usize) -> bool {
        let left = b.len() - j;
        let w = if left >= 8 {
            u64::from_le_bytes(b[j..j + 8].try_into().unwrap())
        } else {
            let mut x = [0u8; 8];
            x[..left].copy_from_slice(&b[j..]);
            u64::from_le_bytes(x)
        };
        r.tail_km
            .iter()
            .fold(false, |hit, &(k, m)| hit | (w & m == k))
            || r.long_tails
                .iter()
                .any(|t| t.len() <= left && b[j..j + t.len()] == t[..])
    }

    /// The compact id of the atom for the char at `b[i..i + len]` (a multibyte
    /// char), when pre-forming this occurrence is exact (see the type docs).
    #[inline(always)]
    fn atom_at(&self, b: &[u8], i: usize, len: usize, cp: u32) -> Option<u32> {
        let (c, rule) = if cp < 0x10000 {
            // SAFETY: `bmp` has 0x10000 entries (checked non-empty by the caller).
            let e = unsafe { *self.bmp.get_unchecked(cp as usize) };
            if e == u32::MAX {
                return None;
            }
            (e & ATOM_ID_MASK, e >> ATOM_ID_BITS)
        } else {
            *self.astral.get(&cp)?
        };
        if rule != 0 {
            // SAFETY: rule indices come from `rules`.
            let r = unsafe { self.rules.get_unchecked(rule as usize) };
            let j = i + len;
            if i > 0 && Self::bit(&r.before, b[i - 1]) && Self::head_at(r, b, i) {
                return None;
            }
            if j < b.len() && Self::bit(&r.after, b[j]) && Self::tail_at(r, b, j) {
                return None;
            }
        }
        Some(c)
    }
}

/// Bytes past which a run merges from its atoms wherever its script has any
/// (see [`CharAtoms::pays_off`]).
const LONG_ATOM_RUN: usize = 32;

/// A harvested span awaiting its (prefetched) cache probe in the fused encode
/// loop, with the hash its prefetch used (32 bytes either way: it fills the key's
/// padding). `key == LONG_KEY`: not inline-cacheable (> 15 B); `hash` is then the
/// [`LongCache::hash`].
#[derive(Clone, Copy, Default)]
struct Pending {
    key: u128,
    hash: u64,
    start: u32,
    end: u32,
}

/// The [`Pending::key`] of a span too long for an inline key: no table entry
/// holds it (a real key's top byte is its length, at most 15; an empty slot's
/// key is 0).
const LONG_KEY: u128 = u128::MAX;

/// Spans harvested per [`Bpe::probe_emit`] call.
const SPAN_CHUNK: usize = 256;

/// Output slots [`Bpe::probe_emit`] reserves for a full chunk of spans (the
/// inline ids of each, stored before its hit is known).
pub(crate) const SPAN_CHUNK_RESERVE: usize = PT_INLINE * SPAN_CHUNK;

/// Spans of prefetch look-ahead in [`Bpe::probe_emit`].
const PROBE_AHEAD: usize = 16;

/// Spans ahead at which [`Bpe::probe_emit`] peeks at a span's (by then
/// prefetched) cache slot: a predicted miss gets its [`FoldTable`] lines
/// prefetched, so the probe a miss starts with is not a cold load. Smaller than
/// [`PROBE_AHEAD`], so the peeked slot has arrived.
const PEEK_AHEAD: usize = 8;

/// A chunk of harvested spans (see [`Bpe::tokenize_scanned_segment`]), with
/// [`PROBE_AHEAD`] slack entries past a full chunk: the probe loop's look-ahead
/// reads them without a bound check (stale or default entries only prefetch some
/// line of the table).
struct SpanChunk {
    p: [Pending; SPAN_CHUNK + PROBE_AHEAD],
    n: usize,
}

impl SpanChunk {
    fn new() -> Self {
        Self {
            p: [Pending::default(); SPAN_CHUNK + PROBE_AHEAD],
            n: 0,
        }
    }

    /// Append the span `sb[start..end]` (non-empty): pack its key and hash it
    /// (a long span's long-cache slot is prefetched: it is probed out of line).
    /// `n < SPAN_CHUNK`. A short span's cache line is prefetched into L2 only —
    /// a chunk (hundreds of cycles) before its probe, enough to cover a DRAM
    /// miss; the probe loop's short look-ahead then lifts it into L1. (Into L1
    /// here, a chunk's lines would evict the walker's working set.)
    #[inline(always)]
    fn harvest(&mut self, cache: &PretokenCache, sb: &[u8], start: usize, end: usize) {
        let len = end - start;
        let (key, hash) = if len <= 15 {
            let key = PretokenCache::pack_key_at(sb, start, len);
            let hash = PretokenCache::hash(key);
            // SAFETY: `hash & mask < slots.len()`; a prefetch reads nothing.
            prefetch_l2(
                unsafe { cache.slots.as_ptr().add(hash as usize & cache.mask) } as *const u8,
            );
            (key, hash)
        } else if long_uncached(sb[start]) {
            (LONG_KEY, 0)
        } else {
            let hash = LongCache::hash(&sb[start..end]);
            cache.long.prefetch(hash);
            (LONG_KEY, hash)
        };
        // SAFETY: the caller keeps `n < SPAN_CHUNK`.
        unsafe {
            *self.p.get_unchecked_mut(self.n) = Pending {
                key,
                hash,
                start: start as u32,
                end: end as u32,
            }
        };
        self.n += 1;
    }
}

/// Whole-pretoken vocab table for `ignore_merges` models: raw pretoken bytes
/// (≤ 15, packed like the pretoken-cache key) → the vocab entry they spell.
///
/// Under `ignore_merges` a pretoken that is itself a vocab entry encodes to that
/// entry, merges or not. The generic check ByteLevel-encodes the pretoken into a
/// `String` and hashes it into the vocab map on every cache miss; this answers it
/// with one probe keyed by the hash the pretoken cache already computed, and —
/// holding *every* entry of <= 15 raw bytes — a failed probe is a definitive "not
/// a token" (see [`Bpe::process_miss`]). Buckets are one cache line of four keys;
/// ids live in a parallel array read only on a hit.
///
/// (A merge-proven variant for ordinary models — keep an entry iff merging its
/// bytes yields it, as `tk-encode`'s `prove_fold` — measured neutral to negative:
/// the probe on every miss costs about what the hits save.)
#[derive(Clone)]
struct FoldTable {
    shift: u32,
    mask: usize,
    keys: HugeTable<FoldBucket>,
    ids: HugeTable<[u32; 4]>,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct FoldBucket {
    k: [u128; 4],
}

impl FoldTable {
    fn empty() -> Self {
        Self {
            shift: 64,
            mask: 0,
            keys: HugeTable::new(1, FoldBucket { k: [0; 4] }),
            ids: HugeTable::new(1, [0; 4]),
        }
    }

    #[inline(always)]
    fn home(&self, hash: u64) -> usize {
        if self.mask == 0 {
            0
        } else {
            (hash >> self.shift) as usize
        }
    }

    /// Build from `(packed key, token)` pairs (keys distinct and nonzero).
    fn from_entries(entries: &[(u128, u32)]) -> Self {
        if entries.is_empty() {
            return Self::empty();
        }
        let nb = (entries.len() * 2).div_ceil(5).next_power_of_two().max(1);
        let bits = nb.trailing_zeros();
        let mut t = Self {
            shift: 64 - bits,
            mask: nb - 1,
            keys: HugeTable::new(nb, FoldBucket { k: [0; 4] }),
            ids: HugeTable::new(nb, [0; 4]),
        };
        for &(k, id) in entries {
            let mut b = t.home(PretokenCache::hash(k));
            'place: loop {
                for i in 0..4 {
                    if t.keys[b].k[i] == 0 {
                        t.keys[b].k[i] = k;
                        t.ids[b][i] = id;
                        break 'place;
                    }
                }
                b = (b + 1) & t.mask;
            }
        }
        t
    }

    /// Prefetch the lines a lookup of `hash` reads first: its home bucket's keys
    /// and ids (the ids are read on a hit, which is most lookups).
    #[inline(always)]
    fn prefetch(&self, hash: u64) {
        let b = self.home(hash);
        // SAFETY: `b <= mask < keys.len() == ids.len()`; a prefetch reads nothing.
        unsafe {
            prefetch_read(self.keys.as_ptr().add(b) as *const u8);
            prefetch_read(self.ids.as_ptr().add(b) as *const u8);
        }
    }

    /// The folded token for a packed pretoken key, if it is a proven single token.
    #[inline(always)]
    fn get(&self, key: u128, hash: u64) -> Option<u32> {
        let mut b = self.home(hash);
        loop {
            // SAFETY: `b <= mask < keys.len() == ids.len()`.
            let bk = unsafe { self.keys.get_unchecked(b) };
            for i in 0..4 {
                if bk.k[i] == key {
                    return Some(unsafe { self.ids.get_unchecked(b)[i] });
                }
            }
            if bk.k[3] == 0 {
                return None;
            }
            b = (b + 1) & self.mask;
        }
    }
}

/// Bits of [`PairFilter`]: 2^20 bits = 128 KiB. Small enough to stay in L2
/// between misses: at 512 KiB its words were evicted by the cache and grid lines
/// in between, and each check waited on L3 (GLM English: 24% of merge time).
const PAIR_FILTER_BITS: u32 = 20;

/// Which pairs outside [`Bpe::code_grid`] merge, as two hashed bits in one word
/// each (a blocked Bloom filter; a false positive only costs the table probe it
/// would have cost anyway — ~20% at GLM's ~320k merges). A cold word's lookups
/// past the grid mostly find no merge (~75% on GLM English), and the filter
/// answers those from L2 instead of the multi-MiB pair table.
#[derive(Clone)]
struct PairFilter {
    bits: Box<[u64]>,
}

impl PairFilter {
    fn new(keys: impl Iterator<Item = u64>) -> Self {
        let mut bits = vec![0u64; 1 << (PAIR_FILTER_BITS - 6)].into_boxed_slice();
        for k in keys {
            let (w, m) = Self::bit(k);
            bits[w] |= m;
        }
        Self { bits }
    }

    /// The word of a key and its two bits there.
    #[inline(always)]
    fn bit(key: u64) -> (usize, u64) {
        let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let w = (h >> (64 - (PAIR_FILTER_BITS - 6))) as usize;
        (w, (1u64 << (h & 63)) | (1u64 << ((h >> 6) & 63)))
    }

    /// Whether the pair `key` (`MergeAdjacency::packed_key`) may merge.
    #[inline(always)]
    fn may_merge(&self, key: u64) -> bool {
        let (w, m) = Self::bit(key);
        // SAFETY: `w < 2^(PAIR_FILTER_BITS - 6) == bits.len()`.
        unsafe { *self.bits.get_unchecked(w) & m == m }
    }
}

/// Merge-pair table: `(left, right) -> (rank, new_id)` for every merge.
///
/// The merge engines probe this for pairs the dense `merge_grid` doesn't cover —
/// on cold (first-seen) words that is ~45% of lookups — and most probes are for
/// pairs that *don't* merge. A CSR adjacency list answered with a binary search
/// over the left operand's neighbours: a chain of ~log₂(degree) dependent loads.
/// This is a hash table whose buckets are one 64-byte cache line of 4 slots,
/// sized for ≤ 2.5 keys per bucket, so a lookup — hit or miss — almost always
/// resolves in a single cache line (the role `tk-encode`'s perfect hash plays).
/// Slots fill in order, so a bucket with an empty last slot never overflowed:
/// reaching it proves the key is absent.
#[derive(Clone)]
struct MergeAdjacency {
    /// Bucket count (any; a key's home is the high product of its hash with it).
    nb: usize,
    buckets: HugeTable<PairBucket>,
    /// When the ranks are [`rank_code`]s: one `u64` per merge (`left << 18 |
    /// right` above a 22-bit rank code that also names the product), eight to a
    /// cache line — half the footprint of `buckets`, which is then empty.
    packed: HugeTable<PackedBucket>,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct PackedBucket {
    e: [u64; 8],
}

/// Bits of a packed entry below the key: the rank code.
const PACKED_CODE_BITS: u32 = PACKED_ID_BITS + RANK_SUB_BITS;
// Key (two ids) above the code fits a u64, and a code is a valid small-merge rank.
const _: () =
    assert!(2 * PACKED_ID_BITS + PACKED_CODE_BITS <= 64 && PACKED_CODE_BITS <= 32 - SMALL_KEY_BITS);

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct PairBucket {
    keys: [u64; 4],
    /// `rank << 32 | new_id`.
    vals: [u64; 4],
}

const PAIR_EMPTY: u64 = u64::MAX;

impl MergeAdjacency {
    /// A table without merges.
    fn empty() -> Self {
        Self {
            nb: 1,
            buckets: HugeTable::new(
                0,
                PairBucket {
                    keys: [PAIR_EMPTY; 4],
                    vals: [0; 4],
                },
            ),
            packed: HugeTable::new(1, PackedBucket { e: [PAIR_EMPTY; 8] }),
        }
    }

    /// The [`rank_code`] of a pair of compact ids (packed layout), or `u32::MAX`.
    #[inline(always)]
    fn get_code(&self, left: u32, right: u32) -> u32 {
        if (left | right) >> PACKED_ID_BITS != 0 {
            return u32::MAX;
        }
        let key = Self::packed_key(left, right);
        let mut b = self.home(key);
        loop {
            // SAFETY: `b <= mask < packed.len()`.
            let bk = unsafe { self.packed.get_unchecked(b) };
            let mut code = u32::MAX;
            for &e in &bk.e {
                code = if e >> PACKED_CODE_BITS == key {
                    (e & ((1 << PACKED_CODE_BITS) - 1)) as u32
                } else {
                    code
                };
            }
            if code != u32::MAX || bk.e[7] == PAIR_EMPTY {
                return code;
            }
            b = self.next(b);
        }
    }

    /// Prefetch the home bucket of a pair (packed layout).
    #[inline(always)]
    fn prefetch(&self, left: u32, right: u32) {
        if let Some(bk) = self.packed.get(self.home(Self::packed_key(left, right))) {
            prefetch_read(bk as *const PackedBucket as *const u8);
        }
    }

    /// Packed value for a pair of compact ids, or `u64::MAX` (either layout).
    #[inline(always)]
    fn lookup(&self, left: u32, right: u32) -> u64 {
        if self.buckets.is_empty() {
            self.get_packed_compact(left, right)
        } else {
            self.get_packed(left, right)
        }
    }

    /// Keys and products are *compact* ids (see [`build_compact_grid`]).
    fn from_parsed(
        parsed: &ParsedMergeMap,
        compact: &[u32],
        min_rank: &[u32],
        code: Option<&[u32]>,
    ) -> Self {
        if let Some(first) = code {
            // Half full (4 of 8 slots on average; any count works, see `home`):
            // an overflowing bucket costs a second line, a sparser table more
            // of L3 (GLM: 5 MiB, where the power of two above was 8).
            let nb = (parsed.len().max(1) * 10).div_ceil(40);
            return Self::packed_from(parsed, compact, first, nb);
        }
        let n = parsed.len().max(1);
        let nb = (n * 2).div_ceil(5).next_power_of_two().max(1);
        let mut t = Self {
            nb,
            buckets: HugeTable::new(
                nb,
                PairBucket {
                    keys: [PAIR_EMPTY; 4],
                    vals: [0; 4],
                },
            ),
            packed: HugeTable::new(0, PackedBucket { e: [PAIR_EMPTY; 8] }),
        };
        for (&(left, right), &(rank, new_id)) in parsed {
            t.insert(
                Self::key(compact[left as usize], compact[right as usize]),
                pack_merge_value(min_rank, compact, None, rank, new_id),
            );
        }
        t
    }

    /// The packed layout over `nb` buckets.
    fn packed_from(parsed: &ParsedMergeMap, compact: &[u32], first: &[u32], nb: usize) -> Self {
        let nb = nb.max(parsed.len().div_ceil(8) + 1);
        let mut t = Self {
            nb,
            buckets: HugeTable::new(
                0,
                PairBucket {
                    keys: [PAIR_EMPTY; 4],
                    vals: [0; 4],
                },
            ),
            packed: HugeTable::new(nb, PackedBucket { e: [PAIR_EMPTY; 8] }),
        };
        for (&(left, right), &(rank, product)) in parsed {
            let key = Self::packed_key(compact[left as usize], compact[right as usize]);
            let e = key << PACKED_CODE_BITS | rank_code(compact, first, rank, product) as u64;
            let mut b = t.home(key);
            'place: loop {
                for slot in &mut t.packed[b].e {
                    if *slot == PAIR_EMPTY {
                        *slot = e;
                        break 'place;
                    }
                }
                b = t.next(b);
            }
        }
        t
    }

    #[inline(always)]
    fn packed_key(left: u32, right: u32) -> u64 {
        ((left as u64) << PACKED_ID_BITS) | right as u64
    }

    /// [`Self::get_packed`] for the packed layout: the same `rank << 32 | product`
    /// value, rebuilt from the entry's rank code.
    #[inline(always)]
    fn get_packed_compact(&self, left: u32, right: u32) -> u64 {
        if (left | right) >> PACKED_ID_BITS != 0 {
            return u64::MAX;
        }
        let key = Self::packed_key(left, right);
        let mut b = self.home(key);
        loop {
            // SAFETY: `b <= mask < packed.len()`.
            let bk = unsafe { self.packed.get_unchecked(b) };
            // Keys are unique: select the matching slot branch-free.
            let mut code = u64::MAX;
            for &e in &bk.e {
                code = if e >> PACKED_CODE_BITS == key {
                    e & ((1 << PACKED_CODE_BITS) - 1)
                } else {
                    code
                };
            }
            if code != u64::MAX {
                return (code << 32) | (code >> RANK_SUB_BITS);
            }
            if bk.e[7] == PAIR_EMPTY {
                return u64::MAX;
            }
            b = self.next(b);
        }
    }

    #[inline(always)]
    fn key(left: u32, right: u32) -> u64 {
        ((left as u64) << 32) | right as u64
    }

    #[inline(always)]
    fn home(&self, key: u64) -> usize {
        // Fold then Fibonacci-hash; the high product with the bucket count picks
        // the bucket (for a power of two, simply the top bits of the hash).
        let h = (key ^ (key >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((h as u128 * self.nb as u128) >> 64) as usize
    }

    /// The bucket after `b`, wrapping.
    #[inline(always)]
    fn next(&self, b: usize) -> usize {
        if b + 1 == self.nb { 0 } else { b + 1 }
    }

    fn insert(&mut self, key: u64, val: u64) {
        let mut b = self.home(key);
        loop {
            let bk = &mut self.buckets[b];
            for i in 0..4 {
                if bk.keys[i] == PAIR_EMPTY {
                    bk.keys[i] = key;
                    bk.vals[i] = val;
                    return;
                }
            }
            b = self.next(b);
        }
    }

    /// Packed `rank << 32 | product` (compact ids) for the pair, or `u64::MAX`.
    #[inline(always)]
    fn get_packed(&self, left: u32, right: u32) -> u64 {
        let key = Self::key(left, right);
        if key == PAIR_EMPTY {
            return u64::MAX; // never a real pair (both operands u32::MAX)
        }
        let mut b = self.home(key);
        loop {
            // SAFETY: `b <= mask < buckets.len()`.
            let bk = unsafe { self.buckets.get_unchecked(b) };
            // Keys are unique, so at most one slot matches: select it branch-free
            // (which slot holds the key is unpredictable — an early-exit loop
            // mispredicts on most lookups).
            let mut v = u64::MAX;
            for i in 0..4 {
                v = if bk.keys[i] == key { bk.vals[i] } else { v };
            }
            // Found, or the bucket has a free slot (the key would be in it): done.
            // Only a full bucket without the key continues to the next one.
            if v != u64::MAX || bk.keys[3] == PAIR_EMPTY {
                return v;
            }
            b = self.next(b);
        }
    }
}

/// Lowest rank of any merge in which each external id is an operand.
fn min_rank_involving(merge_map: &ParsedMergeMap, vocab_size: usize) -> Vec<u32> {
    let mut m = vec![u32::MAX; vocab_size];
    for (&(a, b), &(rank, _)) in merge_map.iter() {
        for t in [a, b] {
            if let Some(slot) = m.get_mut(t as usize) {
                *slot = (*slot).min(rank);
            }
        }
    }
    m
}

/// Packed merge value (see [`PV_SAFE`]) for a merge `rank → product` (external
/// product id). `min_rank` is [`min_rank_involving`] of the vocab being built: the
/// merge is `SAFE` when its product takes part in no lower-ranked merge.
fn pack_merge_value(
    min_rank: &[u32],
    compact: &[u32],
    code: Option<&[u32]>,
    rank: u32,
    product: u32,
) -> u64 {
    let safe = rank < min_rank.get(product as usize).copied().unwrap_or(u32::MAX);
    let r = match code {
        Some(first) => rank_code(compact, first, rank, product),
        None => rank,
    };
    ((r as u64) << 32) | if safe { PV_SAFE } else { 0 } | compact[product as usize] as u64
}

/// Bits of a [`rank_code`] below the product: a product may be the result of at
/// most `1 << RANK_SUB_BITS` merges.
const RANK_SUB_BITS: u32 = 7;
/// Compact ids (operands and products) must fit this many bits for the packed
/// pair table and rank code.
const PACKED_ID_BITS: u32 = 18;

/// The rank of a merge re-expressed as `compact product << RANK_SUB_BITS | its
/// index among the merges producing that product` (`first[product]` = the lowest
/// rank producing it). In a merge list converted from tiktoken ranks (GLM,
/// Llama-3 style) a token's merges are contiguous and ordered by the token, so
/// this orders merges exactly as their ranks do ([`rank_codes_valid`] checks it)
/// while *carrying the product*: the packed pair table then stores the code alone.
#[inline]
fn rank_code(compact: &[u32], first: &[u32], rank: u32, product: u32) -> u32 {
    (compact[product as usize] << RANK_SUB_BITS) | (rank - first[product as usize])
}

/// `first` (lowest rank producing each external token) when [`rank_code`] is
/// order-equivalent to the ranks and every id fits the packed layout, else `None`.
fn rank_codes_valid(merge_map: &ParsedMergeMap, compact: &[u32]) -> Option<Vec<u32>> {
    let mut first = vec![u32::MAX; compact.len()];
    for &(rank, product) in merge_map.values() {
        let f = first.get_mut(product as usize)?;
        *f = (*f).min(rank);
    }
    let lim = 1u32 << PACKED_ID_BITS;
    let mut by_rank: Vec<(u32, u32)> = Vec::with_capacity(merge_map.len());
    for (&(l, r), &(rank, product)) in merge_map {
        let (cl, cr, cp) = (
            *compact.get(l as usize)?,
            *compact.get(r as usize)?,
            compact[product as usize],
        );
        if cl >= lim
            || cr >= lim
            || cp >= lim
            || rank - first[product as usize] >= 1 << RANK_SUB_BITS
        {
            return None;
        }
        by_rank.push((rank, rank_code(compact, &first, rank, product)));
    }
    by_rank.sort_unstable();
    by_rank.windows(2).all(|w| w[0].1 < w[1].1).then_some(first)
}

/// Vocab entries of more than 15 raw bytes by [`LongCache::hash`]: `(bytes, id)`.
type LongVocab = HashMap<u64, Vec<(Box<[u8]>, u32)>>;

/// [`build_compact_grid`]'s result: the compact-id map, its inverse, the dense
/// merge grid, and — when the merge list allows it — the lowest rank producing
/// each token (see [`rank_codes_valid`]).
type CompactGrid = (Vec<u32>, Box<[u32]>, HugeTable<u64>, Option<Vec<u32>>);

/// Build the compact-id map and the dense merge grid (see [`Bpe::merge_grid`]).
///
/// Compact ids go to byte atoms first (they dominate merge operands), then to
/// merge products in ascending rank (most frequent first), mirroring v1.0.0
/// `tokenizers`' internal-id ordering so the hottest pairs occupy the low
/// indices that fit the grid. Every merge whose two operands both land
/// `< MERGE_GRID_DIM` is written into the grid; the rest stay in `merge_adj`.
fn build_compact_grid(
    merge_map: &ParsedMergeMap,
    byte_to_initial_token: &[u32; 256],
    vocab_size: usize,
    min_rank: &[u32],
) -> CompactGrid {
    let mut compact = vec![u32::MAX; vocab_size];
    let mut next = 0u32;
    // Byte atoms in token-id order (not byte order): in a vocab laid out as the
    // bytes then the merge products (tiktoken-derived ones like GLM's and Kimi's,
    // or DeepSeek's after its leading specials), compact ids are then the
    // external ids less a constant (see `Bpe::unmap_offset`).
    let mut byte_ids: Vec<u32> = byte_to_initial_token
        .iter()
        .copied()
        .filter(|&id| id != INVALID_TOKEN && (id as usize) < vocab_size)
        .collect();
    byte_ids.sort_unstable();
    byte_ids.dedup();
    for id in byte_ids {
        compact[id as usize] = next;
        next += 1;
    }
    // Each product keeps the minimum rank across the merges that produce it,
    // then products are assigned compact ids in that rank order.
    let mut lowest_rank: HashMap<u32, u32> = HashMap::new();
    for &(rank, product) in merge_map.values() {
        lowest_rank
            .entry(product)
            .and_modify(|r| *r = (*r).min(rank))
            .or_insert(rank);
    }
    let mut products: Vec<(u32, u32)> = lowest_rank.into_iter().map(|(p, r)| (r, p)).collect();
    products.sort_unstable();
    for (_, product) in products {
        if (product as usize) < vocab_size && compact[product as usize] == u32::MAX {
            compact[product as usize] = next;
            next += 1;
        }
    }
    // Any merge operand that is neither a byte atom nor a product still gets a
    // compact id, so every merge is addressable in compact space.
    let mut leftover: Vec<u32> = merge_map
        .keys()
        .flat_map(|&(a, b)| [a, b])
        .filter(|&t| (t as usize) < vocab_size && compact[t as usize] == u32::MAX)
        .collect();
    leftover.sort_unstable();
    leftover.dedup();
    for t in leftover {
        compact[t as usize] = next;
        next += 1;
    }
    let mut unmap = vec![INVALID_TOKEN; next as usize];
    for (ext, &c) in compact.iter().enumerate() {
        if c != u32::MAX {
            unmap[c as usize] = ext as u32;
        }
    }
    let code = rank_codes_valid(merge_map, &compact);
    let dim = MERGE_GRID_DIM as usize;
    let mut grid = HugeTable::new(dim * dim, u64::MAX);
    for (&(a, b), &(rank, product)) in merge_map.iter() {
        let (ca, cb) = (compact[a as usize], compact[b as usize]);
        if ca < MERGE_GRID_DIM && cb < MERGE_GRID_DIM {
            grid[ca as usize * dim + cb as usize] =
                pack_merge_value(min_rank, &compact, code.as_deref(), rank, product);
        }
    }
    (compact, unmap.into_boxed_slice(), grid, code)
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
    /// Cross-thread second level of the fused path's pretoken cache (lazy).
    shared_pt: OnceLock<SharedPretokenCache>,
    /// The fused path's pretoken-cache seed (lazy; `None`: nothing to seed).
    pt_seed: OnceLock<Option<Arc<PtSeed>>>,
    id_to_token: Vec<String>,
    token_to_id: FxHashMap<String, u32>,
    byte_to_initial_token: [u32; 256],
    byte_fallback_token_ids: [u32; 256],
    /// Token id for each single ASCII-character string (`INVALID_TOKEN` when
    /// absent). Fast path for the char-based merge engine, avoiding a HashMap
    /// probe per character.
    single_char_token: [u32; 128],
    /// Byte pair → packed `rank << 32 | product` (compact ids), or `u64::MAX`:
    /// the first merge round of every raw pretoken, one direct load per pair.
    pair_initial: HugeTable<u64>,
    /// Byte → compact id of its initial (ByteLevel) token, or `INVALID_TOKEN`.
    byte_compact: [u32; 256],
    /// Compact id → external token id.
    unmap: Box<[u32]>,
    /// Whether any merge lacks the `SAFE` bit (else every sweep batches).
    any_unsafe: bool,
    /// Proven single-token words (see [`FoldTable`]).
    fold: FoldTable,
    /// Multibyte chars pre-formed as one merge symbol (see [`CharAtoms`]).
    atoms: CharAtoms,
    /// `ignore_merges` models: every vocab entry of more than 15 raw bytes, by
    /// [`LongCache::hash`] — the long-pretoken counterpart of [`Self::fold`], so the
    /// whole-pretoken lookup of a long span is one probe with a hash already in hand.
    long_vocab: LongVocab,
    merge_adj: MergeAdjacency,
    /// External token id -> compact ("internal") id, or `u32::MAX` for tokens
    /// that never appear as a merge operand. Byte atoms and the lowest-rank
    /// (most frequent) merge products get the smallest compact ids, so the
    /// hottest pairs land in the dense `merge_grid`. See [`Bpe::merge_lookup`].
    compact_id: Vec<u32>,
    /// Dense `MERGE_GRID_DIM × MERGE_GRID_DIM` grid of packed merge values for
    /// pairs whose *both* compact ids are `< MERGE_GRID_DIM`. Cell holds
    /// `rank << 32 | merged_external_id`, or `u64::MAX` for "no merge". A lookup
    /// is one direct-indexed load — no hash, no binary search — which is what
    /// v1.0.0 `tokenizers` buys with its internal-id remap + 512×512 grid.
    merge_grid: HugeTable<u64>,
    /// With valid [`rank_code`]s, a pair's whole packed value follows from its
    /// code (`rank = code`, product `= code >> RANK_SUB_BITS`), so a 4-byte cell
    /// holds it: this `CODE_GRID_DIM²` grid of codes (`u32::MAX`: no merge)
    /// covers the pairs of the lowest `CODE_GRID_DIM` (1024) compact ids, where
    /// `merge_grid` covers 512. Cold words' mid-merge lookups mostly land here;
    /// the pairs of one frequent left symbol share a row, where the pair table
    /// scatters them.
    code_grid: Option<HugeTable<u32>>,
    /// With `code_grid`: the merges outside it, filtered ([`PairFilter`]).
    pair_filter: Option<PairFilter>,
    /// `Some(k)` when every compact id `c` is external id `c + k` (see
    /// `build_compact_grid`): merge output then skips the `unmap` load.
    unmap_offset: Option<u32>,
    ignore_merges: bool,
    byte_fallback: bool,
    pub bigram_bridge_table: BigramBridgeTable,
}

impl TryFrom<RawBpe> for Bpe {
    type Error = String;

    fn try_from(raw: RawBpe) -> Result<Self> {
        let merge_map = parse_merges(&raw.vocab, &raw.merges)?;
        let mut bpe = Self::new_inner(&raw.vocab, merge_map, raw.ignore_merges)?;
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
        Self::new_inner(vocab, merge_map, false)
    }

    /// [`Self::new`] with the `ignore_merges` flag known up front, so the fold (whose
    /// contents depend on it — see [`Self::build_fold`]) is built once.
    fn new_inner(vocab: &Vocab, merge_map: ParsedMergeMap, ignore_merges: bool) -> Result<Self> {
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

        let mut single_char_token = [INVALID_TOKEN; 128];
        for (byte, slot) in single_char_token.iter_mut().enumerate() {
            let ch = byte as u8 as char;
            let mut buf = [0u8; 1];
            if let Some(&id) = vocab.get(ch.encode_utf8(&mut buf) as &str) {
                *slot = id;
            }
        }

        let vocab_size = id_to_token.len();
        // Small-merge keys pack `rank << SMALL_KEY_BITS | position` into a u32.
        if merge_map
            .values()
            .any(|&(rank, _)| rank >= 1 << (32 - SMALL_KEY_BITS))
        {
            return Err(format!(
                "more than 2^{} merges are not supported",
                32 - SMALL_KEY_BITS
            ));
        }
        let min_rank = min_rank_involving(&merge_map, vocab_size);
        let any_unsafe = merge_map.values().any(|&(rank, product)| {
            rank >= min_rank.get(product as usize).copied().unwrap_or(u32::MAX)
        });
        let (compact_id, unmap, merge_grid, code) =
            build_compact_grid(&merge_map, &byte_to_initial_token, vocab_size, &min_rank);
        let merge_adj =
            MergeAdjacency::from_parsed(&merge_map, &compact_id, &min_rank, code.as_deref());
        let unmap_offset = unmap.first().copied().filter(|&k| {
            k != INVALID_TOKEN
                && unmap
                    .iter()
                    .enumerate()
                    .all(|(c, &e)| e != INVALID_TOKEN && e == c as u32 + k)
        });
        let pair_filter = code.is_some().then(|| {
            let lim = 1u32 << PACKED_ID_BITS;
            PairFilter::new(merge_map.keys().filter_map(|&(a, b)| {
                let (ca, cb) = (compact_id[a as usize], compact_id[b as usize]);
                ((ca >= CODE_GRID_DIM || cb >= CODE_GRID_DIM) && ca < lim && cb < lim)
                    .then(|| MergeAdjacency::packed_key(ca, cb))
            }))
        });
        let code_grid = code.as_deref().map(|first| {
            let dim = CODE_GRID_DIM as usize;
            let mut g = HugeTable::new(dim * dim, u32::MAX);
            for (&(a, b), &(rank, product)) in merge_map.iter() {
                let (ca, cb) = (compact_id[a as usize], compact_id[b as usize]);
                if ca < CODE_GRID_DIM && cb < CODE_GRID_DIM {
                    g[ca as usize * dim + cb as usize] =
                        rank_code(&compact_id, first, rank, product);
                }
            }
            g
        });
        let mut byte_compact = [INVALID_TOKEN; 256];
        for (b, &t) in byte_to_initial_token.iter().enumerate() {
            if t != INVALID_TOKEN {
                byte_compact[b] = compact_id[t as usize];
            }
        }
        // First-round byte-pair merges (256×256), packed in compact ids.
        let mut pair_initial = HugeTable::new(65536, u64::MAX);
        for b1 in 0..256usize {
            let t1 = byte_to_initial_token[b1];
            if t1 == INVALID_TOKEN {
                continue;
            }
            for b2 in 0..256usize {
                let t2 = byte_to_initial_token[b2];
                if t2 == INVALID_TOKEN {
                    continue;
                }
                if let Some(&(rank, new_id)) = merge_map.get(&(t1, t2)) {
                    pair_initial[b1 * 256 + b2] =
                        pack_merge_value(&min_rank, &compact_id, code.as_deref(), rank, new_id);
                }
            }
        }

        let bigram_bridge_table = build_bigram_bridge_table(&id_to_token);

        let mut bpe = Self {
            id: BPE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            daac,
            merge_map: flat_merge_map,
            unmerge_map,
            next_prefix_map,
            token_lens,
            shared_cache: SharedCache::new(),
            shared_pt: OnceLock::new(),
            pt_seed: OnceLock::new(),
            id_to_token,
            token_to_id: {
                let mut m = HashMap::with_capacity_and_hasher(vocab.len(), FxBuildHasher);
                m.extend(vocab.iter().map(|(k, v)| (k.clone(), *v)));
                m
            },
            byte_to_initial_token,
            byte_fallback_token_ids,
            single_char_token,
            pair_initial,
            byte_compact,
            unmap,
            any_unsafe,
            fold: FoldTable::empty(),
            atoms: CharAtoms::empty(),
            long_vocab: HashMap::new(),
            merge_adj,
            compact_id,
            merge_grid,
            code_grid,
            pair_filter,
            unmap_offset,
            ignore_merges,
            byte_fallback: false,
            bigram_bridge_table,
        };
        if ignore_merges {
            bpe.fold = bpe.build_fold();
            bpe.long_vocab = bpe.build_long_vocab();
        }
        bpe.atoms = bpe.build_char_atoms(&merge_map);
        if !bpe.atoms.bmp.is_empty() {
            let mut inv = [u16::MAX; 324];
            for (b, &c) in BYTE_TO_CHAR.iter().enumerate() {
                inv[c as usize] = b as u16;
            }
            // Per token: whether its raw bytes are whole chars, none of them ASCII.
            let pure: Vec<bool> = bpe
                .id_to_token
                .iter()
                .map(|t| {
                    let raw: Option<Vec<u8>> = t
                        .chars()
                        .map(|ch| {
                            inv.get(ch as usize)
                                .copied()
                                .filter(|&v| v != u16::MAX)
                                .map(|v| v as u8)
                        })
                        .collect();
                    raw.is_some_and(|r| {
                        !r.is_empty()
                            && r.iter().all(|&b| b >= 0x80)
                            && std::str::from_utf8(&r).is_ok()
                    })
                })
                .collect();
            let is_pure = |t: u32| pure.get(t as usize).copied().unwrap_or(false);
            let sub: ParsedMergeMap = merge_map
                .iter()
                .filter(|&(&(a, b), _)| is_pure(a) && is_pure(b))
                .map(|(&k, &v)| (k, v))
                .collect();
            bpe.atoms.merges = match code.as_deref() {
                // <= 60% load, any bucket count: as small as it can be.
                Some(first) => MergeAdjacency::packed_from(
                    &sub,
                    &bpe.compact_id,
                    first,
                    (sub.len() * 10).div_ceil(48),
                ),
                None => MergeAdjacency::from_parsed(&sub, &bpe.compact_id, &min_rank, None),
            };
        }
        Ok(bpe)
    }

    /// Build the [`CharAtoms`]: the fragment tokens, then every single-char token
    /// meeting conditions 1 and 2 of the type docs.
    fn build_char_atoms(&self, merge_map: &ParsedMergeMap) -> CharAtoms {
        let mut inv = [u16::MAX; 324];
        for (b, &c) in BYTE_TO_CHAR.iter().enumerate() {
            inv[c as usize] = b as u16;
        }
        let raw_of = |tok: &str| -> Option<Vec<u8>> {
            tok.chars()
                .map(|ch| {
                    inv.get(ch as usize)
                        .copied()
                        .filter(|&v| v != u16::MAX)
                        .map(|v| v as u8)
                })
                .collect()
        };
        // Fragment tokens: leading continuation bytes → the rest; trailing incomplete
        // char → the bytes before it. (A token with both is in both maps.)
        let mut tails: HashMap<Vec<u8>, Vec<Box<[u8]>>> = HashMap::new();
        let mut heads: HashMap<Vec<u8>, Vec<Box<[u8]>>> = HashMap::new();
        let mut candidates: Vec<(u32, u32, Vec<u8>)> = Vec::new(); // (token, codepoint, bytes)
        for (t, tok) in self.id_to_token.iter().enumerate() {
            let Some(r) = raw_of(tok) else { continue };
            if r.is_empty() {
                continue;
            }
            let lead = r.iter().take_while(|&&x| x & 0xC0 == 0x80).count().min(3);
            if lead > 0 && lead < r.len() {
                tails
                    .entry(r[..lead].to_vec())
                    .or_default()
                    .push(r[lead..].into());
            }
            if let Some(p) = r.iter().rposition(|&x| x & 0xC0 != 0x80) {
                let need = match r[p] {
                    0xC0..=0xDF => 2,
                    0xE0..=0xEF => 3,
                    0xF0..=0xF7 => 4,
                    _ => 1,
                };
                if r.len() - p < need && p > 0 {
                    heads
                        .entry(r[p..].to_vec())
                        .or_default()
                        .push(r[..p].into());
                }
            }
            if (2..=4).contains(&r.len())
                && let Ok(sch) = std::str::from_utf8(&r)
                && sch.chars().count() == 1
            {
                candidates.push((t as u32, sch.chars().next().unwrap() as u32, r));
            }
        }
        // Lowest rank of a merge using each token as an operand — in the engine's
        // rank space (`merge_lookup`), which is also what the formation ranks below use.
        let mut min_op = vec![u32::MAX; self.id_to_token.len()];
        for &(l, r) in merge_map.keys() {
            let Some((rank, _)) = self.merge_lookup(l, r) else {
                continue;
            };
            for t in [l, r] {
                if let Some(m) = min_op.get_mut(t as usize) {
                    *m = (*m).min(rank);
                }
            }
        }
        let mut a = CharAtoms::empty();
        let mut rule_ix: HashMap<AtomRule, u32> = HashMap::new();
        let mut bmp = vec![u32::MAX; 0x10000];
        for (t, cp, r) in candidates {
            // Classic BPE over the char's bytes alone, tracking the ranks applied.
            let mut w: Vec<u32> = r
                .iter()
                .map(|&x| self.byte_to_initial_token[x as usize])
                .collect();
            if w.contains(&INVALID_TOKEN) {
                continue;
            }
            let mut max_rank = 0u32;
            loop {
                let best = (0..w.len().saturating_sub(1))
                    .filter_map(|i| self.merge_lookup(w[i], w[i + 1]).map(|(rk, p)| (rk, i, p)))
                    .min();
                let Some((rk, i, p)) = best else { break };
                max_rank = max_rank.max(rk);
                w[i] = p;
                w.remove(i + 1);
            }
            if w != [t] || min_op[t as usize] <= max_rank {
                continue;
            }
            let mut rule = AtomRule::default();
            for k in 1..r.len() {
                for h in heads.get(&r[..k]).into_iter().flatten() {
                    rule.before[(h[h.len() - 1] >> 6) as usize] |= 1 << (h[h.len() - 1] & 63);
                    rule.heads.push(h.clone());
                }
                for tl in tails.get(&r[k..]).into_iter().flatten() {
                    rule.after[(tl[0] >> 6) as usize] |= 1 << (tl[0] & 63);
                    rule.tails.push(tl.clone());
                }
            }
            let pack = |f: &[u8]| {
                let mut x = [0u8; 8];
                x[..f.len()].copy_from_slice(f);
                u64::from_le_bytes(x)
            };
            let low = |l: usize| {
                if l == 8 {
                    u64::MAX
                } else {
                    (1u64 << (8 * l)) - 1
                }
            };
            let short = |f: &[u8]| f.len() <= CharAtoms::FRAG_MAX;
            rule.long_heads = rule.heads.iter().filter(|f| !short(f)).cloned().collect();
            rule.long_tails = rule.tails.iter().filter(|f| !short(f)).cloned().collect();
            rule.head_km = rule
                .heads
                .iter()
                .filter(|f| short(f))
                .map(|h| {
                    let sh = 64 - 8 * h.len() as u32;
                    (pack(h) << sh, low(h.len()) << sh)
                })
                .collect();
            rule.tail_km = rule
                .tails
                .iter()
                .filter(|f| short(f))
                .map(|t| (pack(t), low(t.len())))
                .collect();
            let ix = if rule.heads.is_empty() && rule.tails.is_empty() {
                0
            } else {
                rule.heads.sort();
                rule.heads.dedup();
                rule.tails.sort();
                rule.tails.dedup();
                *rule_ix.entry(rule.clone()).or_insert_with(|| {
                    a.rules.push(rule);
                    (a.rules.len() - 1) as u32
                })
            };
            let c = self.compact_id[t as usize];
            if cp < 0x10000 {
                // (An id or rule index too wide to pack: the char is simply not an
                // atom.)
                if c <= ATOM_ID_MASK && ix < (u32::MAX >> ATOM_ID_BITS) {
                    bmp[cp as usize] = ix << ATOM_ID_BITS | c;
                }
            } else {
                a.astral.insert(cp, (c, ix));
            }
        }
        // A block is good with at least 8 atoms whose fragment checks stay cheap:
        // on average at most 8 heads to compare where the byte before the char
        // (inside a word of the script, a continuation byte) passes the check's
        // prefilter. (DeepSeek's Hebrew letters carry ~160 such heads each: the
        // checks then cost far more than the byte merges they save.)
        for block in 0..1024usize {
            let (mut atoms, mut heads) = (0usize, 0usize);
            for &e in &bmp[block << 6..(block + 1) << 6] {
                if e == u32::MAX {
                    continue;
                }
                atoms += 1;
                let rule = &a.rules[(e >> ATOM_ID_BITS) as usize];
                if rule.before[2] != 0 {
                    heads += rule.heads.len();
                }
            }
            if atoms > 0 {
                a.some[block >> 6] |= 1 << (block & 63);
            }
            if atoms >= 8 && heads <= 8 * atoms {
                a.good[block >> 6] |= 1 << (block & 63);
            }
            if atoms >= 8 && heads <= 32 * atoms {
                a.fair[block >> 6] |= 1 << (block & 63);
            }
        }
        a.bmp = bmp.into_boxed_slice();
        // Consecutive complete chars inside any token (a pair is recorded only when
        // both chars are whole and adjacent in the token).
        let mut pairs: Vec<u64> = Vec::new();
        for tok in &self.id_to_token {
            let Some(r) = raw_of(tok) else { continue };
            let mut prev: Option<(u32, usize)> = None; // (char, end offset)
            let mut i = 0usize;
            while i < r.len() {
                let len = match r[i] {
                    0x00..=0x7F => 1,
                    0xC0..=0xDF => 2,
                    0xE0..=0xEF => 3,
                    0xF0..=0xF7 => 4,
                    _ => 0,
                };
                let ch = (len > 0 && i + len <= r.len())
                    .then(|| std::str::from_utf8(&r[i..i + len]).ok())
                    .flatten()
                    .and_then(|s| s.chars().next());
                match ch {
                    Some(c) => {
                        if let Some((p, end)) = prev
                            && end == i
                        {
                            pairs.push((p as u64) << 21 | c as u64);
                        }
                        prev = Some((c as u32, i + len));
                        i += len;
                    }
                    None => {
                        prev = None;
                        i += 1;
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        // ~32 filter bits per pair (a false-positive rate of ~3%).
        let bits = (pairs.len() * 32).next_power_of_two().max(64);
        let mut set = vec![0u64; bits / 64];
        for &k in &pairs {
            let bit = CharAtoms::bigram_bit(k, bits);
            set[bit >> 6] |= 1 << (bit & 63);
        }
        a.bigrams = set.into_boxed_slice();
        a
    }

    /// [`Self::long_vocab`]: vocab entries whose ByteLevel text maps back to more
    /// than 15 raw bytes forming valid UTF-8, keyed by [`LongCache::hash`].
    fn build_long_vocab(&self) -> LongVocab {
        let mut inv = [u16::MAX; 324];
        for (b, &c) in BYTE_TO_CHAR.iter().enumerate() {
            inv[c as usize] = b as u16;
        }
        let mut m = LongVocab::new();
        for (t, tok) in self.id_to_token.iter().enumerate() {
            let raw: Option<Vec<u8>> = tok
                .chars()
                .map(|ch| {
                    inv.get(ch as usize)
                        .copied()
                        .filter(|&v| v != u16::MAX)
                        .map(|v| v as u8)
                })
                .collect();
            let Some(raw) = raw else { continue };
            if raw.len() <= 15 || std::str::from_utf8(&raw).is_err() {
                continue;
            }
            m.entry(LongCache::hash(&raw))
                .or_default()
                .push((raw.into(), t as u32));
        }
        m
    }

    /// What a cold miss on the inline-packable pretoken `text` (packed as `key`)
    /// encodes to, exactly as [`Self::process_miss`] computes it: its whole-token
    /// fold under `ignore_merges`, else its BPE merge. The cache seed's values,
    /// so a seeded entry is bit-identical to the miss it replaces.
    #[inline(always)]
    fn miss_value_into(&self, text: &str, key: u128, hash: u64, out: &mut Vec<u32>) -> Result<()> {
        if self.ignore_merges
            && let Some(id) = self.fold.get(key, hash)
        {
            out.push(id);
            Ok(())
        } else {
            // (With `ignore_merges`, the fold holds every vocab entry of <= 15 bytes
            // — see `build_fold` — so a miss there already answers the whole-pretoken
            // lookup: not a token.)
            self.merge_all_raw_into(text, out)
        }
    }

    /// Bind this thread's pretoken cache (`TL_FUSED_CACHE`) to this model.
    ///
    /// A seeded cache is costly to rebuild — a copy of the seed (16 MiB for GLM)
    /// and a pass over the merge tables, ~1 ms — so the one this thread leaves is
    /// parked, up to [`PARKED_CACHES`] of them, and resumed when the thread comes
    /// back to its model: a thread alternating between models rebuilds each one's
    /// cache once, not at every switch. An unseeded cache (a parallel job's worker)
    /// is reset in place, as it is cheap to. A new binding is seeded outside
    /// parallel jobs ([`Self::pt_seed`]), then the merge tables warmed
    /// ([`Self::warm_merge_tables`]).
    fn bind_cache(&self, cache: &mut PretokenCache) {
        let resumed = TL_PARKED_CACHES.with(|p| {
            let mut parked = p.borrow_mut();
            let resumed = parked
                .iter()
                .position(|c| c.bpe_id == self.id)
                .map(|j| parked.remove(j));
            let is_resumed = resumed.is_some();
            let next = match resumed {
                Some(c) => Some(c),
                // Recycle the least recently used parked cache's allocation.
                None if cache.seeded && parked.len() >= PARKED_CACHES => parked.pop(),
                None if cache.seeded => Some(PretokenCache::new()),
                None => None,
            };
            if let Some(next) = next {
                let left = std::mem::replace(cache, next);
                if left.seeded {
                    parked.insert(0, left);
                }
            }
            is_resumed
        });
        if resumed {
            return;
        }
        let seed = self.pt_seed();
        cache.reset_for(self.id, seed);
        if seed.is_some() {
            self.warm_merge_tables();
        }
    }

    /// Read one word of every cache line of the tables a cold word's merge probes.
    ///
    /// Building the seed walks them, but copying it into the thread's cache (16
    /// MiB) that follows pushes them out, and the merges of the first unseen words
    /// would then wait on DRAM. A single pass over them (~10 MiB for GLM, about a
    /// millisecond, once per thread and model) leaves them in L3, the small and
    /// hottest tables last so they also stay in L2. GLM in tokbench's cold regime
    /// (a fresh tokenizer, ~1 MiB encoded): +3–5%.
    #[inline(never)]
    fn warm_merge_tables(&self) {
        fn lines<T: Copy>(t: &[T], acc: &mut u64) {
            let step = (64 / size_of::<T>()).max(1);
            let p = t.as_ptr() as *const u8;
            for i in (0..t.len()).step_by(step) {
                // SAFETY: `i < t.len()`, and every `T` here is at least a byte.
                *acc =
                    acc.wrapping_add(unsafe { p.add(i * size_of::<T>()).read_volatile() } as u64);
            }
        }
        let mut acc = 0u64;
        lines(&self.merge_adj.packed, &mut acc);
        lines(&self.merge_adj.buckets, &mut acc);
        match &self.code_grid {
            Some(g) => lines(g, &mut acc),
            None => lines(&self.merge_grid, &mut acc),
        }
        lines(&self.pair_initial, &mut acc);
        if let Some(f) = &self.pair_filter {
            lines(&f.bits, &mut acc);
        }
        std::hint::black_box(acc);
    }

    /// The model's [`PtSeed`], built on first use (`None` if it has no short
    /// vocab entry to seed) — for a thread binding its cache outside a parallel
    /// job. A job's workers keep unseeded caches: they share misses through
    /// [`SharedPretokenCache`], and a seeded table per worker would cost each one
    /// a copy and tens of MiB.
    fn pt_seed(&self) -> Option<&PtSeed> {
        if crate::fanout::in_parallel_task() {
            return None;
        }
        self.pt_seed
            .get_or_init(|| self.build_pt_seed().map(Arc::new))
            .as_deref()
    }

    /// Every vocab entry whose ByteLevel text maps back to 1..=15 raw bytes of
    /// valid UTF-8 (a scanner pretoken is always whole chars), valued by
    /// [`Self::miss_value_into`], in a table at most a third full.
    fn build_pt_seed(&self) -> Option<PtSeed> {
        let mut inv = [u16::MAX; 324];
        for (b, &c) in BYTE_TO_CHAR.iter().enumerate() {
            inv[c as usize] = b as u16;
        }
        let mut raws = Vec::new();
        for tok in &self.id_to_token {
            let mut raw = [0u8; 15];
            let mut n = 0usize;
            let mut ok = true;
            for ch in tok.chars() {
                let cp = ch as usize;
                if n == 15 || cp >= inv.len() || inv[cp] == u16::MAX {
                    ok = false;
                    break;
                }
                raw[n] = inv[cp] as u8;
                n += 1;
            }
            if ok && n > 0 && std::str::from_utf8(&raw[..n]).is_ok() {
                raws.push((raw, n));
            }
        }
        if raws.is_empty() {
            return None;
        }
        // At most a third full: the hot loop probes only an entry's home slot,
        // and a seed displaced from it costs the slow path on its first hit.
        let cap = (raws.len() * 3)
            .next_power_of_two()
            .max(1 << PRETOKEN_CACHE_MIN_BITS);
        let mut cache = PretokenCache {
            bpe_id: 0,
            mask: cap - 1,
            cap,
            len: 0,
            slots: HugeTable::new(cap, PtEntry::empty()),
            spill: Vec::new(),
            long: LongCache::new(),
            span_chunk: None,
            seeded: false,
        };
        let mut ids = Vec::new();
        for (raw, n) in &raws {
            let text = std::str::from_utf8(&raw[..*n]).expect("checked above");
            let key = PretokenCache::pack_key(text.as_bytes()).expect("1..=15 bytes");
            let hash = PretokenCache::hash(key);
            if let Err(vacancy) = cache.get_or_vacancy(key, hash, &mut ids) {
                ids.clear();
                self.miss_value_into(text, key, hash, &mut ids).ok()?;
                cache.insert_at(key, hash, vacancy, &ids);
            }
            ids.clear();
        }
        debug_assert_eq!(cache.cap, cap, "a seed never grows its table");
        Some(PtSeed {
            cap,
            len: cache.len,
            slots: cache.slots,
            spill: cache.spill,
        })
    }

    /// The [`FoldTable`] of an `ignore_merges` model: every vocab entry whose
    /// ByteLevel text maps back to 1..=15 raw bytes forming valid UTF-8 (a scanner
    /// pretoken is always whole chars).
    fn build_fold(&self) -> FoldTable {
        let mut inv = [u16::MAX; 324];
        for (b, &c) in BYTE_TO_CHAR.iter().enumerate() {
            inv[c as usize] = b as u16;
        }
        let mut entries = Vec::new();
        for (t, tok) in self.id_to_token.iter().enumerate() {
            let mut raw = [0u8; 15];
            let mut n = 0usize;
            let mut ok = true;
            for ch in tok.chars() {
                let cp = ch as usize;
                if n == 15 || cp >= inv.len() || inv[cp] == u16::MAX {
                    ok = false;
                    break;
                }
                raw[n] = inv[cp] as u8;
                n += 1;
            }
            if !ok || n == 0 || std::str::from_utf8(&raw[..n]).is_err() {
                continue;
            }
            if let Some(k) = PretokenCache::pack_key(&raw[..n]) {
                entries.push((k, t as u32));
            }
        }
        FoldTable::from_entries(&entries)
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

    /// Priority-queue BPE merge on already-encoded (ByteLevel) text.
    fn merge_all_encoded_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
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
                let s = ch.encode_utf8(&mut buf);
                let found = if ch.is_ascii() {
                    let id = self.single_char_token[ch as usize];
                    (id != INVALID_TOKEN).then_some(id)
                } else {
                    self.token_to_id.get(s).copied()
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

                for &byte in s.as_bytes() {
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

    /// Linear-scan BPE merge for short pretokens (`n <= SMALL_MERGE_MAX`
    /// initial symbols). Avoids the `BinaryHeap` entirely: a stack-resident
    /// doubly-linked list plus a per-position rank array, find-min by a short
    /// scan over stack `u32`s, merge (O(1) pointer update), then refresh only
    /// the two neighbor pairs. At these sizes this beats the heap's
    /// sift/stale-entry traffic and does zero heap allocation.
    ///
    /// Produces the identical token sequence as [`Self::run_merge_loop`]: both
    /// process the globally lowest-`(rank, pos)` active pair each step (the
    /// heap's `MergeEntry` key is `(rank << 32) | pos`; the scan's strict `<`
    /// keeps the leftmost/lowest-`pos` position on ties). Enforced by the
    /// `merge_small_matches_heap` differential test. `ids[..n]` are the
    /// per-byte initial token ids.
    /// Packed merge value (`rank << 32 | product`, compact ids) for a pair of
    /// *compact* ids, or `u64::MAX`. Hot pairs (both `< MERGE_GRID_DIM`) are one
    /// direct grid load; the rest one cache-line probe of `merge_adj`. The raw-byte
    /// merge engines work entirely in compact ids, so a lookup is a single load —
    /// not the `compact_id[a]`, `compact_id[b]` → table chain an external-id
    /// lookup needs on every merge's critical path.
    #[inline(always)]
    fn lookup_c(&self, ca: u32, cb: u32) -> u64 {
        #[cfg(feature = "diag")]
        DIAG_T.with(|d| {
            let mut a = d.get();
            if ca < MERGE_GRID_DIM && cb < MERGE_GRID_DIM {
                a[7] += 1
            } else {
                a[7] += 1 << 32
            }
            d.set(a);
        });
        if let Some(g) = &self.code_grid {
            if ca < CODE_GRID_DIM && cb < CODE_GRID_DIM {
                // SAFETY: `ca, cb < CODE_GRID_DIM`, so the index is `< CODE_GRID_DIM²`.
                let c =
                    unsafe { *g.get_unchecked(ca as usize * CODE_GRID_DIM as usize + cb as usize) };
                return if c == u32::MAX {
                    u64::MAX
                } else {
                    ((c as u64) << 32) | (c >> RANK_SUB_BITS) as u64
                };
            }
        } else if ca < MERGE_GRID_DIM && cb < MERGE_GRID_DIM {
            // SAFETY: `ca, cb < MERGE_GRID_DIM`, so the index is `< MERGE_GRID_DIM²`.
            return unsafe {
                *self
                    .merge_grid
                    .get_unchecked(ca as usize * MERGE_GRID_DIM as usize + cb as usize)
            };
        }
        if self.merge_adj.buckets.is_empty() {
            self.merge_adj.get_packed_compact(ca, cb)
        } else {
            self.merge_adj.get_packed(ca, cb)
        }
    }

    /// Rank + merged id for an adjacent pair, in *external* ids (the encoded-text
    /// engine's API): translate to compact, look up, map the product back.
    #[inline(always)]
    fn merge_lookup(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let ca = self.compact_id.get(a as usize).copied().unwrap_or(u32::MAX);
        let cb = self.compact_id.get(b as usize).copied().unwrap_or(u32::MAX);
        let v = self.lookup_c(ca, cb);
        if v == u64::MAX {
            None
        } else {
            Some(((v >> 32) as u32, self.unmap[(v & PV_ID_MASK) as usize]))
        }
    }

    /// Minimum of `keys[..len]` rounded up to 8 lanes (entries past the live
    /// prefix are `u32::MAX`). Fixed 8-wide strides let LLVM emit vector `umin`s:
    /// the selection is branch-free, where a scalar `if key < best` loop
    /// mispredicts on nearly every merge.
    #[inline(always)]
    fn min_key<const N: usize>(keys: &[u32; N], len: usize) -> u32 {
        let lim = (len + 7) & !7;
        let mut acc = [u32::MAX; 8];
        for c in keys[..lim].chunks_exact(8) {
            for j in 0..8 {
                acc[j] = acc[j].min(c[j]);
            }
        }
        let mut m = acc[0];
        for &a in &acc[1..] {
            m = m.min(a);
        }
        m
    }

    /// `N` is the scratch capacity (a multiple of 8, `n <= N <= SMALL_MERGE_MAX`):
    /// short words use a small instance so the per-word scratch set-up stays a
    /// few stores instead of ~2 KB of fills.
    /// `init(i)` gives the packed value of the initial pair at `i` (the byte-pair
    /// table for raw bytes; a pair-table lookup, possibly skipped, for symbols with
    /// pre-formed [`CharAtoms`]).
    #[inline(always)]
    fn merge_small_raw<const N: usize>(
        &self,
        ids: &mut [u32; N],
        n: usize,
        out: &mut Vec<u32>,
        init: impl Fn(usize) -> u64,
    ) {
        const { assert!(N.is_multiple_of(8) && N <= SMALL_MERGE_MAX) };
        // Linked list over the live symbols; `keys[i]` is the pair starting at live
        // position `i` packed as `rank << SMALL_KEY_BITS | i` (`u32::MAX` = no merge),
        // so one unsigned min yields the lowest rank with the leftmost tie-break.
        let mut next = [0u8; N];
        let mut prev = [0u8; N];
        let mut keys = [u32::MAX; N];
        let mut new_ids = [0u32; N];
        let pack = |rank: u32, i: usize| (rank << SMALL_KEY_BITS) | i as u32;
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = (i as u8).wrapping_sub(1); // prev[0] = 255 (>= n): sentinel
        }
        // Round-1 ranks via the dense byte-pair table: one direct-indexed load
        // per pair instead of a CSR neighbor scan.
        for i in 0..n - 1 {
            let v = init(i);
            if v != u64::MAX {
                keys[i] = pack((v >> 32) as u32, i);
                new_ids[i] = (v & PV_ID_MASK) as u32;
            }
        }
        // Each merge's critical path would otherwise be serial: reduce all keys →
        // lookup (an L2 load) → store the new keys → the next reduction waits on
        // those stores. Instead the affected keys are cleared first, the lookups
        // issued, and the minimum over every *other* key is reduced while the
        // loads are in flight; the next best is that minimum or one of the (at
        // most two) fresh keys.
        let mut best = Self::min_key(&keys, n - 1);
        while best != u32::MAX {
            let i = (best & SMALL_KEY_MASK) as usize;
            ids[i] = new_ids[i];
            let dead = next[i] as usize;
            let new_right = next[dead] as usize;
            next[i] = new_right as u8;
            let left = prev[i] as usize;
            keys[dead] = u32::MAX;
            keys[i] = u32::MAX;
            let vr = if new_right < n {
                prev[new_right] = i as u8;
                self.lookup_c(ids[i], ids[new_right])
            } else {
                u64::MAX
            };
            let vl = if left < n {
                keys[left] = u32::MAX;
                self.lookup_c(ids[left], ids[i])
            } else {
                u64::MAX
            };
            let others = Self::min_key(&keys, n - 1);
            let ki = if vr == u64::MAX {
                u32::MAX
            } else {
                pack((vr >> 32) as u32, i)
            };
            keys[i] = ki;
            new_ids[i] = (vr & PV_ID_MASK) as u32;
            let mut kl = u32::MAX;
            if left < n {
                kl = if vl == u64::MAX {
                    u32::MAX
                } else {
                    pack((vl >> 32) as u32, left)
                };
                keys[left] = kl;
                new_ids[left] = (vl & PV_ID_MASK) as u32;
            }
            best = others.min(ki).min(kl);
        }
        let mut i = 0usize;
        while i < n {
            // SAFETY: `ids` hold compact ids, all `< unmap.len()`.
            out.push(unsafe { *self.unmap.get_unchecked(ids[i] as usize) });
            i = next[i] as usize;
        }
    }

    /// [`Self::merge_small_raw`] with the pair lookup of the merges it makes
    /// (`lookup`: compact ids → packed value, as [`Self::lookup_c`]).
    #[inline(always)]
    fn merge_small_raw_by<const N: usize>(
        &self,
        ids: &mut [u32; N],
        n: usize,
        out: &mut Vec<u32>,
        init: impl Fn(usize) -> u64,
        lookup: impl Fn(u32, u32) -> u64,
    ) {
        const { assert!(N.is_multiple_of(8) && N <= SMALL_MERGE_MAX) };
        // Linked list over the live symbols; `keys[i]` is the pair starting at live
        // position `i` packed as `rank << SMALL_KEY_BITS | i` (`u32::MAX` = no merge),
        // so one unsigned min yields the lowest rank with the leftmost tie-break.
        let mut next = [0u8; N];
        let mut prev = [0u8; N];
        let mut keys = [u32::MAX; N];
        let mut new_ids = [0u32; N];
        let pack = |rank: u32, i: usize| (rank << SMALL_KEY_BITS) | i as u32;
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = (i as u8).wrapping_sub(1); // prev[0] = 255 (>= n): sentinel
        }
        // Round-1 ranks via the dense byte-pair table: one direct-indexed load
        // per pair instead of a CSR neighbor scan.
        for i in 0..n - 1 {
            let v = init(i);
            if v != u64::MAX {
                keys[i] = pack((v >> 32) as u32, i);
                new_ids[i] = (v & PV_ID_MASK) as u32;
            }
        }
        // Each merge's critical path would otherwise be serial: reduce all keys →
        // lookup (an L2 load) → store the new keys → the next reduction waits on
        // those stores. Instead the affected keys are cleared first, the lookups
        // issued, and the minimum over every *other* key is reduced while the
        // loads are in flight; the next best is that minimum or one of the (at
        // most two) fresh keys.
        let mut best = Self::min_key(&keys, n - 1);
        while best != u32::MAX {
            let i = (best & SMALL_KEY_MASK) as usize;
            ids[i] = new_ids[i];
            let dead = next[i] as usize;
            let new_right = next[dead] as usize;
            next[i] = new_right as u8;
            let left = prev[i] as usize;
            keys[dead] = u32::MAX;
            keys[i] = u32::MAX;
            let vr = if new_right < n {
                prev[new_right] = i as u8;
                lookup(ids[i], ids[new_right])
            } else {
                u64::MAX
            };
            let vl = if left < n {
                keys[left] = u32::MAX;
                lookup(ids[left], ids[i])
            } else {
                u64::MAX
            };
            let others = Self::min_key(&keys, n - 1);
            let ki = if vr == u64::MAX {
                u32::MAX
            } else {
                pack((vr >> 32) as u32, i)
            };
            keys[i] = ki;
            new_ids[i] = (vr & PV_ID_MASK) as u32;
            let mut kl = u32::MAX;
            if left < n {
                kl = if vl == u64::MAX {
                    u32::MAX
                } else {
                    pack((vl >> 32) as u32, left)
                };
                keys[left] = kl;
                new_ids[left] = (vl & PV_ID_MASK) as u32;
            }
            best = others.min(ki).min(kl);
        }
        let mut i = 0usize;
        while i < n {
            // SAFETY: `ids` hold compact ids, all `< unmap.len()`.
            out.push(unsafe { *self.unmap.get_unchecked(ids[i] as usize) });
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
        // Atoms pay off per script and vocab (`CharAtoms::pays_off`): CJK and
        // Hangul, scripts whose chars are atoms with cheap fragment checks (per
        // model: DeepSeek's Hebrew letters are costly, GLM's not), and long runs
        // (Thai), cut into short segments; Ethiopic, with no atoms, merges as bytes.
        if !bytes.is_ascii()
            && !self.atoms.bmp.is_empty()
            && self.atoms.pays_off(bytes)
            && self.merge_with_atoms(raw_input, out)?
        {
            return Ok(());
        }
        if bytes.len() <= 16 {
            return self.merge_small_into::<16>(bytes, out);
        }
        if bytes.len() <= SMALL_MERGE_MAX {
            return self.merge_small_into::<SMALL_MERGE_MAX>(bytes, out);
        }

        self.merge_all_raw_heap_into(raw_input, out)
    }

    /// Merge `bytes` (not pure ASCII) starting from its pre-formable [`CharAtoms`];
    /// `Ok(false)` (nothing written) when none applies, so the caller merges the
    /// bytes as usual.
    ///
    /// The symbols are cut into segments wherever two adjacent atoms form a pair
    /// no vocab token holds ([`CharAtoms::joinable`]): a merge across the cut
    /// would produce such a token, so none happens, and each segment merges on
    /// its own. CJK runs are long, unique pretokens but split into segments of ~2
    /// chars, most of them a lone atom or a single pair — so this skips both the
    /// probes of pairs that straddle a cut and the whole-run merge scratch.
    #[inline(never)]
    fn merge_with_atoms(&self, text: &str, out: &mut Vec<u32>) -> Result<bool> {
        let bytes = text.as_bytes();
        let mut seg = [0u32; SMALL_MERGE_MAX];
        // `n` symbols in the current segment, which starts at byte `seg_start`;
        // `overflow`: it outgrew `seg` (it is then merged from its bytes).
        let (mut n, mut i, mut any, mut overflow) = (0usize, 0usize, false, false);
        let mut seg_start = 0usize;
        // Codepoint of the segment's last symbol when that is an atom; whether
        // all its symbols are.
        let mut prev_cp = u32::MAX;
        let mut pure = true;
        while i < bytes.len() {
            let b0 = bytes[i];
            if b0 < 0x80 {
                let c = self.byte_compact[b0 as usize];
                if c == INVALID_TOKEN {
                    return Err(format!("byte 0x{b0:02x} has no token in vocabulary"));
                }
                if n < SMALL_MERGE_MAX {
                    seg[n] = c;
                    n += 1;
                } else {
                    overflow = true;
                }
                prev_cp = u32::MAX;
                pure = false;
                i += 1;
                continue;
            }
            let len = match b0 {
                0xC0..=0xDF => 2,
                0xE0..=0xEF => 3,
                _ => 4,
            };
            let at = |k: usize| bytes[i + k] as u32 & 0x3F;
            let cp = match len {
                2 => ((b0 as u32 & 0x1F) << 6) | at(1),
                3 => ((b0 as u32 & 0x0F) << 12) | (at(1) << 6) | at(2),
                _ => ((b0 as u32 & 0x07) << 18) | (at(1) << 12) | (at(2) << 6) | at(3),
            };
            if let Some(c) = self.atoms.atom_at(bytes, i, len, cp) {
                if prev_cp != u32::MAX {
                    if !self.atoms.joinable(prev_cp, cp) {
                        self.merge_segment(&mut seg, n, pure, overflow, &bytes[seg_start..i], out)?;
                        (n, pure, overflow, seg_start) = (0, true, false, i);
                    } else if n > 0 && n <= SMALL_MERGE_MAX {
                        // The pair's probe comes when the segment is merged.
                        self.atoms.merges.prefetch(seg[n - 1], c);
                    }
                }
                if n < SMALL_MERGE_MAX {
                    seg[n] = c;
                    n += 1;
                } else {
                    overflow = true;
                }
                any = true;
                prev_cp = cp;
            } else {
                for &x in &bytes[i..i + len] {
                    let c = self.byte_compact[x as usize];
                    if c == INVALID_TOKEN {
                        return Err(format!("byte 0x{x:02x} has no token in vocabulary"));
                    }
                    if n < SMALL_MERGE_MAX {
                        seg[n] = c;
                        n += 1;
                    } else {
                        overflow = true;
                    }
                }
                prev_cp = u32::MAX;
                pure = false;
            }
            i += len;
        }
        // No atom: no cut either, so nothing is written yet.
        if !any {
            return Ok(false);
        }
        self.merge_segment(&mut seg, n, pure, overflow, &bytes[seg_start..], out)?;
        Ok(true)
    }

    /// One segment of [`Self::merge_with_atoms`]: its `n` symbols in `seg`, or,
    /// when it outgrew `seg`, merged from its bytes `text` (a segment merges the
    /// same on its own as within its pretoken). `pure`: every symbol is an atom,
    /// so every merge joins whole non-ASCII chars — the small table has them all.
    #[inline(always)]
    fn merge_segment(
        &self,
        seg: &mut [u32; SMALL_MERGE_MAX],
        n: usize,
        pure: bool,
        overflow: bool,
        text: &[u8],
        out: &mut Vec<u32>,
    ) -> Result<()> {
        if overflow {
            // SAFETY: segments are cut at char boundaries of a `str`.
            return self
                .merge_all_raw_heap_into(unsafe { std::str::from_utf8_unchecked(text) }, out);
        }
        if n == 1 {
            // SAFETY: compact ids are `< unmap.len()`.
            out.push(unsafe { *self.unmap.get_unchecked(seg[0] as usize) });
        } else if let Some(g) = &self.code_grid
            && self.merge_adj.buckets.is_empty()
            && n <= 16
        {
            // Rank codes: the code engine (see `merge_small_codes`), on the small
            // all-atom table when every symbol is an atom. (Longer segments keep
            // the engine below, whose vector minimum beats a scalar scan there.)
            if pure {
                let m = &self.atoms.merges;
                self.merge_segment_codes(
                    seg,
                    n,
                    out,
                    |a, b| m.get_code(a, b),
                    |a, b| m.prefetch(a, b),
                );
            } else {
                self.merge_segment_codes(
                    seg,
                    n,
                    out,
                    |a, b| self.code_of(g, a, b),
                    |a, b| self.prefetch_code(g, a, b),
                );
            }
        } else if pure {
            self.merge_segment_by(seg, n, out, |a, b| self.atoms.merges.lookup(a, b));
        } else {
            self.merge_segment_by(seg, n, out, |a, b| self.lookup_c(a, b));
        }
        Ok(())
    }

    /// A segment's merge (`n <= 16` symbols) by rank codes (`code`: a pair's
    /// [`rank_code`] or `u32::MAX`; `prefetch`: request its line), in
    /// `merge_small_codes`' loop.
    #[inline(always)]
    fn merge_segment_codes(
        &self,
        seg: &mut [u32; SMALL_MERGE_MAX],
        n: usize,
        out: &mut Vec<u32>,
        code: impl Fn(u32, u32) -> u32 + Copy,
        prefetch: impl Fn(u32, u32) + Copy,
    ) {
        let mut ids = [0u32; 16];
        ids[..n].copy_from_slice(&seg[..n]);
        self.merge_ids_codes::<16>(&mut ids, n, out, code, prefetch);
    }

    /// [`Self::merge_small_codes`] from symbols `ids[..n]` (`n >= 2`) of any kind,
    /// their pairs' codes read with `code`.
    #[inline(always)]
    fn merge_ids_codes<const N: usize>(
        &self,
        ids: &mut [u32; N],
        n: usize,
        out: &mut Vec<u32>,
        code: impl Fn(u32, u32) -> u32,
        prefetch: impl Fn(u32, u32),
    ) {
        let mut next = [0u8; N];
        let mut prev = [0u8; N];
        let mut codes = [u32::MAX; N];
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = (i as u8).wrapping_sub(1); // prev[0] = 255 (>= n): sentinel
        }
        for i in 0..n - 1 {
            codes[i] = code(ids[i], ids[i + 1]);
        }
        loop {
            let (mut best, mut i) = (u32::MAX, 0usize);
            for (j, &c) in codes[..n - 1].iter().enumerate() {
                if c < best {
                    best = c;
                    i = j;
                }
            }
            if best == u32::MAX {
                break;
            }
            let product = best >> RANK_SUB_BITS;
            let dead = next[i] as usize;
            let new_right = next[dead] as usize;
            let left = prev[i] as usize;
            if new_right < n {
                prefetch(product, ids[new_right]);
            }
            if left < n {
                prefetch(ids[left], product);
            }
            ids[i] = product;
            next[i] = new_right as u8;
            codes[dead] = u32::MAX;
            codes[i] = if new_right < n {
                prev[new_right] = i as u8;
                code(product, ids[new_right])
            } else {
                u32::MAX
            };
            if left < n {
                codes[left] = code(ids[left], product);
            }
        }
        let mut i = 0usize;
        if let Some(k) = self.unmap_offset {
            while i < n {
                out.push(ids[i] + k);
                i = next[i] as usize;
            }
            return;
        }
        while i < n {
            // SAFETY: `ids` hold compact ids, all `< unmap.len()`.
            out.push(unsafe { *self.unmap.get_unchecked(ids[i] as usize) });
            i = next[i] as usize;
        }
    }

    #[inline(always)]
    fn merge_segment_by(
        &self,
        seg: &mut [u32; SMALL_MERGE_MAX],
        n: usize,
        out: &mut Vec<u32>,
        lookup: impl Fn(u32, u32) -> u64 + Copy,
    ) {
        // SAFETY (throughout): compact ids are `< unmap.len()`.
        if n == 2 {
            let v = lookup(seg[0], seg[1]);
            if v == u64::MAX {
                out.push(unsafe { *self.unmap.get_unchecked(seg[0] as usize) });
                out.push(unsafe { *self.unmap.get_unchecked(seg[1] as usize) });
            } else {
                out.push(unsafe { *self.unmap.get_unchecked((v & PV_ID_MASK) as usize) });
            }
        } else if n <= 16 {
            let mut small = [0u32; 16];
            small[..n].copy_from_slice(&seg[..n]);
            let snap = small;
            self.merge_small_raw_by::<16>(
                &mut small,
                n,
                out,
                |i| lookup(snap[i], snap[i + 1]),
                lookup,
            );
        } else {
            let snap = *seg;
            self.merge_small_raw_by::<SMALL_MERGE_MAX>(
                seg,
                n,
                out,
                |i| lookup(snap[i], snap[i + 1]),
                lookup,
            );
        }
    }

    /// [`Self::merge_small_raw`] with scratch capacity `N >= bytes.len()`.
    #[inline(always)]
    fn merge_small_into<const N: usize>(&self, bytes: &[u8], out: &mut Vec<u32>) -> Result<()> {
        let n = bytes.len();
        let mut ids = [0u32; N];
        for (i, &byte) in bytes.iter().enumerate() {
            let c = self.byte_compact[byte as usize];
            if c == INVALID_TOKEN {
                return Err(format!("byte 0x{byte:02x} has no token in vocabulary"));
            }
            ids[i] = c;
        }
        if n == 1 {
            out.push(self.byte_to_initial_token[bytes[0] as usize]);
        } else if let Some(g) = &self.code_grid
            && self.merge_adj.buckets.is_empty()
        {
            self.merge_small_codes::<N>(g, &mut ids, n, bytes, out);
        } else {
            // SAFETY: the index is `< 256 * 256 == pair_initial.len()`.
            let init = |i: usize| unsafe {
                *self
                    .pair_initial
                    .get_unchecked(bytes[i] as usize * 256 + bytes[i + 1] as usize)
            };
            self.merge_small_raw::<N>(&mut ids, n, out, init);
        }
        Ok(())
    }

    /// The [`rank_code`] of a pair of compact ids: the code grid, else the pair
    /// table (valid codes only: `code_grid` set, packed `merge_adj`).
    #[inline(always)]
    fn code_of(&self, g: &HugeTable<u32>, ca: u32, cb: u32) -> u32 {
        if ca < CODE_GRID_DIM && cb < CODE_GRID_DIM {
            // SAFETY: `ca, cb < CODE_GRID_DIM`, so the index is `< CODE_GRID_DIM²`.
            unsafe { *g.get_unchecked(ca as usize * CODE_GRID_DIM as usize + cb as usize) }
        } else {
            if let Some(f) = &self.pair_filter
                && (ca | cb) >> PACKED_ID_BITS == 0
                && !f.may_merge(MergeAdjacency::packed_key(ca, cb))
            {
                return u32::MAX;
            }
            self.merge_adj.get_code(ca, cb)
        }
    }

    /// Prefetch the line [`Self::code_of`] reads first.
    #[inline(always)]
    fn prefetch_code(&self, g: &HugeTable<u32>, ca: u32, cb: u32) {
        if ca < CODE_GRID_DIM && cb < CODE_GRID_DIM {
            // SAFETY: in bounds as in `code_of`; a prefetch reads nothing.
            prefetch_read(unsafe {
                g.as_ptr()
                    .add(ca as usize * CODE_GRID_DIM as usize + cb as usize)
            } as *const u8);
        } else if self.pair_filter.as_ref().is_none_or(|f| {
            (ca | cb) >> PACKED_ID_BITS != 0 || f.may_merge(MergeAdjacency::packed_key(ca, cb))
        }) {
            self.merge_adj.prefetch(ca, cb);
        }
    }

    /// [`Self::merge_small_raw`] for a vocab with valid [`rank_code`]s: a pair's
    /// code alone orders the merges (it *is* the rank) and names the product
    /// (`code >> RANK_SUB_BITS`), so the scratch is one `u32` per pair, the next
    /// merge a leftmost-minimum scan (equal codes are equal pairs: leftmost first,
    /// as the rank order requires), and both pairs a merge refreshes are known —
    /// and prefetched — before the list surgery.
    #[inline(always)]
    fn merge_small_codes<const N: usize>(
        &self,
        g: &HugeTable<u32>,
        ids: &mut [u32; N],
        n: usize,
        bytes: &[u8],
        out: &mut Vec<u32>,
    ) {
        let mut next = [0u8; N];
        let mut prev = [0u8; N];
        let mut codes = [u32::MAX; N];
        for i in 0..n {
            next[i] = (i + 1) as u8;
            prev[i] = (i as u8).wrapping_sub(1); // prev[0] = 255 (>= n): sentinel
        }
        for i in 0..n - 1 {
            // SAFETY: the index is `< 256 * 256 == pair_initial.len()`.
            let v = unsafe {
                *self
                    .pair_initial
                    .get_unchecked(bytes[i] as usize * 256 + bytes[i + 1] as usize)
            };
            codes[i] = if v == u64::MAX {
                u32::MAX
            } else {
                (v >> 32) as u32
            };
        }
        loop {
            let (mut best, mut i) = (u32::MAX, 0usize);
            for (j, &c) in codes[..n - 1].iter().enumerate() {
                if c < best {
                    best = c;
                    i = j;
                }
            }
            if best == u32::MAX {
                break;
            }
            let product = best >> RANK_SUB_BITS;
            let dead = next[i] as usize;
            let new_right = next[dead] as usize;
            let left = prev[i] as usize;
            if new_right < n {
                self.prefetch_code(g, product, ids[new_right]);
            }
            if left < n {
                self.prefetch_code(g, ids[left], product);
            }
            ids[i] = product;
            next[i] = new_right as u8;
            codes[dead] = u32::MAX;
            codes[i] = if new_right < n {
                prev[new_right] = i as u8;
                self.code_of(g, product, ids[new_right])
            } else {
                u32::MAX
            };
            if left < n {
                codes[left] = self.code_of(g, ids[left], product);
            }
        }
        let mut i = 0usize;
        if let Some(k) = self.unmap_offset {
            while i < n {
                out.push(ids[i] + k);
                i = next[i] as usize;
            }
            return;
        }
        while i < n {
            // SAFETY: `ids` hold compact ids, all `< unmap.len()`.
            out.push(unsafe { *self.unmap.get_unchecked(ids[i] as usize) });
            i = next[i] as usize;
        }
    }

    /// Priority-queue BPE merge on raw (pre-ByteLevel) bytes for long pretokens,
    /// in compact ids (see [`Self::lookup_c`]).
    fn merge_all_raw_heap_into(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();
            scratch.heap_buf.clear();

            let bytes = raw_input.as_bytes();
            let n = bytes.len();
            let mut prev_c = 0u32;
            for (i, &byte) in bytes.iter().enumerate() {
                let c = self.byte_compact[byte as usize];
                if c == INVALID_TOKEN {
                    return Err(format!("byte 0x{byte:02x} has no token in vocabulary"));
                }
                scratch.symbols.push(MergeSymbol {
                    c,
                    prev: if i == 0 { -1 } else { (i - 1) as i32 },
                    next: if i == n - 1 { -1 } else { (i + 1) as i32 },
                });
                if i > 0 {
                    let v = self.pair_initial[bytes[i - 1] as usize * 256 + byte as usize];
                    if v != u64::MAX {
                        scratch.heap_buf.push(Reverse(MergeEntry::new(
                            (v >> 32) as u32,
                            (i - 1) as u32,
                            prev_c,
                            c,
                        )));
                    }
                }
                prev_c = c;
            }

            if n == 1 {
                out.push(self.unmap[scratch.symbols[0].c as usize]);
                return Ok(());
            }

            // Bulk heapify.
            let mut tmp = std::mem::take(&mut scratch.heap_buf);
            scratch.heap.extend(tmp.drain(..));
            scratch.heap_buf = tmp;

            self.run_merge_loop_c(&mut scratch, out);
            Ok(())
        })
    }

    /// [`Self::run_merge_loop`] over compact ids; emits external ids.
    fn run_merge_loop_c(&self, scratch: &mut MergeScratch, out: &mut Vec<u32>) {
        let symbols = &mut scratch.symbols;
        let heap = &mut scratch.heap;
        while let Some(Reverse(entry)) = heap.pop() {
            let pos = entry.pos() as usize;
            let sym = symbols[pos];
            let (left_c, right_c) = (entry.left_c(), entry.right_c());
            if sym.c != left_c || sym.next < 0 {
                continue;
            }
            let next_idx = sym.next as usize;
            let next_sym = symbols[next_idx];
            if next_sym.c != right_c {
                continue;
            }
            let v = self.lookup_c(left_c, right_c);
            if v == u64::MAX {
                continue;
            }
            let new_c = (v & PV_ID_MASK) as u32;
            symbols[pos].c = new_c;
            symbols[pos].next = next_sym.next;
            if next_sym.next >= 0 {
                symbols[next_sym.next as usize].prev = pos as i32;
            }
            symbols[next_idx].c = INVALID_TOKEN;
            if sym.prev >= 0 {
                let pc = symbols[sym.prev as usize].c;
                let v = self.lookup_c(pc, new_c);
                if v != u64::MAX {
                    heap.push(Reverse(MergeEntry::new(
                        (v >> 32) as u32,
                        sym.prev as u32,
                        pc,
                        new_c,
                    )));
                }
            }
            let nn = symbols[pos].next;
            if nn >= 0 {
                let nc = symbols[nn as usize].c;
                let v = self.lookup_c(new_c, nc);
                if v != u64::MAX {
                    heap.push(Reverse(MergeEntry::new(
                        (v >> 32) as u32,
                        pos as u32,
                        new_c,
                        nc,
                    )));
                }
            }
        }
        let mut i: i32 = 0;
        while i >= 0 {
            let sym = symbols[i as usize];
            out.push(self.unmap[sym.c as usize]);
            i = sym.next;
        }
    }

    /// Test oracle: the external-id heap merge (the pre-compact engine), for the
    /// `merge_small_matches_heap` differential test.
    #[cfg(test)]
    fn merge_all_raw_ext_ref(&self, raw_input: &str, out: &mut Vec<u32>) -> Result<()> {
        TL_MERGE_SCRATCH.with(|s| {
            let mut scratch = s.borrow_mut();
            scratch.symbols.clear();
            scratch.heap.clear();
            let bytes = raw_input.as_bytes();
            let n = bytes.len();
            for (i, &byte) in bytes.iter().enumerate() {
                scratch.symbols.push(MergeSymbol {
                    c: self.byte_to_initial_token[byte as usize],
                    prev: if i == 0 { -1 } else { (i - 1) as i32 },
                    next: if i == n - 1 { -1 } else { (i + 1) as i32 },
                });
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

    /// Seed the priority queue with all initial adjacent pairs.
    #[inline(always)]
    fn init_merge_heap(&self, scratch: &mut MergeScratch, n: usize) {
        let symbols = &scratch.symbols;
        scratch.heap.extend((0..n - 1).filter_map(|i| {
            let left = symbols[i].c;
            let right = symbols[i + 1].c;
            self.merge_lookup(left, right)
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
            let new_id = match self.merge_lookup(left_c, right_c) {
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
                if let Some((rank, _)) = self.merge_lookup(prev_c, new_id) {
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
                if let Some((rank, _)) = self.merge_lookup(new_id, next_c) {
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
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                // Bound before the lookup: a resumed cache may hold it.
                self.bind_cache(&mut c);
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
                self.bind_cache(&mut c);
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

    /// Cache miss for an inline-packable span: BPE-merge it, then insert at the
    /// vacancy the failed probe ended on. Inlined into the out-of-line slow path
    /// ([`Self::process_pending_slow`]), so the per-span hot loop stays small; not
    /// `#[cold]`: on cold text misses are frequent, and a cold-section miss path
    /// pays instruction-cache misses on each one.
    #[inline(always)]
    fn process_miss(
        &self,
        cache: &mut PretokenCache,
        text: &str,
        key: u128,
        hash: u64,
        vacancy: usize,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        #[cfg(feature = "diag")]
        DIAG.with(|d| {
            let mut a = d.get();
            a[4] += 1;
            d.set(a);
        });
        let start = out.len();
        #[cfg(feature = "diag")]
        let t0 = std::time::Instant::now();
        // Keep in step with `miss_value_into` (the cache seed's values). A seeded
        // cache already holds every fold entry, so its misses skip the probe.
        if self.ignore_merges
            && !cache.seeded
            && let Some(id) = self.fold.get(key, hash)
        {
            out.push(id);
        } else if crate::fanout::in_parallel_task() {
            // Only within a multi-threaded job does another thread's merge help;
            // for a lone encode the shared table merely duplicates this thread's
            // cache, and probing it costs a likely-DRAM load per miss.
            let shared = self.shared_pt.get_or_init(SharedPretokenCache::new);
            if !shared.get(key, hash, out) {
                self.merge_all_raw_into(text, out)?;
                shared.insert(key, hash, &out[start..]);
            }
        } else {
            // (With `ignore_merges`, the fold holds every vocab entry of <= 15 bytes
            // — see `build_fold` — so a miss there already answers the whole-pretoken
            // lookup: not a token.)
            self.merge_all_raw_into(text, out)?;
        }
        #[cfg(feature = "diag")]
        let t1 = std::time::Instant::now();
        cache.insert_at(key, hash, vacancy, &out[start..]);
        #[cfg(feature = "diag")]
        {
            let t2 = std::time::Instant::now();
            DIAG_T.with(|d| {
                let mut a = d.get();
                a[0] += (t1 - t0).as_nanos() as u64;
                a[1] += (t2 - t1).as_nanos() as u64;
                a[2] += text.len() as u64;
                a[3] += (out.len() - start) as u64;
                if out.len() - start == 1 {
                    a[4] += 1;
                }
                d.set(a);
            });
        }
        Ok(())
    }

    /// A span too long to pack inline (> 15 bytes): the exact long-pretoken cache,
    /// merging and inserting on a miss. Out of line (≈2% of spans).
    #[inline(never)]
    fn process_long_span(
        &self,
        cache: &mut PretokenCache,
        text: &str,
        h: u64,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        // `h == 0`: a span not worth caching (see [`long_uncached`]).
        if h != 0 && cache.long.get(text.as_bytes(), h, out) {
            #[cfg(feature = "diag")]
            DIAG.with(|d| {
                let mut a = d.get();
                a[3] += 1;
                d.set(a);
            });
            return Ok(());
        }
        #[cfg(feature = "diag")]
        DIAG.with(|d| {
            let mut a = d.get();
            a[4] += 1;
            d.set(a);
        });
        let start = out.len();
        if self.ignore_merges {
            // Exhaustive for > 15 bytes: a miss here answers the whole-pretoken
            // lookup (not a token) without re-encoding the text.
            let vh = if h == 0 {
                LongCache::hash(text.as_bytes())
            } else {
                h
            };
            match self
                .long_vocab
                .get(&vh)
                .and_then(|v| v.iter().find(|(r, _)| &r[..] == text.as_bytes()))
            {
                Some(&(_, id)) => out.push(id),
                None => self.merge_all_raw_into(text, out)?,
            }
        } else {
            self.merge_all_raw_into(text, out)?;
        }
        if h != 0 {
            cache.long.insert(text.as_bytes(), h, &out[start..]);
        }
        Ok(())
    }

    /// Test/bench-only: merge each word with [`CharAtoms`] and bytes-only, returning
    /// how many differ, how many used atoms, and the first differing word.
    #[doc(hidden)]
    pub fn __check_atoms(&self, words: &[&str]) -> (usize, usize, Option<String>) {
        let (mut diff, mut used, mut first) = (0usize, 0usize, None);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for w in words {
            a.clear();
            b.clear();
            let bytes = w.as_bytes();
            let ok_a = if !bytes.is_ascii() && self.merge_with_atoms(w, &mut a).unwrap_or(false) {
                used += 1;
                true
            } else {
                false
            };
            if !ok_a {
                continue;
            }
            let _ = if bytes.len() <= SMALL_MERGE_MAX {
                self.merge_small_into::<SMALL_MERGE_MAX>(bytes, &mut b)
            } else {
                self.merge_all_raw_heap_into(w, &mut b)
            };
            if a != b {
                diff += 1;
                first.get_or_insert_with(|| w.to_string());
            }
        }
        (diff, used, first)
    }

    /// Benchmark-only: BPE-merge each word (raw bytes) with no caching, `reps`
    /// times; returns the total token count of one repetition.
    #[doc(hidden)]
    pub fn __bench_merge_words(&self, words: &[&str], reps: usize) -> usize {
        let mut out = Vec::with_capacity(64);
        let mut total = 0usize;
        for r in 0..reps {
            for w in words {
                out.clear();
                let _ = self.merge_all_raw_into(w, &mut out);
                if r == 0 {
                    total += out.len();
                }
                std::hint::black_box(&out);
            }
        }
        total
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
                self.bind_cache(&mut cache);
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

    /// Fused scan+BPE of one segment under a single thread-local cache borrow.
    ///
    /// The scanner's spans are taken in chunks of [`SPAN_CHUNK`], in two tight
    /// loops each: [`SpanChunk::harvest`] packs every span's cache key and hash,
    /// then [`Self::probe_emit`] probes them in order, prefetching each line a
    /// fixed number of spans ahead — hiding the pretoken cache's memory latency,
    /// which dominates on long-tail text. Two small loops keep their
    /// state in registers where one fused loop (scanner state, prefetch ring,
    /// table and output cursors together) spilled it on every span. When one bulk
    /// pass covers the segment its start bitmap is walked here; otherwise spans
    /// arrive through the scanner's per-span callback.
    pub fn tokenize_scanned_segment(
        &self,
        kind: crate::pre_tokenizers::scan::ScanKind,
        seg: &str,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let bpe_id = self.id;
        TL_FUSED_CACHE.with(|c| {
            let mut guard = c.borrow_mut();
            let cache: &mut PretokenCache = &mut guard;
            if cache.bpe_id != bpe_id {
                self.bind_cache(cache);
            }
            let sb = seg.as_bytes();
            // The reused buffer: its entries past `n` are stale spans, which the
            // look-ahead reads only to prefetch (or peek at) some line of the table.
            let mut chunk = cache
                .span_chunk
                .take()
                .unwrap_or_else(|| Box::new(SpanChunk::new()));
            chunk.n = 0;
            let r = (|| -> Result<()> {
                // Whole-text bulk scan when one covers the segment: walk its start
                // bitmap here. Otherwise (o200k with non-ASCII, other grammars) the
                // per-span callback of `scan_fast`. Both are byte-exact with `scan_core`.
                if let Some(bs) = crate::pre_tokenizers::scan_simd::bulk_starts(kind, seg) {
                    let (words, runs) = bs.starts_and_runs();
                    // The start bits in order, as byte offsets (`$byte` maps a bit
                    // offset), each span harvested and every full chunk encoded —
                    // one loop per mapping, written out so it stays in registers.
                    macro_rules! walk {
                        ($byte:expr) => {{
                            let mut prev = usize::MAX;
                            for (wk, &word) in words.iter().enumerate() {
                                let mut bits = word;
                                while bits != 0 {
                                    let b = $byte(wk * 64 + bits.trailing_zeros() as usize);
                                    bits &= bits - 1;
                                    if prev != usize::MAX {
                                        chunk.harvest(cache, sb, prev, b);
                                        if chunk.n == SPAN_CHUNK {
                                            self.probe_emit(cache, seg, &chunk, out)?;
                                            chunk.n = 0;
                                        }
                                    }
                                    prev = b;
                                }
                            }
                            // The last span runs to the end of the text.
                            if prev != usize::MAX {
                                chunk.harvest(cache, sb, prev, seg.len());
                            }
                        }};
                    }
                    if runs.is_empty() {
                        walk!(|c: usize| c);
                    } else {
                        let mut map = crate::pre_tokenizers::scan_simd::ByteMap::new(runs);
                        walk!(|c: usize| map.byte(c));
                    }
                    return self.probe_emit(cache, seg, &chunk, out);
                }
                crate::pre_tokenizers::scan_simd::scan_fast(kind, seg, |start, end| {
                    if start < end {
                        chunk.harvest(cache, sb, start, end);
                        if chunk.n == SPAN_CHUNK {
                            self.probe_emit(cache, seg, &chunk, out)?;
                            chunk.n = 0;
                        }
                    }
                    Ok(())
                })?;
                self.probe_emit(cache, seg, &chunk, out)
            })();
            cache.span_chunk = Some(chunk);
            r
        })
    }

    /// Encode a chunk of harvested spans, in order: the inline cache hit (~99% of
    /// spans on warm text) branch-free right here — the entry's inline ids are
    /// stored unconditionally and the output advances only on a hit — everything
    /// else (a displaced entry, a miss, a long span) out of line. Out of line
    /// itself, so its loop gets the registers to itself.
    #[inline(never)]
    fn probe_emit(
        &self,
        cache: &mut PretokenCache,
        seg: &str,
        chunk: &SpanChunk,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        let n = chunk.n;
        let spans = &chunk.p[..n];
        // Room for every span's inline ids: the unconditional store of one never
        // leaves the allocation (re-established after each slow path).
        out.reserve(PT_INLINE * n);
        let (mut slots, mut mask) = (cache.slots.as_ptr(), cache.mask);
        let (mut optr, mut olen) = (out.as_mut_ptr(), out.len());
        // Fold look-ahead only where misses probe the fold (see `process_miss`).
        let fold = (self.ignore_merges && !cache.seeded).then_some(&self.fold);
        for p in &chunk.p[..PROBE_AHEAD] {
            // SAFETY: `hash & mask < slots.len()`; a prefetch reads nothing.
            prefetch_read(unsafe { slots.add(p.hash as usize & mask) } as *const u8);
        }
        for (i, p) in spans.iter().enumerate() {
            // SAFETY: `i + PROBE_AHEAD < chunk.p.len()` (the slack entries).
            let q = unsafe { chunk.p.get_unchecked(i + PROBE_AHEAD) };
            // SAFETY: as above.
            prefetch_read(unsafe { slots.add(q.hash as usize & mask) } as *const u8);
            if let Some(fold) = fold {
                // SAFETY: `i + PEEK_AHEAD < chunk.p.len()`; `hash & mask < slots.len()`.
                let r = unsafe { chunk.p.get_unchecked(i + PEEK_AHEAD) };
                let f = unsafe { &*slots.add(r.hash as usize & mask) };
                if f.key != r.key && r.key != LONG_KEY {
                    fold.prefetch(r.hash);
                }
            }
            // SAFETY: `hash & mask < slots.len()` for the table `mask` belongs to
            // (re-read after any slow path). A long span (`LONG_KEY`) reads some
            // slot and never matches it.
            let e = unsafe { &*slots.add(p.hash as usize & mask) };
            let len = e.len as usize;
            let hit = (e.key == p.key) & (len <= PT_INLINE);
            // SAFETY: `PT_INLINE` ids fit past `olen` (reserved above).
            unsafe { std::ptr::copy_nonoverlapping(e.v.as_ptr(), optr.add(olen), PT_INLINE) };
            if hit {
                olen += len;
                continue;
            }
            // SAFETY: the ids below `olen` are written.
            unsafe { out.set_len(olen) };
            // SAFETY: scanner spans are in-bounds and on char boundaries.
            let text = unsafe { seg.get_unchecked(p.start as usize..p.end as usize) };
            self.process_pending_slow(cache, text, p.key, p.hash, out)?;
            out.reserve(PT_INLINE * (n - i));
            (slots, mask) = (cache.slots.as_ptr(), cache.mask);
            (optr, olen) = (out.as_mut_ptr(), out.len());
        }
        // SAFETY: as above.
        unsafe { out.set_len(olen) };
        Ok(())
    }

    #[inline(never)]
    fn process_pending_slow(
        &self,
        cache: &mut PretokenCache,
        text: &str,
        key: u128,
        hash: u64,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        if key == LONG_KEY {
            #[cfg(feature = "diag")]
            DIAG.with(|d| {
                let mut a = d.get();
                a[0] += 1;
                a[1] += 1;
                a[5] += text.len() as u64;
                d.set(a);
            });
            return self.process_long_span(cache, text, hash, out);
        }
        #[cfg(feature = "diag")]
        DIAG.with(|d| {
            let mut a = d.get();
            a[0] += 1;
            d.set(a);
        });
        match cache.get_or_vacancy_promote(key, hash, out) {
            Ok(()) => {
                #[cfg(feature = "diag")]
                DIAG.with(|d| {
                    let mut a = d.get();
                    a[3] += 1;
                    d.set(a);
                });
                Ok(())
            }
            Err(vacancy) => self.process_miss(cache, text, key, hash, vacancy, out),
        }
    }

    /// Like [`Self::tokenize_scanned_segment`], but also appends fine-grained reuse
    /// boundaries to `bounds`: for every pretoken that begins at a hard boundary
    /// (preceded by a `\r`/`\n`, followed by an ASCII non-whitespace byte), the
    /// `(local_byte_offset, local_token_index)` at that point — i.e.
    /// `out[..token_index]` is exactly the encoding of `seg[..byte_offset]`.
    /// Used only by the prefix cache's (cold) miss path, so the extra per-
    /// pretoken check stays out of the hot [`Self::tokenize_scanned_segment`].
    /// (Not recorded for GPT-2, whose whitespace before such a pretoken can
    /// encode differently once the text is cut there.)
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
                self.bind_cache(&mut cache);
            }
            crate::pre_tokenizers::scan::scan_core(kind, seg, |start, end| {
                // (Not for GPT-2: without `\s*[\r\n]+` the whitespace before such
                // a pretoken may encode differently once the text is cut there.)
                if kind != crate::pre_tokenizers::scan::ScanKind::Gpt2
                    && start > 0
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
            shared_pt: OnceLock::new(),
            pt_seed: self.pt_seed.clone(),
            id_to_token: self.id_to_token.clone(),
            token_to_id: self.token_to_id.clone(),
            byte_to_initial_token: self.byte_to_initial_token,
            byte_fallback_token_ids: self.byte_fallback_token_ids,
            single_char_token: self.single_char_token,
            pair_initial: self.pair_initial.clone(),
            byte_compact: self.byte_compact,
            unmap: self.unmap.clone(),
            any_unsafe: self.any_unsafe,
            fold: self.fold.clone(),
            atoms: self.atoms.clone(),
            long_vocab: self.long_vocab.clone(),
            merge_adj: self.merge_adj.clone(),
            compact_id: self.compact_id.clone(),
            merge_grid: self.merge_grid.clone(),
            code_grid: self.code_grid.clone(),
            pair_filter: self.pair_filter.clone(),
            unmap_offset: self.unmap_offset,
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

    /// The shared pretoken cache never returns a torn entry: under concurrent
    /// writers overwriting colliding slots, every hit holds exactly the ids that
    /// were stored for its key.
    #[test]
    fn packed_keys_match_the_reference_packer() {
        let buf: Vec<u8> = (0..64u32).map(|i| (i * 37 + 11) as u8).collect();
        for start in 0..buf.len() {
            for len in 1..=15.min(buf.len() - start) {
                assert_eq!(
                    PretokenCache::pack_key_at(&buf, start, len),
                    PretokenCache::pack_key(&buf[start..start + len]).unwrap(),
                    "start {start} len {len}"
                );
            }
        }
    }

    #[test]
    fn shared_pretoken_cache_hits_are_never_torn() {
        let cache = SharedPretokenCache::new();
        // Keys collide on few slots (hash & 1023) to force overwrites.
        let ids_of = |k: u64| -> Vec<u32> {
            let n = 1 + (k % 3) as usize;
            (0..n)
                .map(|i| (k as u32).wrapping_mul(2654435761).wrapping_add(i as u32))
                .collect()
        };
        let key = |k: u64| (k as u128) | (1u128 << 120);
        let hits = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for t in 0..8u64 {
                let (cache, hits) = (&cache, &hits);
                sc.spawn(move || {
                    let mut out = Vec::new();
                    for i in 0..200_000u64 {
                        let k = (i * 7 + t * 13) % 5000;
                        let h = k & 1023;
                        if (i + t) % 3 == 0 {
                            cache.insert(key(k), h, &ids_of(k));
                        } else {
                            out.clear();
                            if cache.get(key(k), h, &mut out) {
                                assert_eq!(out, ids_of(k), "torn entry for key {k}");
                                hits.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });
        assert!(hits.load(Ordering::Relaxed) > 0);
        // Entries that do not fit inline are not cached.
        cache.insert(key(1), 1, &[1, 2, 3, 4]);
        let mut out = Vec::new();
        assert!(!cache.get(key(1), 1, &mut out) || out == ids_of(1));
    }
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
        // The vocab has valid rank codes, so the short merges below run the
        // rank-code engine (`merge_small_codes`).
        assert!(bpe.code_grid.is_some() && bpe.merge_adj.buckets.is_empty());
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
                let mut ext = Vec::new();
                bpe.merge_all_raw_into(input, &mut small).unwrap();
                bpe.merge_all_raw_heap_into(input, &mut heap).unwrap();
                bpe.merge_all_raw_ext_ref(input, &mut ext).unwrap();
                assert_eq!(small, ext, "short merge mismatch for {input:?}");
                assert_eq!(heap, ext, "compact heap mismatch for {input:?}");
            }
        }
    }

    /// Pre-formed [`CharAtoms`] never change a merge result, on a GLM-like vocab:
    /// a fragment token (`\x84件`, spanning the end of 的 and 件) that outranks 件's
    /// own formation, and a merge using 件 as an operand before 件 forms (so 件
    /// is no atom). Checked against the bytes-only engine on random strings.
    #[test]
    fn char_atoms_match_byte_merges() {
        let enc = |b: &[u8]| {
            b.iter()
                .map(|&x| BYTE_TO_CHAR[x as usize])
                .collect::<String>()
        };
        let mut vocab: Vocab = (0..=255u8).map(|b| (enc(&[b]), b as u32)).collect();
        let mut merges: Vec<Value> = Vec::new();
        let (de, jian, zhong, wen) = (
            "的".as_bytes(),
            "件".as_bytes(),
            "中".as_bytes(),
            "文".as_bytes(),
        );
        let mut add = |a: &[u8], b: &[u8]| {
            let t = [a, b].concat();
            let n = vocab.len() as u32;
            vocab.entry(enc(&t)).or_insert(n);
            merges.push(Value::String(format!("{} {}", enc(a), enc(b))));
        };
        // Fragments longer than a word, formed before the atom 中 is:
        // `aaaaaaaaa\xe4` (a 9-byte head of it) and `\xadbbbbbbbbb` (a 9-byte tail).
        add(b"a", b"a");
        add(b"aa", b"aa");
        add(b"aaaa", b"aaaa");
        add(b"aaaaaaaa", b"a");
        add(b"aaaaaaaaa", &jian[..1]);
        add(b"b", b"b");
        add(b"bb", b"bb");
        add(b"bbbb", b"bbbb");
        add(b"bbbbbbbb", b"b");
        add(&zhong[2..], b"bbbbbbbbb");
        add(&zhong[..1], &zhong[1..2]);
        add(&wen[..1], &wen[1..2]);
        add(&zhong[..2], &zhong[2..]); // 中 and 文 form early
        add(&wen[..2], &wen[2..]);
        add(&de[..1], &de[1..2]);
        add(&jian[..1], &jian[1..2]);
        add(&de[2..], &jian[..2]); // \x84 + first two bytes of 件
        add(&[de[2], jian[0], jian[1]], &jian[2..]); // → \x84件
        add(jian, zhong); // uses 件 before it forms...
        add(zhong, wen); // ...and 中文 fires in between: pre-forming 件 would pick 件中
        add(&de[..2], &de[2..]); // 的
        add(&jian[..2], &jian[2..]); // 件
        add(de, zhong); // 的中
        add(b"a", b"b");
        let merge_map = parse_merges(&vocab, &merges).unwrap();
        let bpe = Bpe::new(&vocab, merge_map).unwrap();
        assert!(!bpe.atoms.bmp.is_empty());
        let pool = [
            "的",
            "件",
            "中",
            "文",
            "a",
            "b",
            " ",
            "é",
            "aaaaaaaaa",
            "bbbbbbbbb",
        ];
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut used = 0;
        for round in 0..24000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Mostly short words; every 8th long enough that a segment outgrows
            // the small engine.
            let len = if round % 8 == 0 {
                1 + (state % 150) as usize
            } else {
                1 + (state % 12) as usize
            };
            let mut w = String::new();
            let mut x = state;
            for _ in 0..len {
                w.push_str(pool[(x % pool.len() as u64) as usize]);
                x /= pool.len() as u64;
                if x == 0 {
                    x = state.rotate_left(17) | 1;
                }
            }
            let mut a = Vec::new();
            if !bpe.merge_with_atoms(&w, &mut a).unwrap() {
                continue;
            }
            used += 1;
            let mut b = Vec::new();
            if w.len() <= SMALL_MERGE_MAX {
                bpe.merge_small_into::<SMALL_MERGE_MAX>(w.as_bytes(), &mut b)
                    .unwrap();
            } else {
                bpe.merge_all_raw_heap_into(&w, &mut b).unwrap();
            }
            assert_eq!(a, b, "atoms vs bytes for {w:?}");
        }
        assert!(used > 1000, "atoms applied to only {used} words");
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

    /// A thread moving between models parks each one's seeded cache and resumes
    /// it on coming back (entries and all, no reseed); past `PARKED_CACHES` the
    /// least recently used one is recycled. Every encode stays exact.
    #[test]
    fn switching_models_resumes_parked_caches() {
        __bench_reset_thread_caches();
        let models: Vec<Bpe> = (0..PARKED_CACHES + 2).map(|_| test_bpe()).collect();
        let bound = || {
            TL_FUSED_CACHE.with(|c| {
                let c = c.borrow();
                (c.bpe_id, c.seeded, c.len)
            })
        };
        let parked =
            || TL_PARKED_CACHES.with(|p| p.borrow().iter().map(|c| c.bpe_id).collect::<Vec<_>>());
        let encode = |m: &Bpe, s: &str| {
            let mut out = Vec::new();
            m.tokenize_into_fused(s, &mut out).unwrap();
            out
        };
        // Model 0 caches a pretoken past its seed (not a vocab entry).
        assert_eq!(encode(&models[0], "abcdab"), vec![6, 4]);
        let (id, seeded, len0) = bound();
        assert_eq!((id, seeded), (models[0].id, true));
        // Away to model 1 and back: model 0's cache is resumed as it was left.
        assert_eq!(encode(&models[1], "cdab"), vec![5, 4]);
        assert_eq!(bound().0, models[1].id);
        assert_eq!(parked(), vec![models[0].id]);
        assert_eq!(encode(&models[0], "abcdab"), vec![6, 4]);
        assert_eq!(bound(), (models[0].id, true, len0));
        assert_eq!(parked(), vec![models[1].id]);
        // Round-robin over more models than park: the least recent (0) is recycled.
        for m in &models[1..] {
            assert_eq!(encode(m, "abcdab"), vec![6, 4]);
        }
        let ids: Vec<usize> = models[1..PARKED_CACHES + 1]
            .iter()
            .rev()
            .map(|m| m.id)
            .collect();
        assert_eq!(parked(), ids);
        assert_eq!(encode(&models[0], "abcdab"), vec![6, 4]);
        assert_eq!(bound().0, models[0].id);
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
