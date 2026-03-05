use std::{
    cell::RefCell,
    collections::HashMap,
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};
use serde::Deserialize;
use serde_json::Value;

use super::Result;
use crate::pre_tokenizers::encode_bytes_into;

type TokenId = u32;
type ParsedMergeMap = HashMap<(u32, u32), (u32, u32)>;
type Vocab = HashMap<String, u32>;

const INVALID_TOKEN: u32 = u32::MAX;

/// Open-addressing hash table for merge lookups using FxHash.
///
/// Stores `(left_id, right_id) → merged_id` where the merged_id encodes
/// the token produced by the merge. Uses u64 keys (packed pair) and linear
/// probing. Empty slots are marked with `EMPTY_KEY`.
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
        // Target ~50% load factor, round up to power of 2.
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

    /// Look up the merged token ID for a pair. Returns `None` if no merge.
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

#[inline(always)]
fn pack_pair(t1: u32, t2: u32) -> u64 {
    (t1 as u64) << 32 | t2 as u64
}

#[inline(always)]
fn fx_hash(key: u64) -> u64 {
    key.wrapping_mul(0x517cc1b727220a95)
}

/// FxHash-based [`BuildHasher`] for the BPE token cache.
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
const FLAT_CACHE_SIZE: usize = 1 << FLAT_CACHE_BITS;
const FLAT_CACHE_MASK: usize = FLAT_CACHE_SIZE - 1;
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

/// Maximum load factor (fraction of slots occupied) before the cache is cleared.
/// At 75% occupancy, linear probing chains average ~2.5 probes; beyond this,
/// performance degrades rapidly and approaching 100% causes infinite loops.
const FLAT_CACHE_MAX_LOAD: usize = FLAT_CACHE_SIZE * 3 / 4;

struct FlatCache {
    bpe_id: usize,
    slots: Vec<CacheSlot>,
    pool: Vec<u32>,
    key_pool: Vec<u8>,
    count: usize,
}

impl FlatCache {
    fn new() -> Self {
        Self {
            bpe_id: 0,
            slots: vec![
                CacheSlot {
                    hash: EMPTY_SLOT,
                    offset: 0,
                    len: 0,
                    key_len: 0,
                    key_offset: 0,
                };
                FLAT_CACHE_SIZE
            ],
            pool: Vec::with_capacity(256 * 1024),
            key_pool: Vec::with_capacity(512 * 1024),
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
        let mut idx = hash as usize & FLAT_CACHE_MASK;
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
            idx = (idx + 1) & FLAT_CACHE_MASK;
        }
    }

    #[inline(always)]
    fn insert(&mut self, key: &str, ids: &[u32]) {
        if self.count >= FLAT_CACHE_MAX_LOAD {
            self.clear();
        }
        let hash = Self::hash_str(key);
        let key_bytes = key.as_bytes();
        let mut idx = hash as usize & FLAT_CACHE_MASK;
        loop {
            let slot = unsafe { self.slots.get_unchecked(idx) };
            let h = slot.hash;
            if h == EMPTY_SLOT {
                self.count += 1;
                let offset = self.pool.len() as u32;
                self.pool.extend_from_slice(ids);
                let key_offset = self.key_pool.len() as u32;
                self.key_pool.extend_from_slice(key_bytes);
                let slot = unsafe { self.slots.get_unchecked_mut(idx) };
                slot.hash = hash;
                slot.offset = offset;
                slot.len = ids.len() as u16;
                slot.key_offset = key_offset;
                slot.key_len = key_bytes.len() as u16;
                return;
            }
            if h == hash {
                let ks = slot.key_offset as usize;
                let ke = ks + slot.key_len as usize;
                if unsafe { self.key_pool.get_unchecked(ks..ke) } == key_bytes {
                    // Same key — update the cached value.
                    let offset = self.pool.len() as u32;
                    self.pool.extend_from_slice(ids);
                    let slot = unsafe { self.slots.get_unchecked_mut(idx) };
                    slot.offset = offset;
                    slot.len = ids.len() as u16;
                    return;
                }
                // Hash collision with different key — continue probing.
            }
            idx = (idx + 1) & FLAT_CACHE_MASK;
        }
    }
}

thread_local! {
    static TL_BPE_CACHE: RefCell<FlatCache> = RefCell::new(FlatCache::new());
    static TL_FUSED_CACHE: RefCell<FlatCache> = RefCell::new(FlatCache::new());
    static DP_SCRATCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

const CACHE_SHARDS: usize = 64;

struct SharedCache {
    shards: Vec<Mutex<FxHashMap<String, Vec<u32>>>>,
}

impl SharedCache {
    fn new() -> Self {
        Self {
            shards: (0..CACHE_SHARDS)
                .map(|_| Mutex::new(HashMap::with_hasher(FxBuildHasher)))
                .collect(),
        }
    }

    #[inline]
    fn shard_index(key: &str) -> usize {
        let bytes = key.as_bytes();
        let mut h: u64 = bytes.len() as u64;
        for &b in &bytes[..bytes.len().min(8)] {
            h = h
                .wrapping_add(b as u64)
                .wrapping_mul(0x9E3779B97F4A7C15);
        }
        h as usize & (CACHE_SHARDS - 1)
    }

    #[inline]
    fn get_into(&self, key: &str, out: &mut Vec<u32>) -> bool {
        let shard = self.shards[Self::shard_index(key)].lock().unwrap();
        if let Some(ids) = shard.get(key) {
            out.extend_from_slice(ids);
            true
        } else {
            false
        }
    }

    fn insert(&self, key: String, value: Vec<u32>) {
        self.shards[Self::shard_index(&key)]
            .lock()
            .unwrap()
            .insert(key, value);
    }
}

/// Raw deserialization helper for [`Bpe`].
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
    #[allow(dead_code)]
    byte_fallback: bool,
}

/// Monotonic counter for unique Bpe instance IDs, avoiding address-reuse
/// bugs where the allocator gives a new Bpe the same address as a freed one.
static BPE_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

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
    fused_shared_cache: SharedCache,
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
}

impl TryFrom<RawBpe> for Bpe {
    type Error = String;

    fn try_from(raw: RawBpe) -> Result<Self> {
        let merge_map = parse_merges(&raw.vocab, &raw.merges)?;
        Self::new(&raw.vocab, merge_map)
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
            .map(|t| vocab_r[&t].len() as u16)
            .collect();

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

        Ok(Self {
            id: BPE_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            daac,
            merge_map: flat_merge_map,
            unmerge_map,
            next_prefix_map,
            token_lens,
            shared_cache: SharedCache::new(),
            fused_shared_cache: SharedCache::new(),
            id_to_token,
            token_to_id: vocab.clone(),
        })
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

        if let Some(token) = self.next_match(input) {
            if self.token_lens[token as usize] as usize == input.len() {
                out.push(token);
                return Ok(());
            }
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

        if self.try_greedy_into(input, out)? {
            let ids = &out[start..];
            TL_BPE_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.bpe_id != bpe_id {
                    c.bpe_id = bpe_id;
                    c.clear();
                }
                c.insert(input, ids);
            });
            self.shared_cache.insert(input.to_string(), ids.to_vec());
            return Ok(());
        }

        self.tokenize_dp_into(input, out)?;

        let ids = &out[start..];
        TL_BPE_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                c.bpe_id = bpe_id;
                c.clear();
            }
            c.insert(input, ids);
        });
        self.shared_cache.insert(input.to_string(), ids.to_vec());

        Ok(())
    }

    /// Greedy tokenization: always take the longest match, then verify all
    /// adjacent pairs are BPE-compatible. Returns `true` if the greedy result
    /// is correct; on failure, truncates `out` back and returns `false`.
    #[inline(always)]
    fn try_greedy_into(&self, input: &str, out: &mut Vec<u32>) -> Result<bool> {
        let start = out.len();
        let mut pos = 0;
        while pos < input.len() {
            let Some(token) = self.next_match(&input[pos..]) else {
                out.truncate(start);
                return Ok(false);
            };
            out.push(token);
            pos += self.token_lens[token as usize] as usize;
        }

        let tokens = &out[start..];
        if tokens.len() < 2 {
            return Ok(true);
        }
        for i in 0..tokens.len() - 1 {
            if !self.is_compatible_token_pair(tokens[i], tokens[i + 1]) {
                out.truncate(start);
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Full DP tokenization. Used as fallback when greedy fails.
    fn tokenize_dp_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        let n = input.len();
        let mut last_token = DP_SCRATCH.with(|s| {
            let mut v = s.borrow_mut();
            v.clear();
            v.resize(n + 1, INVALID_TOKEN);
            std::mem::take(&mut *v)
        });

        let longest = self.next_match(input).ok_or("no match at position 0")?;
        let mut token = longest;
        loop {
            let end = self.token_lens[token as usize] as usize;
            if last_token[end] == INVALID_TOKEN {
                last_token[end] = token;
            }
            let shorter = self.next_prefix_map[token as usize];
            if shorter == INVALID_TOKEN {
                break;
            }
            token = shorter;
        }

        for pos in 1..n {
            let prev = last_token[pos];
            if prev == INVALID_TOKEN {
                continue;
            }
            let Some(longest) = self.next_match(&input[pos..]) else {
                continue;
            };
            let mut token = longest;
            loop {
                let end = pos + self.token_lens[token as usize] as usize;
                if last_token[end] == INVALID_TOKEN && self.is_compatible_token_pair(prev, token) {
                    last_token[end] = token;
                }
                let shorter = self.next_prefix_map[token as usize];
                if shorter == INVALID_TOKEN {
                    break;
                }
                token = shorter;
            }
        }

        if last_token[n] == INVALID_TOKEN {
            return Err(format!(
                "failed to tokenize: end of input not \
                 reachable (input length {n})"
            ));
        }

        let start = out.len();
        let mut pos = n;
        while pos > 0 {
            let token = last_token[pos];
            out.push(token);
            pos -= self.token_lens[token as usize] as usize;
        }
        out[start..].reverse();

        DP_SCRATCH.with(|s| {
            *s.borrow_mut() = last_token;
        });

        Ok(())
    }

    /// Runs BPE tokenization without interacting with any cache layers.
    /// Used by the fused batch path which manages its own caching.
    #[inline(always)]
    fn tokenize_nocache_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        if let Some(token) = self.next_match(input) {
            if self.token_lens[token as usize] as usize == input.len() {
                out.push(token);
                return Ok(());
            }
        }

        if self.try_greedy_into(input, out)? {
            return Ok(());
        }

        self.tokenize_dp_into(input, out)
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
        if self.fused_shared_cache.get_into(raw_input, out) {
            TL_FUSED_CACHE.with(|c| {
                let mut c = c.borrow_mut();
                if c.bpe_id != bpe_id {
                    c.bpe_id = bpe_id;
                    c.clear();
                }
                c.insert(raw_input, &out[start..]);
            });
            return Ok(());
        }

        let mut encoded = String::with_capacity(raw_input.len() * 2);
        encode_bytes_into(raw_input, &mut encoded);

        self.tokenize_nocache_into(&encoded, out)?;

        let ids = &out[start..];
        TL_FUSED_CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.bpe_id != bpe_id {
                c.bpe_id = bpe_id;
                c.clear();
            }
            c.insert(raw_input, ids);
        });
        self.fused_shared_cache
            .insert(raw_input.to_string(), ids.to_vec());

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

            let mut encode_buf = String::new();

            for split in splits {
                if let Some(id) = split.token_id {
                    out.push(id);
                } else if !split.range.is_empty() {
                    let text = &buffer[split.range.clone()];
                    if text.is_empty() {
                        continue;
                    }

                    if cache.get(text, out) {
                        continue;
                    }

                    let start = out.len();
                    if self.fused_shared_cache.get_into(text, out) {
                        cache.insert(text, &out[start..]);
                        continue;
                    }

                    encode_buf.clear();
                    encode_bytes_into(text, &mut encode_buf);
                    self.tokenize_nocache_into(&encode_buf, out)?;
                    cache.insert(text, &out[start..]);
                    self.fused_shared_cache
                        .insert(text.to_string(), out[start..].to_vec());
                }
            }
            Ok(())
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
            fused_shared_cache: SharedCache::new(),
            id_to_token: self.id_to_token.clone(),
            token_to_id: self.token_to_id.clone(),
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
