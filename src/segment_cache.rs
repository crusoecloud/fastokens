//! Opt-in cache of the token ids of text segments between added tokens.
//!
//! Chat-template output, the input a serving stack encodes, is text segments
//! separated by added tokens (`<|im_start|>`, `<|im_end|>`, tool-call
//! markers, …). A chat or coding agent re-sends its whole history on every
//! request, so all but the newest segments were already encoded by an earlier
//! request, but not as a byte prefix of it: templates drop the reasoning of
//! earlier turns, and agents compact their history into a summary, so the
//! prefix cache cannot reuse them. This cache reuses them by content.
//!
//! A segment's ids depend on nothing but its normalized text: added tokens are
//! hard boundaries, the normalizer runs per segment, and every pre-tokenizer in
//! this crate and the model act on each split on its own, independent of its
//! position in the input (a pre-tokenizer that treated the first split
//! differently, like HF's `Metaspace` with `prepend_scheme: first`, would break
//! this; none is implemented). Every hit is confirmed against the stored text,
//! so a hash collision is a miss, never a wrong answer, and the result is
//! identical to encoding from scratch. Inputs that are a single segment are not
//! cached, so plain-text traffic is unaffected.
//!
//! The cache is sharded (concurrent sessions hit it from the thread pool, so
//! one lock would serialize them), with a byte budget per shard, and evicts
//! with two generations rather than an LRU, which would need a linked list
//! over thousands of entries: entries are inserted into a young map; when the
//! next insert would overfill it, the old map is dropped and the young one
//! becomes old. A hit on an old entry moves it back into the young map, so
//! recently used segments survive. Both generations stay within the shard's
//! share of the budget, so the cache as a whole stays within `max_bytes`. A segment costing
//! more than 1/64 of the budget is never cached (with a 1 MiB budget, one of
//! more than about 8 KB of text), so the budget should be sized for the
//! longest segments, tool outputs typically, not just for the total.

use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

/// Number of independently locked shards.
const SHARDS: usize = 16;

/// Bookkeeping bytes charged per entry on top of its text and ids.
const ENTRY_OVERHEAD: usize = 64;

/// 64-bit hash of `bytes`.
///
/// Four multiply-rotate lanes over 32-byte blocks keep the loop from being a
/// single dependency chain, and a murmur3 finalizer mixes every output bit.
/// It is not collision-resistant and need not be: a hit is confirmed by
/// comparing the stored text, so a collision only costs a cache miss.
fn hash_bytes(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    #[inline(always)]
    fn round(acc: u64, word: u64) -> u64 {
        (acc ^ word).wrapping_mul(K).rotate_left(29)
    }
    #[inline(always)]
    fn word(bytes: &[u8]) -> u64 {
        let mut buf = [0u8; 8];
        buf[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(buf)
    }

    let mut lanes: [u64; 4] = [
        0x243F_6A88_85A3_08D3,
        0x1319_8A2E_0370_7344,
        0xA409_3822_299F_31D0,
        0x082E_FA98_EC4E_6C89,
    ];
    let mut i = 0;
    while i + 32 <= bytes.len() {
        for (k, lane) in lanes.iter_mut().enumerate() {
            *lane = round(*lane, word(&bytes[i + 8 * k..i + 8 * k + 8]));
        }
        i += 32;
    }
    let mut h = bytes.len() as u64;
    for lane in lanes {
        h = round(h, lane);
    }
    for chunk in bytes[i..].chunks(8) {
        h = round(h, word(chunk));
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^ (h >> 33)
}

/// Map hasher for keys that are already [`hash_bytes`] outputs.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8) | u64::from(b);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

type HashMapU64<V> = HashMap<u64, V, BuildHasherDefault<PassThrough>>;

struct Entry {
    text: Box<[u8]>,
    ids: Arc<[u32]>,
}

impl Entry {
    fn cost(&self) -> usize {
        self.text.len() + self.ids.len() * 4 + ENTRY_OVERHEAD
    }
}

#[derive(Default)]
struct Shard {
    young: HashMapU64<Entry>,
    old: HashMapU64<Entry>,
    young_bytes: usize,
    old_bytes: usize,
}

impl Shard {
    /// Insert into the young generation, first turning it old if `entry`
    /// would not fit. Keeps `young_bytes <= budget` for entries costing at
    /// most `budget`.
    fn insert_young(&mut self, hash: u64, entry: Entry, budget: usize) {
        if let Some(replaced) = self.young.remove(&hash) {
            self.young_bytes -= replaced.cost();
        }
        if self.young_bytes + entry.cost() > budget {
            self.old = std::mem::take(&mut self.young);
            self.old_bytes = std::mem::take(&mut self.young_bytes);
        }
        self.young_bytes += entry.cost();
        self.young.insert(hash, entry);
    }

    fn remove_old(&mut self, hash: u64) -> Option<Entry> {
        let entry = self.old.remove(&hash)?;
        self.old_bytes -= entry.cost();
        Some(entry)
    }
}

/// A snapshot of a segment cache's activity and size, from
/// [`crate::Tokenizer::segment_cache_stats`].
///
/// Every text segment of an input that added tokens split is a hit or a miss
/// (an input without any is neither): a hit's ids were reused, a miss was
/// tokenized.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentCacheStats {
    /// Segments whose ids were reused.
    pub hits: u64,
    /// Segments that were tokenized.
    pub misses: u64,
    /// Bytes of segment text the hits covered.
    pub hit_bytes: u64,
    /// Bytes of segment text the misses covered.
    pub miss_bytes: u64,
    /// Segments currently cached.
    pub entries: usize,
    /// Bytes currently charged against the budget: text, ids and a fixed
    /// overhead per entry.
    pub bytes: usize,
    /// The budget the cache was created with.
    pub max_bytes: usize,
}

/// Content-addressed cache of text-segment encodings. See the module docs.
pub(crate) struct SegmentCache {
    shards: Box<[Mutex<Shard>]>,
    /// Byte budget of each shard's young generation; the old generation is at
    /// most as large, so the shards together stay within `max_bytes`.
    young_budget: usize,
    max_bytes: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    hit_bytes: AtomicU64,
    miss_bytes: AtomicU64,
}

impl SegmentCache {
    /// A cache holding at most `max_bytes` of segment text and ids.
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| Mutex::default()).collect(),
            young_budget: (max_bytes / SHARDS / 2).max(1),
            max_bytes,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            hit_bytes: AtomicU64::new(0),
            miss_bytes: AtomicU64::new(0),
        }
    }

    /// Append the ids of the segment `text` to `out`: the cached ones, or
    /// those `tokenize` appends, which are then cached.
    pub(crate) fn encode<E>(
        &self,
        text: &str,
        out: &mut Vec<u32>,
        tokenize: impl FnOnce(&mut Vec<u32>) -> Result<(), E>,
    ) -> Result<(), E> {
        let hash = hash_bytes(text.as_bytes());
        if let Some(ids) = self.get(hash, text.as_bytes()) {
            out.extend_from_slice(&ids);
            return Ok(());
        }
        let start = out.len();
        tokenize(out)?;
        self.insert(hash, text.as_bytes(), &out[start..]);
        Ok(())
    }

    fn shard(&self, hash: u64) -> &Mutex<Shard> {
        // Bits the maps' bucket index (low) and tag (top) do not use.
        &self.shards[(hash >> 32) as usize % SHARDS]
    }

    /// The cached ids of `text`, whose [`hash_bytes`] is `hash`, counted as a
    /// hit or a miss.
    fn get(&self, hash: u64, text: &[u8]) -> Option<Arc<[u32]>> {
        let ids = self.lookup(hash, text);
        let (count, bytes) = match ids {
            Some(_) => (&self.hits, &self.hit_bytes),
            None => (&self.misses, &self.miss_bytes),
        };
        count.fetch_add(1, Relaxed);
        bytes.fetch_add(text.len() as u64, Relaxed);
        ids
    }

    fn lookup(&self, hash: u64, text: &[u8]) -> Option<Arc<[u32]>> {
        let mut shard = self.shard(hash).lock().unwrap();
        if let Some(entry) = shard.young.get(&hash) {
            return (*entry.text == *text).then(|| entry.ids.clone());
        }
        if shard
            .old
            .get(&hash)
            .is_none_or(|entry| *entry.text != *text)
        {
            return None;
        }
        let entry = shard.remove_old(hash).unwrap();
        let ids = entry.ids.clone();
        shard.insert_young(hash, entry, self.young_budget);
        Some(ids)
    }

    /// Cache `ids` as the encoding of `text`, whose [`hash_bytes`] is `hash`.
    /// A segment costing more than half a generation's budget (1/64 of the
    /// cache's) is not cached: it would evict everything else by itself.
    fn insert(&self, hash: u64, text: &[u8], ids: &[u32]) {
        let entry = Entry {
            text: text.into(),
            ids: ids.into(),
        };
        if entry.cost() > self.young_budget / 2 {
            return;
        }
        let mut shard = self.shard(hash).lock().unwrap();
        shard.remove_old(hash);
        shard.insert_young(hash, entry, self.young_budget);
    }

    /// Drop every entry. The counters keep counting.
    pub(crate) fn clear(&self) {
        for shard in self.shards.iter() {
            *shard.lock().unwrap() = Shard::default();
        }
    }

    pub(crate) fn stats(&self) -> SegmentCacheStats {
        let (mut entries, mut bytes) = (0, 0);
        for shard in self.shards.iter() {
            let shard = shard.lock().unwrap();
            entries += shard.young.len() + shard.old.len();
            bytes += shard.young_bytes + shard.old_bytes;
        }
        SegmentCacheStats {
            hits: self.hits.load(Relaxed),
            misses: self.misses.load(Relaxed),
            hit_bytes: self.hit_bytes.load(Relaxed),
            miss_bytes: self.miss_bytes.load(Relaxed),
            entries,
            bytes,
            max_bytes: self.max_bytes,
        }
    }
}

/// Build a [`SegmentCache`] from the `FASTOKENS_SEGMENT_CACHE` env var (a size
/// in MiB), or `None` if it is unset or zero, the default.
pub(crate) fn from_env() -> Option<SegmentCache> {
    std::env::var("FASTOKENS_SEGMENT_CACHE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mib| mib >= 1)
        .map(|mib| SegmentCache::new(mib.saturating_mul(1 << 20)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids_of(cache: &SegmentCache, text: &[u8]) -> Option<Arc<[u32]>> {
        cache.get(hash_bytes(text), text)
    }

    fn put(cache: &SegmentCache, text: &[u8], ids: &[u32]) {
        cache.insert(hash_bytes(text), text, ids);
    }

    #[test]
    fn hash_depends_on_every_byte_and_the_length() {
        let base = b"<|im_start|>user\nread src/lib.rs and fix the bug<|im_end|>".to_vec();
        let h = hash_bytes(&base);
        for i in 0..base.len() {
            let mut changed = base.clone();
            changed[i] ^= 1;
            assert_ne!(hash_bytes(&changed), h, "byte {i}");
        }
        assert_ne!(hash_bytes(&base[..base.len() - 1]), h);
        assert_ne!(hash_bytes(b""), hash_bytes(b"\0"));
    }

    #[test]
    fn encode_tokenizes_once_and_counts() {
        let cache = SegmentCache::new(1 << 20);
        let mut calls = 0;
        let mut tokenize = |out: &mut Vec<u32>| {
            calls += 1;
            out.extend([1, 2, 3]);
            Ok::<(), ()>(())
        };
        let mut out = vec![9];
        cache
            .encode("hello world", &mut out, &mut tokenize)
            .unwrap();
        cache
            .encode("hello world", &mut out, &mut tokenize)
            .unwrap();
        assert_eq!(out, [9, 1, 2, 3, 1, 2, 3]);
        assert_eq!(calls, 1);
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses), (1, 1));
        assert_eq!((stats.hit_bytes, stats.miss_bytes), (11, 11));
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.bytes, 11 + 12 + ENTRY_OVERHEAD);
        assert_eq!(stats.max_bytes, 1 << 20);
    }

    #[test]
    fn a_hash_collision_is_a_miss_not_a_wrong_answer() {
        let cache = SegmentCache::new(1 << 20);
        cache.insert(42, b"first", &[1]);
        assert!(cache.get(42, b"second").is_none());
        cache.insert(42, b"second", &[2]);
        assert_eq!(&*cache.get(42, b"second").unwrap(), &[2]);
        assert!(cache.get(42, b"first").is_none());
    }

    #[test]
    fn stays_within_budget_and_keeps_recently_used_entries() {
        let budget = 64 * 1024;
        let cache = SegmentCache::new(budget);
        let hot = b"the system prompt every request repeats";
        put(&cache, hot, &[7; 10]);
        // Small entries, and entries as large as the cache accepts.
        let large = vec![b'x'; budget / 64 - ENTRY_OVERHEAD - 4 * 8];
        for i in 0..10_000u32 {
            let text = format!("tool output number {i}");
            put(&cache, text.as_bytes(), &[i; 8]);
            if i % 3 == 0 {
                let mut large = large.clone();
                large[..4].copy_from_slice(&i.to_le_bytes());
                put(&cache, &large, &[i; 8]);
            }
            // Touching the hot entry keeps promoting it out of the old generation.
            assert!(ids_of(&cache, hot).is_some(), "hot entry evicted at {i}");
            let stats = cache.stats();
            assert!(stats.bytes <= budget, "{} bytes at {i}", stats.bytes);
        }
        assert!(cache.stats().entries > 0);
    }

    #[test]
    fn oversized_segments_are_not_cached() {
        let cache = SegmentCache::new(SHARDS * 2 * 1024);
        let big = vec![b'x'; 4096];
        put(&cache, &big, &[1; 100]);
        assert!(ids_of(&cache, &big).is_none());
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn clear_drops_everything() {
        let cache = SegmentCache::new(1 << 20);
        put(&cache, b"a", &[1]);
        cache.clear();
        assert!(ids_of(&cache, b"a").is_none());
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.bytes), (0, 0));
    }
}
