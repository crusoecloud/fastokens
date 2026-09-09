pub mod added_tokens;
pub mod decoders;
pub mod json_structs;
pub mod models;
pub mod normalizers;
pub mod post_processors;
pub mod pre_tokenized;
pub mod pre_tokenizers;
pub mod tiktoken;

use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::Path,
    sync::{Arc, Mutex},
};

use rayon::prelude::*;
use serde_json::Value;

pub use self::{
    added_tokens::{AddedTokenInfo, AddedTokens},
    json_structs::{
        AddedTokenConfig, DecoderConfig, DecoderKind, ModelConfig, ModelKind, NormalizerConfig,
        NormalizerKind, PostProcessorConfig, PostProcessorKind, PreTokenizerConfig,
        PreTokenizerKind, TokenizerConfig, TokenizerJson,
    },
    models::Model,
    normalizers::{Nfc, Normalizer, Replace},
    post_processors::PostProcessor,
    pre_tokenizers::{ByteLevel, Pcre2Limits, PreTokenizer, Split, SplitBehavior},
    tiktoken::{
        CL100K_BASE_PATTERN, KIMI_PATTERN, KIMI_RESERVED_SPECIAL_TOKENS, O200K_BASE_PATTERN,
        TiktokenConfig, TiktokenFamily,
    },
};

use self::{
    added_tokens::Segment,
    decoders::Decoder,
    pre_tokenized::{PreTokenizedString, Split as PtSplit},
};

#[cfg(feature = "hf-hub")]
mod hf_hub_support {
    pub use hf_hub::api::sync::ApiError;

    use super::{
        AddedTokenConfig, Error, KIMI_PATTERN, TiktokenConfig, TiktokenFamily, Tokenizer,
        TokenizerConfig, TokenizerJson, TokenizerOptions, tiktoken::parse_tiktoken_model,
    };
    use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
    use std::{collections::HashMap, fs};

    /// Build an `hf-hub` [`Api`] client, optionally overriding the token that
    /// would otherwise be read from the local HuggingFace credential cache
    /// (`~/.cache/huggingface/token`).
    pub(super) fn make_api(token: Option<&str>) -> Result<Api, ApiError> {
        match token {
            Some(t) => ApiBuilder::new().with_token(Some(t.to_owned())).build(),
            None => Api::new(),
        }
    }

    /// Validate that the model identifier is well-formed.
    fn validate_model_id(model: &str) -> Result<(), Error> {
        if model.contains("..") {
            return Err(Error::InvalidIdentifier(
                "model identifier must not contain \"..\"".into(),
            ));
        }
        Ok(())
    }

    /// Used by `Tokenizer::from_model` and `Tokenizer::from_model_with_token` to fetch
    /// `tokenizer.json` from the HuggingFace Hub and build a `Tokenizer`.
    pub fn from_model_with_token(model: &str, token: Option<&str>) -> Result<Tokenizer, Error> {
        from_model_with_token_and_options(model, token, TokenizerOptions::default())
    }

    pub fn from_model_with_token_and_options(
        model: &str,
        token: Option<&str>,
        options: TokenizerOptions,
    ) -> Result<Tokenizer, Error> {
        validate_model_id(model)?;
        let api = make_api(token)?;
        let repo = api.model(model.to_string());

        // `tokenizer.json` first: it is the common case, and `ApiRepo::get`
        // short-circuits on the local cache, so a warm cache needs no network.
        // Only a genuinely absent file falls through to the tiktoken layout —
        // a transport or auth failure must propagate, not be misread as
        // "this repo must be tiktoken".
        let json_path = match repo.get("tokenizer.json") {
            Ok(path) => path,
            Err(e) => {
                let json_err = Error::from(e);
                if !json_err.is_not_found() {
                    return Err(json_err);
                }
                // Report the missing `tokenizer.json` rather than the missing
                // `tiktoken.model` when the repo is neither: for a repo that is
                // simply misconfigured, that is the more useful error.
                return match from_tiktoken_repo(&repo)? {
                    Some(tokenizer) => Ok(tokenizer),
                    None => Err(json_err),
                };
            }
        };
        let raw = fs::read_to_string(&json_path)?;
        let json: TokenizerJson = serde_json::from_str(&raw)?;
        // Some models (e.g. Qwen2-VL) declare added tokens only in
        // `tokenizer_config.json`; fetch it too when present.
        let config_path = json_path.with_file_name("tokenizer_config.json");
        let tokenizer_config = if config_path.exists() {
            Some(serde_json::from_str(&fs::read_to_string(config_path)?)?)
        } else if repo
            .info()?
            .siblings
            .iter()
            .any(|sibling| sibling.rfilename == "tokenizer_config.json")
        {
            let config_path = repo.get("tokenizer_config.json")?;
            Some(serde_json::from_str(&fs::read_to_string(config_path)?)?)
        } else {
            None
        };
        Tokenizer::build_with_options(json, tokenizer_config, options)
    }

    /// Build a [`Tokenizer`] from a repository that ships a bare
    /// `tiktoken.model` instead of a `tokenizer.json` (e.g. Moonshot's Kimi).
    ///
    /// Returns `Ok(None)` when the repo has no `tiktoken.model` either, leaving
    /// it to the caller to report the absent `tokenizer.json`. A `tiktoken.model`
    /// that is present but unusable is an error, not a `None` — silently
    /// reporting "no tokenizer.json" would hide the real cause.
    fn from_tiktoken_repo(repo: &ApiRepo) -> Result<Option<Tokenizer>, Error> {
        let ranks_path = match repo.get("tiktoken.model") {
            Ok(path) => path,
            Err(e) => {
                let err = Error::from(e);
                if err.is_not_found() {
                    return Ok(None);
                }
                return Err(err);
            }
        };

        // The ranks file carries no pattern or special tokens, so
        // `tokenizer_config.json` is required to identify the family.
        let config_path = ranks_path.with_file_name("tokenizer_config.json");
        let config_raw = if config_path.exists() {
            fs::read_to_string(config_path)?
        } else {
            fs::read_to_string(repo.get("tokenizer_config.json").map_err(|e| {
                let err = Error::from(e);
                if err.is_not_found() {
                    Error::Tiktoken(
                        "repository has tiktoken.model but no tokenizer_config.json, so the \
                         pre-tokenization pattern cannot be determined"
                            .into(),
                    )
                } else {
                    err
                }
            })?)?
        };
        let config: TokenizerConfig = serde_json::from_str(&config_raw)?;

        let family = TiktokenFamily::detect(config.tokenizer_class(), config.auto_map_tokenizer())
            .ok_or_else(|| {
                Error::Tiktoken(format!(
                    "unrecognized tiktoken model family (tokenizer_class={:?}, \
                     auto_map.AutoTokenizer={:?}); supply the pattern explicitly via \
                     Tokenizer::from_tiktoken_file",
                    config.tokenizer_class(),
                    config.auto_map_tokenizer(),
                ))
            })?;

        let ranks = parse_tiktoken_model(&fs::read_to_string(&ranks_path)?)?;
        let declared = config.added_token_configs().map_err(Error::Tiktoken)?;

        let tokenizer = match family {
            TiktokenFamily::Kimi => {
                let num_ranks = u32::try_from(ranks.len()).map_err(|_| {
                    Error::Tiktoken(format!(
                        "tiktoken.model has too many ranks: {}",
                        ranks.len()
                    ))
                })?;
                // `TiktokenConfig::kimi` owns the reserved-window layout (which
                // ids exist and how undeclared ones are named); overlay the
                // declared entries so their flags survive instead of being
                // flattened to `special: true` across the whole window.
                //
                // On today's Kimi repos the effect is on decode, not encode:
                // every declared token has `lstrip`/`rstrip` false, so the ids
                // are identical either way. What differs is that K2.6 declares 7
                // of its 23 tokens `special: false` (K3, 3 of 16) —
                // `<|tool_call_begin|>`, `<think>`, … — and those must survive
                // `decode(skip_special_tokens = true)`.
                let by_id: HashMap<u32, &AddedTokenConfig> =
                    declared.iter().map(|c| (c.id, c)).collect();
                let named = declared.iter().map(|c| (c.id, c.content.clone()));
                let added: Vec<AddedTokenConfig> = TiktokenConfig::kimi(num_ranks, named)
                    .special_tokens
                    .into_iter()
                    .map(|(content, id)| {
                        by_id.get(&id).map_or_else(
                            || AddedTokenConfig {
                                id,
                                content,
                                single_word: false,
                                lstrip: false,
                                rstrip: false,
                                normalized: false,
                                special: true,
                            },
                            |declared| (*declared).clone(),
                        )
                    })
                    .collect();
                Tokenizer::from_tiktoken_ranks_with_added_tokens(&ranks, KIMI_PATTERN, &added)?
            }
        };
        Ok(Some(tokenizer))
    }

    /// Used by the Python layer to fetch `tokenizer.json` from the HuggingFace Hub and
    /// build a `Tokenizer`.
    pub fn download_tokenizer_json(model: &str) -> Result<String, Error> {
        validate_model_id(model)?;
        let api = make_api(None)?;
        let repo = api.model(model.to_string());
        let json_path = repo.get("tokenizer.json")?;
        Ok(fs::read_to_string(json_path)?)
    }
}

/// Errors that can occur when constructing a [`Tokenizer`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[cfg(feature = "hf-hub")]
    #[error("failed to download tokenizer files: {0}")]
    Hub(#[from] hf_hub_support::ApiError),

    #[error("failed to read tokenizer files: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse tokenizer files: {0}")]
    Json(#[from] serde_json::Error),

    #[error("normalizer error: {0}")]
    Normalizer(#[from] normalizers::Error),

    #[error("pre-tokenizer error: {0}")]
    PreTokenizer(#[from] pre_tokenizers::Error),

    #[error("post-processor error: {0}")]
    PostProcessor(#[from] post_processors::Error),

    #[error("decoder error: {0}")]
    Decoder(#[from] decoders::Error),

    #[error("model error: {0}")]
    Model(String),

    #[error("invalid tiktoken model: {0}")]
    Tiktoken(String),

    #[error("invalid model identifier: {0}")]
    InvalidIdentifier(String),
}

impl Error {
    /// Whether this error means "the requested file does not exist", as opposed
    /// to a transport, auth, or parse failure.
    ///
    /// This distinction is load-bearing for callers that probe for an optional
    /// file: a missing file is a permanent, expected outcome to be handled (try
    /// another format), whereas a network or credential failure is transient and
    /// must be propagated and retried. Conflating the two makes a permanent 404
    /// look retryable forever.
    ///
    /// Recognizes an HTTP 404 from the Hub, and a local
    /// [`std::io::ErrorKind::NotFound`]. (`hf-hub` has no offline mode — a cache
    /// miss is a `None` from the cache lookup, not an error.)
    ///
    /// The `Io` arm is unreachable from the Rust resolver, where `ApiRepo::get`
    /// yields an `ApiError` and a failed read propagates on its own. It is live
    /// for the Python layer, which classifies `download_tokenizer_json` — a `get`
    /// followed by a `read_to_string` — so a file pruned from the cache between
    /// those two steps is retried through the resolver instead of failing.
    ///
    /// Only the un-nested `RequestError(Status(404, _))` shape is matched, which
    /// is what a missing remote file produces: the metadata `HEAD` raises before
    /// the body phase, and `max_retries` defaults to 0 so nothing wraps it in
    /// `TooManyRetries`. Enabling retries would need this widened.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        match self {
            #[cfg(feature = "hf-hub")]
            Self::Hub(hf_hub_support::ApiError::RequestError(e)) => {
                matches!(e.as_ref(), ureq::Error::Status(404, _))
            }
            Self::Io(e) => e.kind() == std::io::ErrorKind::NotFound,
            _ => false,
        }
    }
}

/// Don't attempt prefix reuse unless the shared prefix is at least this many
/// bytes — below it, tokenizing from scratch is already cheap and the LCP scan
/// plus bookkeeping isn't worth it.
const PREFIX_CACHE_MIN_LCP: usize = 8 * 1024;
/// Don't reuse a cached prefix unless it covers at least this many tokens — the
/// win has to beat the fixed cost of the id copy.
const PREFIX_CACHE_MIN_REUSE_TOKENS: usize = 256;

/// One cached full encoding, retained so a later input that shares a byte prefix
/// with it can reuse the leading token ids instead of re-tokenizing them.
struct PrefixEntry {
    /// The (normalized) buffer that produced `core_ids`.
    buf: Box<[u8]>,
    /// Core token ids for `buf` — before post-processing (special tokens).
    core_ids: Arc<[u32]>,
    /// Ascending `(byte_offset, token_index)` at each newline-chunk boundary:
    /// `core_ids[..token_index]` is exactly the encoding of `buf[..byte_offset]`.
    /// Reuse is only ever cut at one of these offsets.
    bounds: Box<[(u32, u32)]>,
}

/// A reuse decision produced under the lock and applied without it.
struct Reuse {
    core_ids: Arc<[u32]>,
    /// Number of leading tokens to reuse.
    tokens: usize,
    /// Byte offset in the input from which the tail must be tokenized.
    tail_start: usize,
}

/// Bounded, opt-in **prefix cache** for the scanner fast path. It keeps a small
/// LRU of recent full encodings; when a new input shares a byte prefix with a
/// cached one, the leading token ids are copied straight from the cache (cut at
/// a hard pretoken boundary) and only the differing tail is tokenized.
///
/// This is the mechanism for **shared system prompts / long shared contexts**:
/// the shared prefix is tokenized once, then every later request that begins
/// with it pays only for its own tail. An exact repeat reuses the whole
/// encoding.
///
/// It is **off by default** ([`Tokenizer::enable_input_cache`] or the
/// `FASTOKENS_INPUT_CACHE=<capacity>` env var) because each call then does an
/// LCP scan against the cached buffers — a win only when inputs actually share
/// prefixes, and overhead on wholly-unique traffic. Reuse cuts are only ever
/// made at offsets whose following byte is ASCII non-whitespace, which is an
/// unconditional pretoken boundary, so a reused prefix can never depend on the
/// (differing) tail — the result is bit-identical to tokenizing from scratch.
struct InputCache {
    capacity: usize,
    /// Most-recent-first LRU of cached encodings.
    entries: VecDeque<PrefixEntry>,
}

/// Longest common byte prefix of `a` and `b`, compared 8 bytes at a time.
#[inline]
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
        let y = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
        if x != y {
            break;
        }
        i += 8;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// A byte after which a pretoken boundary is *unconditional* — an ASCII
/// non-whitespace byte. If `buf[p]` is such a byte and `buf[p-1]` ended a
/// newline run, `p` is a hard boundary no matter what precedes or follows.
#[inline]
fn is_hard_reuse_byte(b: u8) -> bool {
    b < 0x80 && !b.is_ascii_whitespace()
}

impl InputCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::new(),
        }
    }

    /// Drop every cached encoding. Required whenever the vocabulary changes,
    /// since entries were tokenized against the previous one.
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Decide how much of `input`'s encoding can be reused from a cached entry.
    /// Returns `None` when nothing worthwhile is shared.
    fn reuse_plan(&self, input: &[u8]) -> Option<Reuse> {
        // Pick the cached entry sharing the longest byte prefix with `input`.
        let mut best: Option<(usize, usize)> = None; // (entry index, prefix len)
        for (i, e) in self.entries.iter().enumerate() {
            let l = common_prefix_len(&e.buf, input);
            if best.is_none_or(|(_, bl)| l > bl) {
                best = Some((i, l));
            }
        }
        let (idx, l) = best?;
        let e = &self.entries[idx];

        // Exact repeat: reuse the whole encoding.
        if l == input.len() && l == e.buf.len() {
            return Some(Reuse {
                core_ids: e.core_ids.clone(),
                tokens: e.core_ids.len(),
                tail_start: input.len(),
            });
        }
        if l < PREFIX_CACHE_MIN_LCP {
            return None;
        }

        // Largest recorded boundary strictly inside the shared prefix whose
        // following byte makes it an unconditional pretoken boundary. `p < l`
        // guarantees `input[p] == e.buf[p]`, so the check holds for `input` too.
        let mut chosen: Option<(usize, usize)> = None;
        for &(bo, tk) in e.bounds.iter() {
            let p = bo as usize;
            if p >= l {
                break;
            }
            if is_hard_reuse_byte(e.buf[p]) {
                chosen = Some((p, tk as usize));
            }
        }
        let (p, tokens) = chosen?;
        if tokens < PREFIX_CACHE_MIN_REUSE_TOKENS {
            return None;
        }
        Some(Reuse {
            core_ids: e.core_ids.clone(),
            tokens,
            tail_start: p,
        })
    }

    /// Store a freshly computed full encoding (LRU-evicting the oldest entry).
    fn insert(&mut self, buf: &[u8], core_ids: &[u32], bounds: Vec<(u32, u32)>) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_back();
        }
        self.entries.push_front(PrefixEntry {
            buf: buf.into(),
            core_ids: core_ids.into(),
            bounds: bounds.into_boxed_slice(),
        });
    }
}

/// Build an [`InputCache`] from the `FASTOKENS_INPUT_CACHE` env var (a capacity),
/// or `None` if unset — the default.
fn input_cache_from_env() -> Option<Mutex<InputCache>> {
    std::env::var("FASTOKENS_INPUT_CACHE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&c| c >= 1)
        .map(|c| Mutex::new(InputCache::new(c)))
}

/// Options applied while constructing a [`Tokenizer`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenizerOptions {
    pub pcre2_limits: Pcre2Limits,
}

/// One piece of input for [`Tokenizer::encode_segments`].
///
/// A segment carries its own trust boundary: when `allow_special` is `true`,
/// added/special vocabulary entries (e.g. `<|im_end|>`) in `text` are
/// recognized as control tokens — appropriate for trusted chat-template output.
/// When `false`, `text` is encoded as ordinary content, so a literal
/// `<|im_end|>` becomes plain tokens and cannot be injected by untrusted input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeSegment<'a> {
    /// The text of this segment.
    pub text: &'a str,
    /// Whether special/added tokens are recognized within `text`.
    pub allow_special: bool,
}

/// Which added-vocabulary entries an encode recognizes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AddedTokenPolicy {
    /// Recognize every added token. The default.
    #[default]
    All,
    /// Recognize only the added tokens that are not flagged `special`; special
    /// ones are encoded as ordinary text. Mirrors `encode_special_tokens` in
    /// HuggingFace `tokenizers`, which is what `transformers` sets for
    /// `split_special_tokens=True`.
    SkipSpecial,
    /// Recognize none: the whole input goes through the base pipeline.
    None,
}

/// A token to append to the vocabulary, before an ID is assigned.
///
/// Mirrors the fields of an `added_tokens` entry in `tokenizer.json` minus its
/// ID, which [`Tokenizer::add_tokens`] assigns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NewToken {
    /// Literal text content that triggers the token.
    pub content: String,
    /// Whether the token should only match as a whole word. Recorded but not
    /// yet honored by matching, as for tokens declared in `tokenizer.json`.
    pub single_word: bool,
    /// Absorb adjacent whitespace on the left when matching.
    pub lstrip: bool,
    /// Absorb adjacent whitespace on the right when matching.
    pub rstrip: bool,
    /// Whether the content is matched against normalized text. Recorded but not
    /// yet honored by matching, as for tokens declared in `tokenizer.json`.
    pub normalized: bool,
    /// Whether this is a "special" token (e.g. BOS/EOS). Special tokens are
    /// dropped by `skip_special_tokens` decodes and by
    /// [`AddedTokenPolicy::SkipSpecial`] encodes.
    pub special: bool,
}

impl NewToken {
    /// An ordinary added token, matched literally.
    ///
    /// `normalized` follows upstream's `AddedToken::from`, which sets it for
    /// ordinary tokens and clears it for special ones. It matters because
    /// re-adding a token is a no-op only when every field matches.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            normalized: true,
            ..Self::default()
        }
    }

    /// A special added token (BOS/EOS/control markers).
    pub fn special(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            special: true,
            ..Self::default()
        }
    }
}

impl<'a> EncodeSegment<'a> {
    /// A trusted segment whose special tokens are recognized.
    pub fn special(text: &'a str) -> Self {
        Self {
            text,
            allow_special: true,
        }
    }

    /// An untrusted segment encoded as ordinary content.
    pub fn ordinary(text: &'a str) -> Self {
        Self {
            text,
            allow_special: false,
        }
    }
}

/// An LLM tokenizer backed by `tokenizer.json`.
pub struct Tokenizer {
    added_tokens: Option<AddedTokens>,
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    model: Model,
    post_processor: Option<PostProcessor>,
    decoder: Option<Decoder>,
    /// When the pre-tokenizer is `Sequence([Split, ByteLevel(bulk)])`,
    /// we store a Split-only pre-tokenizer and fuse ByteLevel into BPE.
    split_only: Option<PreTokenizer>,
    /// Optional whole-input encoding cache; `None` (off) unless enabled.
    input_cache: Option<Mutex<InputCache>>,
    /// Whether to run vocab-aware (unbridgeable-bigram) splitting on encode.
    /// True for metaspace models (e.g. Gemma) whose pre-tokenizer is a no-op
    /// after normalization; false for ByteLevel models, whose regex `Split`
    /// already produces word-level chunks. The pass is output-preserving, so
    /// this flag only affects performance, never correctness.
    needs_vocab_splitting: bool,
}

impl Tokenizer {
    /// Build the pipeline steps from a parsed JSON config.
    fn build(json: TokenizerJson) -> Result<Self, Error> {
        Self::build_with_options(json, None, TokenizerOptions::default())
    }

    fn build_with_options(
        mut json: TokenizerJson,
        tokenizer_config: Option<TokenizerConfig>,
        options: TokenizerOptions,
    ) -> Result<Self, Error> {
        // Merge added tokens declared only in `tokenizer_config.json`
        // (`added_tokens_decoder`) — e.g. Qwen2-VL's `<|image_pad|>`, which is
        // absent from `tokenizer.json`'s `added_tokens` array.
        if let Some(tokenizer_config) = tokenizer_config {
            Self::merge_added_tokens(
                &mut json.added_tokens,
                tokenizer_config
                    .added_token_configs()
                    .map_err(Error::Model)?,
            )?;
        }
        let added_tokens = AddedTokens::from_configs(&json.added_tokens).map_err(Error::Model)?;
        let normalizer = json.normalizer.map(Normalizer::from_config).transpose()?;
        let pre_tokenizer = json
            .pre_tokenizer
            .map(|config| PreTokenizer::from_config_with_limits(config, options.pcre2_limits))
            .transpose()?;
        let model = Model::from_config(json.model).map_err(Error::Model)?;
        let post_processor = json
            .post_processor
            .map(PostProcessor::from_config)
            .transpose()?;
        let decoder = json.decoder.map(Decoder::from_config).transpose()?;

        // Detect Sequence([Split, ByteLevel(bulk)]) for fused byte-level+BPE.
        let split_only = Self::detect_fused_byte_level(&pre_tokenizer);

        // ByteLevel pipelines already chunk at word boundaries via their regex
        // Split, so vocab-aware splitting adds cost without benefit. Only run it
        // when no ByteLevel step is present (metaspace models like Gemma).
        let needs_vocab_splitting = !Self::pre_tokenizer_contains_byte_level(&pre_tokenizer);

        Ok(Self {
            added_tokens,
            normalizer,
            pre_tokenizer,
            model,
            post_processor,
            decoder,
            split_only,
            input_cache: input_cache_from_env(),
            needs_vocab_splitting,
        })
    }

    /// Recursively check whether a pre-tokenizer pipeline contains a `ByteLevel`
    /// step (including inside a `Sequence`).
    fn pre_tokenizer_contains_byte_level(pt: &Option<PreTokenizer>) -> bool {
        fn contains(pt: &PreTokenizer) -> bool {
            match pt {
                PreTokenizer::ByteLevel(_) => true,
                PreTokenizer::Split(_) => false,
                PreTokenizer::Sequence(steps) => steps.iter().any(contains),
            }
        }
        pt.as_ref().is_some_and(contains)
    }

    /// If `pt` is `Sequence([Split, ByteLevel(bulk)])`, return a Split-only
    /// pre-tokenizer for fused mode.
    fn detect_fused_byte_level(pt: &Option<PreTokenizer>) -> Option<PreTokenizer> {
        let PreTokenizer::Sequence(steps) = pt.as_ref()? else {
            return None;
        };
        if steps.len() != 2 {
            return None;
        }
        let is_split = matches!(&steps[0], PreTokenizer::Split(_));
        let is_bulk_bl = matches!(&steps[1], PreTokenizer::ByteLevel(bl) if bl.is_bulk_only());
        if is_split && is_bulk_bl {
            Some(steps[0].clone())
        } else {
            None
        }
    }

    /// Create a tokenizer from a raw JSON value for `tokenizer.json`.
    pub fn from_json(json: Value) -> Result<Self, Error> {
        let json: TokenizerJson = serde_json::from_value(json)?;
        Self::build(json)
    }

    /// Create a tokenizer from a raw JSON value for `tokenizer.json` with construction options.
    pub fn from_json_with_options(json: Value, options: TokenizerOptions) -> Result<Self, Error> {
        let json: TokenizerJson = serde_json::from_value(json)?;
        Self::build_with_options(json, None, options)
    }

    /// Create a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let json: TokenizerJson = serde_json::from_str(&fs::read_to_string(path)?)?;
        let config_path = path.with_file_name("tokenizer_config.json");
        let tokenizer_config = if config_path.exists() {
            Some(serde_json::from_str(&fs::read_to_string(config_path)?)?)
        } else {
            None
        };
        Self::build_with_options(json, tokenizer_config, TokenizerOptions::default())
    }

    /// Create a tokenizer from tiktoken mergeable ranks (`token_bytes -> rank`).
    ///
    /// A tiktoken model carries only the byte-level BPE ranks; the
    /// pre-tokenization regex and special tokens are supplied via `config`
    /// (see [`TiktokenConfig`]). The resulting pipeline is: split special
    /// tokens → split on the regex → fused byte-level BPE → ByteLevel decode.
    ///
    /// Each special token becomes a literal (unnormalized, non-stripping) added
    /// token marked special. To carry per-token `lstrip` / `rstrip` / `special`
    /// flags through instead — as declared in a model's `added_tokens_decoder` —
    /// use [`Self::from_tiktoken_ranks_with_added_tokens`].
    pub fn from_tiktoken_ranks(
        ranks: &[(Vec<u8>, u32)],
        config: TiktokenConfig,
    ) -> Result<Self, Error> {
        let added_configs: Vec<AddedTokenConfig> = config
            .special_tokens
            .into_iter()
            .map(|(content, id)| AddedTokenConfig {
                id,
                content,
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            })
            .collect();
        Self::from_tiktoken_ranks_with_added_tokens(ranks, &config.pattern, &added_configs)
    }

    /// Like [`Self::from_tiktoken_ranks`], but takes fully-specified added
    /// tokens instead of `(content, id)` pairs.
    ///
    /// Use this when the model declares per-token flags — typically via
    /// `tokenizer_config.json`'s `added_tokens_decoder`, which
    /// [`TokenizerConfig::added_token_configs`] converts. The flags are not
    /// cosmetic: `lstrip` / `rstrip` make a match absorb adjacent whitespace and
    /// so change the resulting token ids, while `special` controls whether
    /// decoding can skip the token.
    ///
    /// Takes no [`TokenizerOptions`] because its only member, the PCRE2 limits,
    /// cannot apply here: Kimi-family patterns use character-class intersection
    /// (`&&`), which PCRE2 cannot compile, so pre-tokenization runs on
    /// `fancy-regex` and there is no PCRE2 matcher to bound. Threading limits in
    /// would make `Split` reject the pattern outright
    /// (`try_compile_pcre2_regexes` returns `Unsupported` when limits are set on
    /// an intersection pattern), turning an inert knob into a load failure.
    pub fn from_tiktoken_ranks_with_added_tokens(
        ranks: &[(Vec<u8>, u32)],
        pattern: &str,
        added_configs: &[AddedTokenConfig],
    ) -> Result<Self, Error> {
        let bpe = models::bpe::Bpe::from_tiktoken_ranks(ranks).map_err(Error::Model)?;
        let model = Model::Bpe(bpe);

        // Sequence([Split(pat_str, Isolated), ByteLevel(bulk)]) — the shape the
        // fused byte-level path is detected from. The Split reproduces
        // tiktoken's `regex.findall`; ByteLevel(bulk) marks byte-level BPE.
        let split =
            Split::from_config(&serde_json::json!({ "Regex": pattern }), "Isolated", false)?;
        let byte_level = ByteLevel::from_config(false, false, false)?;
        let pre_tokenizer = Some(PreTokenizer::Sequence(vec![
            PreTokenizer::Split(split),
            PreTokenizer::ByteLevel(byte_level),
        ]));
        let split_only = Self::detect_fused_byte_level(&pre_tokenizer);

        let added_tokens = AddedTokens::from_configs(added_configs).map_err(Error::Model)?;

        let decoder = Some(Decoder::from_config(DecoderConfig::ByteLevel)?);

        // tiktoken pipelines are ByteLevel, so vocab-aware splitting is a no-op
        // cost — the regex `Split` already chunks at word boundaries.
        let needs_vocab_splitting = !Self::pre_tokenizer_contains_byte_level(&pre_tokenizer);

        Ok(Self {
            added_tokens,
            normalizer: None,
            pre_tokenizer,
            model,
            post_processor: None,
            decoder,
            split_only,
            input_cache: input_cache_from_env(),
            needs_vocab_splitting,
        })
    }

    /// Create a tokenizer from the contents of a tiktoken model file
    /// (`base64(token_bytes) rank` lines). See [`Self::from_tiktoken_ranks`].
    pub fn from_tiktoken_str(contents: &str, config: TiktokenConfig) -> Result<Self, Error> {
        let ranks = tiktoken::parse_tiktoken_model(contents)?;
        Self::from_tiktoken_ranks(&ranks, config)
    }

    /// Create a tokenizer from a tiktoken model file on disk (e.g.
    /// `tiktoken.model`). See [`Self::from_tiktoken_ranks`].
    pub fn from_tiktoken_file(path: &Path, config: TiktokenConfig) -> Result<Self, Error> {
        let contents = fs::read_to_string(path)?;
        Self::from_tiktoken_str(&contents, config)
    }

    /// Create a tokenizer from a `tokenizer.json` file with construction options.
    pub fn from_file_with_options(path: &Path, options: TokenizerOptions) -> Result<Self, Error> {
        let json: TokenizerJson = serde_json::from_str(&fs::read_to_string(path)?)?;
        let config_path = path.with_file_name("tokenizer_config.json");
        let tokenizer_config = if config_path.exists() {
            Some(serde_json::from_str(&fs::read_to_string(config_path)?)?)
        } else {
            None
        };
        Self::build_with_options(json, tokenizer_config, options)
    }

    /// Download `tokenizer.json` from HuggingFace Hub for the given model (e.g.
    /// `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    ///
    /// Authentication is resolved automatically from `~/.cache/huggingface/token`
    /// (set via `huggingface-cli login`).  To supply a token explicitly, use
    /// [`Self::from_model_with_token`].
    #[cfg(feature = "hf-hub")]
    pub fn from_model(model: &str) -> Result<Self, Error> {
        Self::from_model_with_token(model, None)
    }

    /// Like [`Self::from_model`] but accepts construction options.
    #[cfg(feature = "hf-hub")]
    pub fn from_model_with_options(model: &str, options: TokenizerOptions) -> Result<Self, Error> {
        Self::from_model_with_token_and_options(model, None, options)
    }

    /// Like [`Self::from_model`] but accepts an explicit HuggingFace token,
    /// overriding the credential cache.  Pass `None` to use the credential
    /// cache (`~/.cache/huggingface/token`, set via `huggingface-cli login`).
    #[cfg(feature = "hf-hub")]
    pub fn from_model_with_token(model: &str, token: Option<&str>) -> Result<Self, Error> {
        hf_hub_support::from_model_with_token(model, token)
    }

    /// Like [`Self::from_model_with_token`] but accepts construction options.
    #[cfg(feature = "hf-hub")]
    pub fn from_model_with_token_and_options(
        model: &str,
        token: Option<&str>,
        options: TokenizerOptions,
    ) -> Result<Self, Error> {
        hf_hub_support::from_model_with_token_and_options(model, token, options)
    }

    /// Download `tokenizer.json` and return its raw content without building
    /// the tokenizer.  Used by the Python layer to extract fields (such as
    /// `post_processor`) before handing the JSON off to [`Self::from_json`].
    #[cfg(feature = "hf-hub")]
    pub fn download_tokenizer_json(model: &str) -> Result<String, Error> {
        hf_hub_support::download_tokenizer_json(model)
    }

    /// Return the normalizer, if any.
    pub fn normalizer(&self) -> Option<&Normalizer> {
        self.normalizer.as_ref()
    }

    fn merge_added_tokens(
        added_tokens: &mut Vec<AddedTokenConfig>,
        extra_tokens: Vec<AddedTokenConfig>,
    ) -> Result<(), Error> {
        let mut ids = HashMap::with_capacity(added_tokens.len());
        let mut contents = HashMap::with_capacity(added_tokens.len());
        for (index, token) in added_tokens.iter().enumerate() {
            ids.insert(token.id, index);
            contents.insert(token.content.clone(), token.id);
        }

        for token in extra_tokens {
            match (
                ids.get(&token.id).copied(),
                contents.get(&token.content).copied(),
            ) {
                (Some(_), Some(existing_id)) if existing_id == token.id => {
                    // Same id + content already present from `tokenizer.json`,
                    // which is authoritative. Field-level differences between
                    // the two files (e.g. `special`, `lstrip`) are benign, so
                    // keep the existing entry rather than rejecting the model.
                }
                (Some(index), _) => {
                    return Err(Error::Model(format!(
                        "added token id {} maps to both {:?} and {:?}",
                        token.id, added_tokens[index].content, token.content
                    )));
                }
                (_, Some(existing_id)) => {
                    return Err(Error::Model(format!(
                        "added token {:?} maps to both ids {} and {}",
                        token.content, existing_id, token.id
                    )));
                }
                (None, None) => {
                    let index = added_tokens.len();
                    ids.insert(token.id, index);
                    contents.insert(token.content.clone(), token.id);
                    added_tokens.push(token);
                }
            }
        }

        Ok(())
    }

    /// Return the pre-tokenizer, if any.
    pub fn pre_tokenizer(&self) -> Option<&PreTokenizer> {
        self.pre_tokenizer.as_ref()
    }

    /// Return the post-processor, if any.
    pub fn post_processor(&self) -> Option<&PostProcessor> {
        self.post_processor.as_ref()
    }

    /// Return the tokenization model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Return the compiled added-token set, if any.
    pub fn added_tokens(&self) -> Option<&AddedTokens> {
        self.added_tokens.as_ref()
    }

    /// The added-vocabulary entries, in declaration order.
    pub fn added_token_configs(&self) -> &[AddedTokenConfig] {
        self.added_tokens.as_ref().map_or(&[], AddedTokens::configs)
    }

    /// Extend the vocabulary with `tokens`, returning the entries created or
    /// changed, in the order given.
    ///
    /// IDs are assigned the way HuggingFace `tokenizers` assigns them, so a
    /// tokenizer extended here and one extended there agree:
    ///
    /// - A content already in the added vocabulary keeps its ID; only its flags
    ///   can change (this is how a token is promoted to `special`). An entry
    ///   that would not change at all is not reported.
    /// - A content the model already carries becomes an added token at that
    ///   same ID, leaving the vocabulary size untouched.
    /// - Anything else is appended above the vocabulary, contiguously.
    ///
    /// The last case is the one serving stacks depend on: a checkpoint whose
    /// embedding matrix is padded above the tokenizer's token count is squared
    /// up by appending placeholder tokens until [`Self::vocab_size`] matches the
    /// embedding count, after which every ID below it resolves.
    ///
    /// Empty contents are ignored, as they are upstream. Repeats within one
    /// call collapse, so a batch can never claim two IDs for one string.
    pub fn add_tokens(&mut self, tokens: &[NewToken]) -> Result<Vec<AddedTokenConfig>, Error> {
        let mut configs = self.added_token_configs().to_vec();
        let mut index_of: HashMap<String, usize> = configs
            .iter()
            .enumerate()
            .map(|(index, config)| (config.content.clone(), index))
            .collect();

        // Upstream appends above the model vocabulary, unless added IDs already
        // reach past it — then it continues from the highest one.
        let model_size = self.model.vocab_size() as u32;
        let mut next_id = match configs.iter().map(|config| config.id).max() {
            Some(max) if max >= model_size => max + 1,
            _ => model_size,
        };

        let mut changed = Vec::new();
        for token in tokens {
            if token.content.is_empty() {
                continue;
            }
            let entry = AddedTokenConfig {
                // Replaced below; every branch decides its own ID.
                id: 0,
                content: token.content.clone(),
                single_word: token.single_word,
                lstrip: token.lstrip,
                rstrip: token.rstrip,
                normalized: token.normalized,
                special: token.special,
            };

            match index_of.get(&token.content).copied() {
                Some(index) => {
                    let updated = AddedTokenConfig {
                        id: configs[index].id,
                        ..entry
                    };
                    if configs[index] == updated {
                        continue;
                    }
                    configs[index] = updated.clone();
                    changed.push(updated);
                }
                None => {
                    let id = self.model.token_to_id(&token.content).unwrap_or_else(|| {
                        let id = next_id;
                        next_id += 1;
                        id
                    });
                    let entry = AddedTokenConfig { id, ..entry };
                    index_of.insert(token.content.clone(), configs.len());
                    configs.push(entry.clone());
                    changed.push(entry);
                }
            }
        }

        if changed.is_empty() {
            return Ok(changed);
        }

        // Compile before committing: a rejected set must leave the tokenizer as
        // it was rather than half-extended.
        let added_tokens = AddedTokens::from_configs(&configs).map_err(Error::Model)?;
        self.added_tokens = added_tokens;

        // Cached encodings were produced against the previous vocabulary.
        if let Some(cache) = &self.input_cache {
            cache.lock().unwrap().clear();
        }

        Ok(changed)
    }

    /// Return the decoder, if any.
    pub fn decoder(&self) -> Option<&Decoder> {
        self.decoder.as_ref()
    }

    // ── Encoding ─────────────────────────────────────────────────────

    /// Run the full encoding pipeline: split added tokens, normalize,
    /// pre-tokenize, tokenize and post-process the input string.
    pub fn encode(&self, input: &str) -> Result<Vec<u32>, Error> {
        self.encode_with_special_tokens(input, false)
    }

    /// Enable the opt-in prefix cache (see [`InputCache`]) retaining up to
    /// `capacity` recent encodings. Reuses the leading token ids of inputs that
    /// share a byte prefix (shared system prompts / long shared contexts), and
    /// the whole encoding of an exact repeat. Leave it off for wholly-unique
    /// traffic.
    pub fn enable_input_cache(&mut self, capacity: usize) {
        self.input_cache = Some(Mutex::new(InputCache::new(capacity)));
    }

    /// Run the full encoding pipeline with control over special token insertion.
    ///
    /// When `add_special_tokens` is true, the post-processor inserts special
    /// tokens (e.g. BOS/EOS) as configured in the tokenizer's post-processor.
    ///
    /// The prefix cache (if enabled) is applied inside the scanner fast path,
    /// which is where shared-prefix inputs are tokenized.
    pub fn encode_with_special_tokens(
        &self,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, Error> {
        self.encode_inner(input, add_special_tokens, AddedTokenPolicy::All)
    }

    /// Encode through the base tokenizer pipeline without recognizing added
    /// vocabulary entries.
    ///
    /// Equivalent to [`Self::encode_with_special_tokens`] with
    /// `add_special_tokens = false`, except that every added-token matcher is
    /// bypassed. Normalization, pre-tokenization, model tokenization, and
    /// post-processing are preserved.
    pub fn encode_ordinary(&self, input: &str) -> Result<Vec<u32>, Error> {
        self.encode_inner(input, false, AddedTokenPolicy::None)
    }

    /// Run the full encoding pipeline under an explicit [`AddedTokenPolicy`].
    ///
    /// [`AddedTokenPolicy::SkipSpecial`] is the one the other entry points do
    /// not cover: control-token strings are encoded as ordinary text while the
    /// rest of the added vocabulary still matches.
    pub fn encode_with_policy(
        &self,
        input: &str,
        add_special_tokens: bool,
        policy: AddedTokenPolicy,
    ) -> Result<Vec<u32>, Error> {
        self.encode_inner(input, add_special_tokens, policy)
    }

    /// Encode a pre-segmented input, concatenating the token ids of each
    /// segment in order.
    ///
    /// This mirrors legacy tiktoken / Dynamo segmented encoding: each
    /// [`EncodeSegment`] is tokenized **independently** and its trust boundary
    /// is honored — special tokens are recognized only in segments with
    /// `allow_special = true` (see [`EncodeSegment`]). Segments are never
    /// flattened into a single string first, so the trust boundary between
    /// control tokens and untrusted content is preserved, and no BPE merge
    /// crosses a segment boundary.
    ///
    /// No post-processor special tokens (BOS/EOS) are inserted — the caller's
    /// segments are expected to already carry the full rendered sequence.
    pub fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Vec<u32>, Error> {
        fn policy(seg: &EncodeSegment<'_>) -> AddedTokenPolicy {
            if seg.allow_special {
                AddedTokenPolicy::All
            } else {
                AddedTokenPolicy::None
            }
        }

        // Single-segment shortcut avoids a second allocation + copy.
        if let [seg] = segments {
            return self.encode_inner(seg.text, false, policy(seg));
        }
        let mut ids = Vec::new();
        for seg in segments {
            let seg_ids = self.encode_inner(seg.text, false, policy(seg))?;
            if ids.is_empty() {
                ids = seg_ids;
            } else {
                ids.extend_from_slice(&seg_ids);
            }
        }
        Ok(ids)
    }

    fn encode_inner(
        &self,
        input: &str,
        add_special_tokens: bool,
        policy: AddedTokenPolicy,
    ) -> Result<Vec<u32>, Error> {
        if input.is_empty() {
            return if add_special_tokens {
                Ok(self.post_process(Vec::new(), true))
            } else {
                Ok(Vec::new())
            };
        }

        // 1. Normalize the input, optionally recognizing added vocabulary.
        let mut pts = match policy {
            AddedTokenPolicy::All => self.build_pre_tokenized_with(input, false),
            AddedTokenPolicy::SkipSpecial => self.build_pre_tokenized_with(input, true),
            AddedTokenPolicy::None => self.build_pre_tokenized_ordinary(input),
        };

        // Fused path: run only Split, then batch-tokenize with inline ByteLevel.
        if let Some(ref split) = self.split_only {
            // Scanner fast path: for a recognized tiktoken pattern with a single
            // plain-text segment (no added/special tokens matched), skip the
            // regex + `Split` materialization — scan pretoken ranges directly
            // and BPE over them. Falls back to the regex path otherwise.
            if pts.splits().len() == 1
                && pts.splits()[0].token_id.is_none()
                && pts.buffer().len() <= u32::MAX as usize
                && let PreTokenizer::Split(inner) = split
                && let Some(kind) = inner.scan_kind()
            {
                let buffer = pts.buffer();
                // Fused scan+BPE of a plain-text segment: one pass, split at
                // newline boundaries, each segment scanned and BPE'd inline
                // while hot in cache — no range list is materialized.
                let scan_seg = |seg: &str| {
                    let mut ids = Vec::with_capacity(seg.len() / 3 + 1);
                    self.model.tokenize_scanned_segment(kind, seg, &mut ids)?;
                    Ok(ids)
                };

                // Prefix cache: reuse the leading ids shared with a cached input
                // and tokenize only the tail, or reuse an exact repeat wholesale.
                if let Some(cache) = &self.input_cache {
                    let plan = cache.lock().unwrap().reuse_plan(buffer.as_bytes());
                    if let Some(r) = plan {
                        let mut ids =
                            Vec::with_capacity(r.tokens + (buffer.len() - r.tail_start) / 3 + 1);
                        ids.extend_from_slice(&r.core_ids[..r.tokens]);
                        if r.tail_start < buffer.len() {
                            let tail = crate::pre_tokenized::tokenize_scanned(
                                &buffer[r.tail_start..],
                                kind,
                                scan_seg,
                            )
                            .map_err(Error::Model)?;
                            ids.extend_from_slice(&tail);
                        }
                        return Ok(self.post_process(ids, add_special_tokens));
                    }
                    // Miss: full encode recording reuse boundaries, then cache it.
                    let scan_seg_rec = |seg: &str| {
                        let mut ids = Vec::with_capacity(seg.len() / 3 + 1);
                        let mut b = Vec::new();
                        self.model
                            .tokenize_scanned_segment_rec(kind, seg, &mut ids, &mut b)?;
                        Ok((ids, b))
                    };
                    let (ids, bounds) = crate::pre_tokenized::tokenize_scanned_with_bounds(
                        buffer,
                        kind,
                        scan_seg_rec,
                    )
                    .map_err(Error::Model)?;
                    cache
                        .lock()
                        .unwrap()
                        .insert(buffer.as_bytes(), &ids, bounds);
                    return Ok(self.post_process(ids, add_special_tokens));
                }

                let ids = crate::pre_tokenized::tokenize_scanned(buffer, kind, scan_seg)
                    .map_err(Error::Model)?;
                return Ok(self.post_process(ids, add_special_tokens));
            }

            split.pre_tokenize(&mut pts)?;
            let ids = pts
                .tokenize_batched(|buf, splits, out| {
                    self.model.tokenize_batch_fused(buf, splits, out)
                })
                .map_err(Error::Model)?;
            return Ok(self.post_process(ids, add_special_tokens));
        }

        // 2. Pre-tokenize (refine splits in place).
        if let Some(ref pt) = self.pre_tokenizer {
            pt.pre_tokenize(&mut pts)?;
        }

        // 2b. Break each text split at unbridgeable byte-pair boundaries.
        //     Split at positions where adjacent bytes never appear together in
        //     any vocab token. This is provably output-preserving and provides
        //     fine-grained word-level chunking for models that don't use
        //     ByteLevel (whose regex Split already chunks at word boundaries).
        if self.needs_vocab_splitting
            && let Some(table) = self.model.bigram_bridge_table()
        {
            split_on_unbridgeable_bigrams(&mut pts, table);
        }

        // 3. Tokenize each text split with the model.
        let ids = pts
            .tokenize(|text, out| self.model.tokenize_into(text, out))
            .map_err(Error::Model)?;

        // 4. Post-process.
        Ok(self.post_process(ids, add_special_tokens))
    }

    /// Encode a batch of inputs.
    pub fn encode_batch<S: AsRef<str> + Sync>(
        &self,
        inputs: &[S],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>, Error> {
        inputs
            .par_iter()
            .map(|input| self.encode_with_special_tokens(input.as_ref(), add_special_tokens))
            .collect()
    }

    /// Replace the post-processor.  Called when transformers dynamically
    /// updates the post-processor (e.g. for `add_bos_token=True`).
    pub fn set_post_processor(&mut self, pp: Option<PostProcessor>) {
        self.post_processor = pp;
    }

    /// Replace the normalizer.
    pub fn set_normalizer(&mut self, normalizer: Option<Normalizer>) {
        self.normalizer = normalizer;
    }

    pub fn post_process(&self, ids: Vec<u32>, add_special_tokens: bool) -> Vec<u32> {
        match &self.post_processor {
            Some(pp) => pp.post_process_single(ids, add_special_tokens),
            None => ids,
        }
    }

    // ── Decoding ─────────────────────────────────────────────────────

    /// Decode token IDs back into text.
    ///
    /// If `skip_special_tokens` is true, added tokens marked as special
    /// are omitted from the output.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            if skip_special_tokens
                && let Some(ref at) = self.added_tokens
                && at.is_special(id)
            {
                continue;
            }
            // Match HuggingFace behavior: silently skip unknown IDs (e.g.
            // models like Qwen3-0.6B-FP8 emit IDs in the gap between
            // tokenizer.json's vocab and the embedding matrix). Erroring
            // here would kill streaming generation on a single bad token.
            if let Some(token_str) = self.id_to_token(id) {
                tokens.push(token_str.to_string());
            }
        }

        match &self.decoder {
            Some(dec) => dec.decode(tokens).map_err(Error::Decoder),
            None => Ok(tokens.join("")),
        }
    }

    /// Decode a sequence of token strings back into text.
    ///
    /// Applies the decoder pipeline (e.g. ByteLevel → convert "Ġ" back to " ")
    /// without going through the ID→string lookup.  When no decoder is
    /// configured the tokens are concatenated with no separator.
    pub fn decode_tokens(&self, tokens: Vec<String>) -> Result<String, Error> {
        match &self.decoder {
            Some(dec) => dec.decode(tokens).map_err(Error::Decoder),
            None => Ok(tokens.join("")),
        }
    }

    /// Decode a batch of token ID sequences.
    pub fn decode_batch(
        &self,
        sentences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, Error> {
        sentences
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    // ── Vocabulary access ────────────────────────────────────────────

    /// Look up the string for a token ID, checking added tokens first,
    /// then the model vocabulary.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        if let Some(ref at) = self.added_tokens
            && let Some(s) = at.id_to_token(id)
        {
            return Some(s);
        }
        self.model.id_to_token(id)
    }

    /// Look up the token ID for a string.
    ///
    /// Added tokens are checked first (they shadow any BPE model entry with
    /// the same string), then the BPE model vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(ref at) = self.added_tokens
            && let Some(id) = at.token_to_id(token)
        {
            return Some(id);
        }
        self.model.token_to_id(token)
    }

    /// Return the vocabulary size.
    ///
    /// A vocabulary is a token -> ID map, so its size is the number of *distinct
    /// token strings*, which is how HuggingFace `tokenizers` computes it
    /// (`get_vocab_size(true) == get_vocab(true).len()`). Adding the two counts
    /// instead overcounts whenever `added_tokens` restates a string that is
    /// already present, in either of two ways:
    ///
    /// - the string is also in `model.vocab` (e.g. a checkpoint that lists
    ///   BOS/EOS/PAD in both places), or
    /// - two `added_tokens` entries share a content under different IDs.
    ///
    /// Both collapse in a real vocabulary, so both are deduplicated here:
    /// [`AddedTokens::contents`] yields distinct strings, and the model lookup
    /// drops the ones the model already provides. Overcounting is not cosmetic —
    /// callers size embedding tables from this and index every ID below it, so an
    /// inflated count points at IDs that do not exist.
    pub fn vocab_size(&self) -> usize {
        let model_size = self.model.vocab_size();
        let added_size = self.added_tokens.as_ref().map_or(0, |at| {
            at.contents()
                .filter(|content| self.model.token_to_id(content).is_none())
                .count()
        });
        model_size + added_size
    }

    /// Return whether this token ID is marked special in the added-token set.
    pub fn is_special_token(&self, id: u32) -> bool {
        self.added_tokens
            .as_ref()
            .is_some_and(|added_tokens| added_tokens.is_special(id))
    }

    // ── Internal helpers ─────────────────────────────────────────────

    /// Build a [`PreTokenizedString`] by splitting on added tokens and
    /// normalizing text segments into a single contiguous buffer.
    pub fn build_pre_tokenized(&self, input: &str) -> PreTokenizedString {
        self.build_pre_tokenized_with(input, false)
    }

    /// [`Self::build_pre_tokenized`], optionally leaving special tokens as
    /// ordinary text (see [`AddedTokens::split_with`]).
    pub fn build_pre_tokenized_with(&self, input: &str, skip_special: bool) -> PreTokenizedString {
        let segments = match &self.added_tokens {
            Some(at) => at.split_with(input, skip_special),
            None => vec![Segment::Text(input)],
        };

        // Fast path: if there's exactly one Text segment (no added token matches)
        // and normalization returns Cow::Borrowed, we just need a string copy.
        if segments.len() == 1
            && let Segment::Text(text) = segments[0]
        {
            return self.build_pre_tokenized_ordinary(text);
        }

        let mut buffer = String::with_capacity(input.len());
        let mut splits = Vec::new();

        for seg in &segments {
            match seg {
                Segment::Token(id) => {
                    let start = buffer.len();
                    splits.push(PtSplit {
                        range: start..start,
                        token_id: Some(*id),
                    });
                }
                Segment::Text(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    let normalized = match &self.normalizer {
                        Some(n) => n.normalize(text),
                        None => std::borrow::Cow::Borrowed(*text),
                    };
                    let start = buffer.len();
                    buffer.push_str(&normalized);
                    let end = buffer.len();
                    splits.push(PtSplit {
                        range: start..end,
                        token_id: None,
                    });
                }
            }
        }

        PreTokenizedString::new(buffer, splits)
    }

    /// Normalize one input as a single text span, bypassing added vocabulary.
    fn build_pre_tokenized_ordinary(&self, input: &str) -> PreTokenizedString {
        let normalized = match &self.normalizer {
            Some(normalizer) => normalizer.normalize(input),
            None => std::borrow::Cow::Borrowed(input),
        };
        match normalized {
            std::borrow::Cow::Borrowed(_) => PreTokenizedString::from_text(input),
            std::borrow::Cow::Owned(buffer) => {
                let len = buffer.len();
                PreTokenizedString::new(
                    buffer,
                    vec![PtSplit {
                        range: 0..len,
                        token_id: None,
                    }],
                )
            }
        }
    }
}

/// Split each text chunk at unbridgeable byte-pair boundaries using the
/// vocab-derived bigram bridge table.
///
/// A byte pair (prev, cur) is "unbridgeable" if no vocabulary token contains
/// that adjacent byte sequence. Splitting at such boundaries is provably
/// output-preserving: any BPE merge that spans the boundary would produce a
/// token containing that byte pair, which cannot exist in the vocabulary.
///
/// This generalizes newline splitting and enables fine-grained word-level
/// chunking even in metaspace tokenizers like Gemma, where the pre-tokenizer
/// is effectively a no-op after normalization.
fn split_on_unbridgeable_bigrams(
    pts: &mut PreTokenizedString,
    bigram_table: &models::bpe::BigramBridgeTable,
) {
    let bytes = pts.buffer().as_bytes();
    let mut new_splits = Vec::with_capacity(pts.splits().len() * 2);

    for split in pts.splits() {
        if split.token_id.is_some() || split.range.is_empty() {
            new_splits.push(split.clone());
            continue;
        }

        let end = split.range.end;
        let mut start = split.range.start;

        for i in (start + 1)..end {
            let prev = bytes[i - 1];
            let cur = bytes[i];

            // Split here if:
            // 1. This byte pair never appears in vocab, AND
            // 2. Position i is a UTF-8 char boundary (cur is not a continuation byte)
            if !bigram_table.is_bridgeable(prev, cur) && (cur & 0xC0) != 0x80 {
                new_splits.push(PtSplit {
                    range: start..i,
                    token_id: None,
                });
                start = i;
            }
        }

        // Push the final segment
        new_splits.push(PtSplit {
            range: start..end,
            token_id: None,
        });
    }

    pts.refine_splits(new_splits);
}

// ---------------------------------------------------------------------------
// Streaming decode
// ---------------------------------------------------------------------------

/// Stateful incremental decoder.
///
/// Wraps the sliding-window state needed by [`decode_stream_step`] so callers
/// don't have to manage `ids`, `prefix`, and `prefix_index` themselves.
pub struct DecodeStream {
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl DecodeStream {
    pub fn new(ids: Vec<u32>, skip_special_tokens: bool) -> Self {
        Self {
            skip_special_tokens,
            ids,
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    pub fn step(
        &mut self,
        tokenizer: &Tokenizer,
        token_ids: Vec<u32>,
    ) -> Result<Option<String>, String> {
        decode_stream_step(
            tokenizer,
            token_ids,
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
    }
}

/// Advance an incremental decode stream by one or more token IDs.
///
/// Maintains a sliding window in `ids` and a `prefix` string to subtract,
/// emitting text chunks as soon as enough context is available.
/// Incomplete UTF-8 (signalled by U+FFFD in the decoder output) is held back
/// until a subsequent token resolves it.
///
/// # Arguments
/// * `token_ids` — new token IDs to append
/// * `skip_special_tokens` — whether to omit special tokens from the output
/// * `ids` — mutable buffer of all IDs decoded so far (updated in place)
/// * `prefix` — previously returned text, subtracted to yield the next chunk
/// * `prefix_index` — index in `ids` where the current prefix window starts
///
/// # Returns
/// `Ok(Some(chunk))` when new text is available, `Ok(None)` when more tokens
/// are needed, `Err(msg)` if the decoder produces output inconsistent with the
/// stored prefix (should be treated as a stream-reset signal).
pub fn decode_stream_step(
    tokenizer: &Tokenizer,
    token_ids: Vec<u32>,
    skip_special_tokens: bool,
    ids: &mut Vec<u32>,
    prefix: &mut String,
    prefix_index: &mut usize,
) -> Result<Option<String>, String> {
    const REPLACEMENT: char = '\u{FFFD}';

    // If the prefix is empty but we already have buffered IDs (e.g. seeded
    // with prompt tokens), prime the prefix before adding the new token.
    if prefix.is_empty() && !ids.is_empty() {
        let s = tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| e.to_string())?;
        if !s.ends_with(REPLACEMENT) {
            *prefix = s;
            *prefix_index = ids.len();
        }
    }

    ids.extend(token_ids);

    let string = tokenizer
        .decode(ids, skip_special_tokens)
        .map_err(|e| e.to_string())?;

    if string.len() > prefix.len() && !string.ends_with(REPLACEMENT) {
        if !string.starts_with(prefix.as_str()) {
            return Err(format!(
                "Invalid prefix encountered while decoding stream. \
                 Expected prefix: '{}', Actual string: '{}'",
                prefix, string,
            ));
        }
        let new_text = string[prefix.len()..].to_string();
        let new_prefix_index = ids.len() - *prefix_index;
        *ids = ids.drain(*prefix_index..).collect();
        *prefix = tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| e.to_string())?;
        *prefix_index = new_prefix_index;
        Ok(Some(new_text))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod local_tests {
    use serde_json::json;

    use super::*;

    // ── Error::is_not_found, feature-independent arms ───────────────────────
    //
    // These live here rather than in `mod tests` because that module is gated on
    // `feature = "hf-hub"`. Without it the `Hub` arm is compiled out and `Io` is
    // the only live arm, so gating its tests would leave the one reachable branch
    // untested in exactly the build where it matters.

    #[test]
    fn is_not_found_detects_io_not_found() {
        let err = Error::Io(std::io::Error::from(std::io::ErrorKind::NotFound));
        assert!(err.is_not_found());
    }

    #[test]
    fn is_not_found_rejects_other_io_and_error_kinds() {
        let denied = Error::Io(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        assert!(!denied.is_not_found());
        assert!(!Error::Model("boom".into()).is_not_found());
        assert!(!Error::Tiktoken("boom".into()).is_not_found());
    }

    #[test]
    fn from_json_with_options_propagates_pcre2_limits() {
        let tokenizer = Tokenizer::from_json_with_options(
            json!({
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0, "!": 1},
                    "merges": []
                },
                "pre_tokenizer": {
                    "type": "Split",
                    "pattern": {"Regex": "^(a+)+$"},
                    "behavior": "Isolated",
                    "invert": false
                }
            }),
            TokenizerOptions {
                pcre2_limits: Pcre2Limits {
                    match_limit: Some(1),
                    ..Default::default()
                },
            },
        )
        .unwrap();

        let err = tokenizer.encode("aaaaaaaaaaaaaaaa!").unwrap_err();
        assert!(
            err.to_string().contains("match limit"),
            "expected match limit error, got {err}"
        );
    }

    fn vocab_size_of(model_vocab: Value, added_tokens: Value) -> usize {
        Tokenizer::from_json(json!({
            "model": {"type": "BPE", "vocab": model_vocab, "merges": []},
            "added_tokens": added_tokens,
        }))
        .unwrap()
        .vocab_size()
    }

    /// Every way `added_tokens` can overlap an existing vocabulary entry.
    ///
    /// Expectations are the values HuggingFace `tokenizers` reports from
    /// `get_vocab_size(true)` for the same `tokenizer.json`, since a vocabulary
    /// counts distinct token strings.
    #[test]
    fn vocab_size_matches_huggingface_across_added_token_overlaps() {
        // No overlap: every added token is new.
        assert_eq!(
            vocab_size_of(
                json!({"a": 0, "b": 1}),
                json!([{"id": 2, "content": "<x>"}, {"id": 3, "content": "<y>"}]),
            ),
            4
        );

        // Added tokens restate strings the model already has. Seen in the wild on
        // checkpoints that list BOS/EOS/PAD in both `model.vocab` and
        // `added_tokens`.
        assert_eq!(
            vocab_size_of(
                json!({"<bos>": 0, "<eos>": 1, "a": 2, "b": 3}),
                json!([
                    {"id": 0, "content": "<bos>", "special": true},
                    {"id": 1, "content": "<eos>", "special": true},
                    {"id": 4, "content": "<extra>"}
                ]),
            ),
            5
        );

        // A gap between the model vocab and the added IDs does not inflate the
        // count: the size follows the strings, not the highest ID.
        assert_eq!(
            vocab_size_of(
                json!({"a": 0, "b": 1}),
                json!([{"id": 5, "content": "<far>"}])
            ),
            3
        );

        // A new string at an ID inside the model range still adds one entry.
        assert_eq!(
            vocab_size_of(
                json!({"a": 0, "b": 1, "c": 2}),
                json!([{"id": 1, "content": "<inside>"}]),
            ),
            4
        );
    }

    /// Two `added_tokens` entries sharing a content are rejected when the
    /// matcher is built, so the count never sees that overlap. Recorded because
    /// it is the reason the vocabulary count only has to deduplicate added
    /// strings against the *model*, and because HuggingFace `tokenizers` accepts
    /// such a file (reporting one entry for the shared string) — a separate
    /// divergence, and the safer direction of the two.
    #[test]
    fn duplicate_added_token_contents_are_rejected_at_construction() {
        // Tokenizer has no Debug impl, so unwrap_err() is unavailable.
        let result = Tokenizer::from_json(json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
            "added_tokens": [
                {"id": 2, "content": "<dup>"},
                {"id": 3, "content": "<dup>"}
            ],
        }));
        let err = match result {
            Ok(_) => panic!("expected duplicate added-token contents to be rejected"),
            Err(e) => e,
        };

        assert!(
            err.to_string().contains("DuplicatePattern"),
            "expected a duplicate-pattern error, got {err}"
        );
    }

    /// The count must not claim IDs that cannot be resolved, which is the
    /// property callers rely on when they size an embedding table from it and
    /// then index every ID below it.
    #[test]
    fn vocab_size_only_counts_resolvable_ids() {
        let tokenizer = Tokenizer::from_json(json!({
            "model": {
                "type": "BPE",
                "vocab": {"<bos>": 0, "<eos>": 1, "a": 2, "b": 3},
                "merges": []
            },
            "added_tokens": [
                {"id": 0, "content": "<bos>", "special": true},
                {"id": 1, "content": "<eos>", "special": true},
                {"id": 4, "content": "<extra>"}
            ]
        }))
        .unwrap();

        for id in 0..tokenizer.vocab_size() as u32 {
            assert!(
                tokenizer.id_to_token(id).is_some(),
                "id {id} is counted but has no token"
            );
        }
    }

    // ── add_tokens ──────────────────────────────────────────────────────

    /// A `tokenizer.json`, as text, so the same bytes can be loaded by both
    /// this crate and HuggingFace `tokenizers` for parity checks.
    ///
    /// The model vocabulary is the distinct characters of `chars`, one ID each,
    /// so any text over them encodes to one ID per character. Added tokens are
    /// appended above it in the order given.
    fn tokenizer_json(chars: &str, added: &[(&str, bool)]) -> String {
        let mut distinct: Vec<char> = chars.chars().collect();
        distinct.sort_unstable();
        distinct.dedup();
        let vocab: serde_json::Map<String, Value> = distinct
            .iter()
            .enumerate()
            .map(|(id, ch)| (ch.to_string(), Value::from(id)))
            .collect();
        let base = vocab.len() as u32;
        let added_tokens: Vec<Value> = added
            .iter()
            .enumerate()
            .map(|(offset, (content, special))| {
                json!({
                    "id": base + offset as u32,
                    "content": content,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": special,
                })
            })
            .collect();
        json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added_tokens,
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
        })
        .to_string()
    }

    fn tokenizer_of(chars: &str, added: &[(&str, bool)]) -> Tokenizer {
        Tokenizer::from_json(serde_json::from_str(&tokenizer_json(chars, added)).unwrap()).unwrap()
    }

    /// The scenario this exists for: a checkpoint whose embedding matrix is
    /// padded above the tokenizer's token count. The loader appends placeholder
    /// tokens until the two agree, then indexes every ID below the count.
    #[test]
    fn add_tokens_pads_a_tokenizer_up_to_an_embedding_count() {
        let mut tok = tokenizer_of("abc", &[("<|endoftext|>", true), ("<think>", false)]);
        assert_eq!(tok.vocab_size(), 5);

        let num_model_embeddings = 9;
        let placeholders: Vec<NewToken> = (0..num_model_embeddings - tok.vocab_size())
            .map(|i| NewToken::new(format!("<|padding_token_{i}|>")))
            .collect();
        let added = tok.add_tokens(&placeholders).unwrap();

        assert_eq!(added.len(), 4);
        assert_eq!(
            added.iter().map(|entry| entry.id).collect::<Vec<_>>(),
            vec![5, 6, 7, 8],
            "placeholders should land contiguously on the padded rows"
        );
        assert_eq!(tok.vocab_size(), num_model_embeddings);
        for id in 0..tok.vocab_size() as u32 {
            assert!(tok.id_to_token(id).is_some(), "id {id} has no token");
        }
        assert_eq!(tok.token_to_id("<|padding_token_0|>"), Some(5));
    }

    /// ID assignment has to agree with HuggingFace `tokenizers`, since callers
    /// swap the two backends under the same loader.
    #[test]
    fn add_tokens_matches_huggingface() {
        use std::str::FromStr;

        // Cases: brand-new contents, a repeat of one, a string the model
        // already carries, and a promotion of an existing added token.
        let batches: [&[(&str, bool)]; 4] = [
            &[("<|pad0|>", false), ("<|pad1|>", false)],
            &[("<|pad0|>", false)],
            &[("a", false)],
            &[("<think>", true)],
        ];

        let json = tokenizer_json("abc", &[("<|endoftext|>", true), ("<think>", false)]);
        let mut ours = Tokenizer::from_json(serde_json::from_str(&json).unwrap()).unwrap();
        let mut theirs = tokenizers::Tokenizer::from_str(&json).unwrap();

        for batch in batches {
            let mine: Vec<NewToken> = batch
                .iter()
                .map(|(content, special)| {
                    if *special {
                        NewToken::special(*content)
                    } else {
                        NewToken::new(*content)
                    }
                })
                .collect();
            let hf: Vec<tokenizers::AddedToken> = batch
                .iter()
                .map(|(content, special)| tokenizers::AddedToken::from(*content, *special))
                .collect();

            ours.add_tokens(&mine).unwrap();
            theirs.add_tokens(&hf);

            assert_eq!(
                ours.vocab_size(),
                theirs.get_vocab_size(true),
                "vocabulary size diverged after adding {batch:?}"
            );
            for (content, _) in batch {
                assert_eq!(
                    ours.token_to_id(content),
                    theirs.token_to_id(content),
                    "id diverged for {content:?} after adding {batch:?}"
                );
            }
        }
    }

    #[test]
    fn add_tokens_is_idempotent() {
        let mut tok = tokenizer_of("abc", &[("<think>", false)]);
        let before = tok.vocab_size();

        assert_eq!(
            tok.add_tokens(&[NewToken::new("<|pad|>")]).unwrap().len(),
            1
        );
        assert_eq!(tok.vocab_size(), before + 1);

        // Re-adding reports nothing changed and claims no second ID, which is
        // what protects the loaders that re-run their padding step.
        assert!(
            tok.add_tokens(&[NewToken::new("<|pad|>")])
                .unwrap()
                .is_empty()
        );
        assert_eq!(tok.vocab_size(), before + 1);

        // Re-declaring a token from `tokenizer.json` is a no-op only when every
        // field matches — upstream compares the whole entry, so a differing
        // flag is an update rather than a duplicate (see the promotion test).
        let declared = NewToken {
            normalized: false,
            ..NewToken::new("<think>")
        };
        assert!(tok.add_tokens(&[declared]).unwrap().is_empty());
        assert_eq!(tok.vocab_size(), before + 1);
    }

    #[test]
    fn add_tokens_collapses_repeats_within_one_batch() {
        let mut tok = tokenizer_of("abc", &[]);
        let before = tok.vocab_size();

        let added = tok
            .add_tokens(&[
                NewToken::new("<|pad|>"),
                NewToken::new("<|pad|>"),
                NewToken::new(""),
            ])
            .unwrap();

        assert_eq!(added.len(), 1);
        assert_eq!(tok.vocab_size(), before + 1);
    }

    #[test]
    fn add_tokens_promotes_an_existing_token_to_special() {
        let mut tok = tokenizer_of("abc", &[("<think>", false)]);
        let think = tok.token_to_id("<think>").unwrap();
        assert!(!tok.is_special_token(think));
        let before = tok.vocab_size();

        let changed = tok.add_tokens(&[NewToken::special("<think>")]).unwrap();

        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].id, think, "promotion must not move the ID");
        assert!(tok.is_special_token(think));
        assert_eq!(tok.vocab_size(), before);
    }

    #[test]
    fn add_tokens_keeps_the_id_of_a_string_the_model_already_has() {
        let mut tok = tokenizer_of("abc", &[]);
        let a = tok.token_to_id("a").unwrap();
        let before = tok.vocab_size();

        let changed = tok.add_tokens(&[NewToken::special("a")]).unwrap();

        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].id, a);
        assert_eq!(tok.vocab_size(), before, "no new string, no new entry");
        assert!(tok.is_special_token(a));
    }

    #[test]
    fn added_tokens_are_recognized_when_encoding() {
        let mut tok = tokenizer_of("ab<|pd|>", &[]);
        tok.add_tokens(&[NewToken::new("<|pd|>")]).unwrap();
        let pad = tok.token_to_id("<|pd|>").unwrap();

        assert_eq!(
            tok.encode("a<|pd|>b").unwrap(),
            vec![
                tok.token_to_id("a").unwrap(),
                pad,
                tok.token_to_id("b").unwrap(),
            ]
        );
        assert_eq!(tok.decode(&[pad], false).unwrap(), "<|pd|>");
    }

    #[test]
    fn input_cache_clear_drops_entries() {
        let mut cache = InputCache::new(2);
        cache.insert(b"hello world", &[1, 2, 3], vec![(0, 0)]);
        assert!(cache.reuse_plan(b"hello world").is_some());

        cache.clear();
        assert!(cache.reuse_plan(b"hello world").is_none());
    }

    // ── AddedTokenPolicy ────────────────────────────────────────────────

    #[test]
    fn encode_policy_skips_only_special_added_tokens() {
        let tok = tokenizer_of(
            "abc<|user|><think>",
            &[("<|user|>", true), ("<think>", false)],
        );
        let user = tok.token_to_id("<|user|>").unwrap();
        let think = tok.token_to_id("<think>").unwrap();
        let text = "a<|user|>b<think>c";

        let all = tok
            .encode_with_policy(text, false, AddedTokenPolicy::All)
            .unwrap();
        assert!(all.contains(&user) && all.contains(&think));

        // The special token becomes its characters; the ordinary added token is
        // still a single ID.
        let skip = tok
            .encode_with_policy(text, false, AddedTokenPolicy::SkipSpecial)
            .unwrap();
        let mut expected = tok.encode_ordinary("a<|user|>b").unwrap();
        expected.push(think);
        expected.extend(tok.encode_ordinary("c").unwrap());
        assert_eq!(skip, expected);

        let none = tok
            .encode_with_policy(text, false, AddedTokenPolicy::None)
            .unwrap();
        assert_eq!(none, tok.encode_ordinary(text).unwrap());
        assert!(!none.contains(&user) && !none.contains(&think));
    }

    #[test]
    fn encode_policy_default_is_unchanged_behavior() {
        let tok = tokenizer_of("abc<|user|>", &[("<|user|>", true)]);
        let text = "a<|user|>c";
        assert_eq!(
            tok.encode_with_policy(text, false, AddedTokenPolicy::default())
                .unwrap(),
            tok.encode(text).unwrap()
        );
    }
}

#[cfg(all(test, feature = "hf-hub"))]
mod tests {
    use crate::hf_hub_support::make_api;

    use super::*;

    const HF_MODELS: &[&str] = &[
        "Qwen/Qwen3-0.6B",
        "zai-org/GLM-4.7",
        "deepseek-ai/DeepSeek-V3.2",
        "MiniMaxAI/MiniMax-M2.1",
        "openai/gpt-oss-120b",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/Qwen3-Nemotron-235B-A22B-GenRM",
        "hoangquan456/Kimi-K2.5",
    ];

    // ── Error::is_not_found ─────────────────────────────────────────────────

    #[test]
    fn is_not_found_detects_http_404() {
        let response = ureq::Response::new(404, "Not Found", "").unwrap();
        let err = Error::Hub(hf_hub_support::ApiError::RequestError(Box::new(
            ureq::Error::Status(404, response),
        )));
        assert!(err.is_not_found(), "404 must be reported as not-found");
    }

    /// A 403 (no access / bad credentials) must NOT read as not-found — it is
    /// retryable once the token is fixed, and misclassifying it would make the
    /// caller silently fall through to a different format.
    #[test]
    fn is_not_found_rejects_other_http_statuses() {
        for status in [401, 403, 429, 500, 503] {
            let response = ureq::Response::new(status, "Err", "").unwrap();
            let err = Error::Hub(hf_hub_support::ApiError::RequestError(Box::new(
                ureq::Error::Status(status, response),
            )));
            assert!(!err.is_not_found(), "{status} must not be not-found");
        }
    }

    // ── tiktoken repositories (no tokenizer.json) ───────────────────────────

    /// A repo that ships only `tiktoken.model` must load through the fallback,
    /// and must produce exactly what the documented manual path produces. This
    /// covers the fallback's plumbing — file discovery, family detection,
    /// `tokenizer_config.json` parsing — against a construction that hardcodes
    /// all of it.
    ///
    /// Bit-exactness against the reference tokenizer is covered separately by
    /// `examples/validate_tiktoken.py`.
    #[test]
    fn tiktoken_repo_matches_explicit_construction() {
        const MODEL: &str = "moonshotai/Kimi-K2.6";

        let from_repo = Tokenizer::from_model(MODEL).unwrap();

        // Same tokenizer, assembled by hand from the repo's raw files.
        let api = make_api(None).unwrap();
        let repo = api.model(MODEL.to_string());
        let ranks_path = repo.get("tiktoken.model").unwrap();
        let config: TokenizerConfig = serde_json::from_str(
            &fs::read_to_string(repo.get("tokenizer_config.json").unwrap()).unwrap(),
        )
        .unwrap();
        let ranks =
            tiktoken::parse_tiktoken_model(&fs::read_to_string(&ranks_path).unwrap()).unwrap();
        let declared = config.added_token_configs().unwrap();
        let named = declared.iter().map(|c| (c.id, c.content.clone()));
        let explicit = Tokenizer::from_tiktoken_ranks(
            &ranks,
            TiktokenConfig::kimi(u32::try_from(ranks.len()).unwrap(), named),
        )
        .unwrap();

        for text in [
            "Hello, world!",
            "另一个测试 with mixed 内容",
            "def f(x):\n    return x * 2  # comment\n",
            "數據處理與分析，機器學習模型訓練。",
            "camelCase HTTPRequest O'Brien don't ALLCAPS 12345",
            "  leading and trailing whitespace \n\n\t",
            "",
        ] {
            assert_eq!(
                from_repo.encode(text).unwrap(),
                explicit.encode(text).unwrap(),
                "mismatch for {text:?}",
            );
        }
    }

    /// The declared names in `added_tokens_decoder` must win over the reserved
    /// placeholders, and must still be matched as single tokens inside ordinary
    /// text — otherwise a chat-templated prompt's token count silently drifts.
    #[test]
    fn tiktoken_repo_resolves_declared_special_tokens() {
        let tok = Tokenizer::from_model("moonshotai/Kimi-K2.6").unwrap();

        // Real ids from the repo's `added_tokens_decoder`.
        assert_eq!(tok.token_to_id("[BOS]"), Some(163_584));
        assert_eq!(tok.token_to_id("[EOS]"), Some(163_585));
        assert_eq!(tok.token_to_id("<|im_end|>"), Some(163_586));
        assert_eq!(tok.token_to_id("[UNK]"), Some(163_838));
        assert_eq!(tok.token_to_id("[PAD]"), Some(163_839));

        // An undeclared slot in the reserved window keeps its placeholder.
        assert_eq!(tok.token_to_id("<|reserved_token_163700|>"), Some(163_700));

        // Declared names must not have been flattened into ordinary text.
        assert_eq!(tok.encode("a<|im_end|>b").unwrap(), vec![64, 163_586, 65]);
    }

    /// The declared `special` flag must reach the added tokens rather than being
    /// flattened to `true` across the reserved window. Kimi marks its tool-call
    /// and thinking markers `special: false` precisely so they survive
    /// `skip_special_tokens`; losing the flag makes decoding swallow them, which
    /// surfaces far from its cause as "the model stopped emitting tool calls".
    ///
    /// This has to assert on `decode`: the flag does not affect ids, so `encode`
    /// is byte-identical whether or not the flags are carried through, and an
    /// encode-only assertion cannot catch a regression here.
    #[test]
    fn tiktoken_repo_preserves_declared_special_flags() {
        let tok = Tokenizer::from_model("moonshotai/Kimi-K2.6").unwrap();

        // `<|im_end|>` is declared `special: true`, `<|tool_call_begin|>` false.
        assert!(tok.is_special_token(163_586), "<|im_end|> must be special");
        assert!(
            !tok.is_special_token(163_597),
            "<|tool_call_begin|> is declared special: false and must stay non-special"
        );

        // So skipping specials drops the former and keeps the latter.
        assert_eq!(
            tok.decode(&[163_586, 163_597], true).unwrap(),
            "<|tool_call_begin|>"
        );
    }

    /// Verify that `TokenizerConfig` and `TokenizerJson` deserialize
    /// successfully for a range of HuggingFace models. This tests the JSON
    /// parsing layer only, not the pipeline construction (which may fail for
    /// unsupported step types).
    #[test]
    fn parse_hf_json() {
        let api = make_api(None).unwrap();
        for model in HF_MODELS {
            let repo = api.model(model.to_string());
            let json_path = repo
                .get("tokenizer.json")
                .unwrap_or_else(|e| panic!("{model}: {e}"));
            let json: TokenizerJson = serde_json::from_str(&fs::read_to_string(json_path).unwrap())
                .unwrap_or_else(|e| panic!("{model}: {e}"));
            assert!(
                !matches!(json.model, ModelConfig::Other(_)),
                "{model}: model parsed as Other",
            );
        }
    }

    /// Verify that encode_batch matches sequential encodes.
    #[test]
    fn encode_batch_matches_sequential() {
        let model = "MiniMaxAI/MiniMax-M2.1";
        let ours = Tokenizer::from_model(model).unwrap();

        let inputs = &["Hello, world!", "The quick brown fox", "Test", ""];
        let batch_results = ours.encode_batch(inputs, false).unwrap();

        for (input, batch_result) in inputs.iter().zip(&batch_results) {
            let sequential_result = ours.encode(input).unwrap();
            assert_eq!(
                batch_result, &sequential_result,
                "batch mismatch for {input:?}"
            );
        }
    }

    /// Verify that vocab access methods work correctly.
    #[test]
    fn vocab_access() {
        let model = "MiniMaxAI/MiniMax-M2.1";
        let ours = Tokenizer::from_model(model).unwrap();

        assert!(ours.vocab_size() > 0);

        let token_str = ours.id_to_token(0).expect("token 0 should exist");
        let id = ours
            .token_to_id(token_str)
            .expect("reverse lookup should work");
        assert_eq!(id, 0);
    }

    #[test]
    fn public_added_token_accessors_expose_added_vocab() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        let added_tokens = tok.added_tokens().expect("expected added tokens");

        let think_id = tok.token_to_id("<think>").expect("<think> should exist");
        assert_eq!(added_tokens.token_to_id("<think>"), Some(think_id));
        assert_eq!(added_tokens.id_to_token(think_id), Some("<think>"));

        let mut entries: Vec<_> = added_tokens.iter().collect();
        entries.sort_by_key(|entry| entry.id);
        let special_entry = entries
            .iter()
            .find(|entry| entry.special)
            .expect("expected at least one special added token");
        assert!(tok.is_special_token(special_entry.id));
        assert!(
            entries
                .iter()
                .any(|entry| entry.id == think_id && entry.content == "<think>"),
            "added-token iterator should expose <think>"
        );
    }

    #[test]
    fn from_file_merges_added_tokens_from_tokenizer_config() {
        let dir = std::env::temp_dir().join(format!(
            "fastokens-added-tokens-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();

        let tokenizer_json = serde_json::json!({
            "added_tokens": [
                {
                    "id": 10,
                    "content": "<|im_start|>",
                    "special": true
                },
                {
                    "id": 11,
                    "content": "<|im_end|>",
                    "special": true
                }
            ],
            "model": {
                "type": "BPE",
                "vocab": {
                    "a": 0,
                    "b": 1,
                    "ab": 2
                },
                "merges": ["a b"]
            }
        });
        let tokenizer_config_json = serde_json::json!({
            "added_tokens_decoder": {
                "10": {
                    "content": "<|im_start|>",
                    "special": true
                },
                "11": {
                    "content": "<|im_end|>",
                    "special": true
                },
                "12": {
                    "content": "<|vision_start|>",
                    "special": true
                },
                "13": {
                    "content": "<|image_pad|>",
                    "special": true
                },
                "14": {
                    "content": "<|vision_end|>",
                    "special": true
                }
            }
        });

        fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer_json).unwrap(),
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_vec(&tokenizer_config_json).unwrap(),
        )
        .unwrap();

        let tok = Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
        assert_eq!(tok.token_to_id("<|image_pad|>"), Some(13));
        assert_eq!(tok.id_to_token(13), Some("<|image_pad|>"));
        assert_eq!(
            tok.encode("<|vision_start|><|image_pad|><|vision_end|>")
                .unwrap(),
            vec![12, 13, 14]
        );

        fs::remove_dir_all(dir).unwrap();
    }

    // ── Correctness tests against HuggingFace tokenizers ─────────────

    /// Comprehensive corpus of inputs designed to exercise tokenizer edge
    /// cases. Used by the multi-model correctness tests below.
    const CORPUS: &[&str] = &[
        // ── empty / trivial ──
        "",
        " ",
        "  ",
        "\n",
        "\t",
        "\r\n",
        // ── single characters ──
        "a",
        "Z",
        "0",
        "!",
        "\u{00e9}", // é (precomposed)
        "\u{4e2d}", // 中
        // ── basic text ──
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "A short sentence.",
        // ── whitespace variations ──
        "  leading spaces",
        "trailing spaces  ",
        "  both  sides  ",
        "multiple    internal    spaces",
        "tabs\there\tand\tthere",
        "line\none\nline\ntwo",
        "windows\r\nline\r\nendings",
        "mixed\n\ttabs and\r\nnewlines  with  spaces",
        // ── numbers ──
        "42",
        "3.14159",
        "1,000,000",
        "0xFF",
        "1e-10",
        "Numbers 1234567890 and mixed ABC123def",
        // ── punctuation / special characters ──
        "Hello!!! How are you???",
        "@user #hashtag $100 %50 ^caret &amp *star",
        "a-b_c.d,e;f:g",
        "(parentheses) [brackets] {braces}",
        "\"double quotes\" 'single quotes' `backticks`",
        "path/to/file.txt",
        "https://example.com/path?q=test&lang=en#section",
        "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        // ── Unicode: Latin accented ──
        "caf\u{00e9} r\u{00e9}sum\u{00e9} na\u{00ef}ve",
        "\u{00fc}ber stra\u{00df}e gr\u{00f6}\u{00df}e",
        "se\u{00f1}or ni\u{00f1}o a\u{00f1}o",
        // ── Unicode: CJK ──
        "\u{4f60}\u{597d}\u{4e16}\u{754c}",         // 你好世界
        "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}", // こんにちは
        "\u{c548}\u{b155}\u{d558}\u{c138}\u{c694}", // 안녕하세요
        // ── Unicode: Cyrillic ──
        "\u{041f}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043c}\u{0438}\u{0440}",
        // ── Unicode: Arabic ──
        "\u{0645}\u{0631}\u{062d}\u{0628}\u{0627}",
        // ── Unicode: Devanagari ──
        "\u{0928}\u{092e}\u{0938}\u{094d}\u{0924}\u{0947}",
        // ── Unicode: Emoji ──
        "\u{1f600}\u{1f680}\u{2764}\u{fe0f}",
        "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}\u{200d}\u{1f466}",
        "\u{1f1fa}\u{1f1f8}", // 🇺🇸
        // ── Unicode: combining marks (NFD forms) ──
        "e\u{0301}", // e + combining acute
        "n\u{0303}", // n + combining tilde
        "a\u{0308}", // a + combining diaeresis
        // ── mixed scripts ──
        "Hello \u{4e16}\u{754c} \u{041c}\u{0438}\u{0440}!",
        "User123 wrote: \u{4f60}\u{597d}!",
        // ── code / programming ──
        "fn main() { println!(\"hello\"); }",
        "def foo(x: int) -> str:\n    return str(x)",
        "SELECT * FROM users WHERE id = 1;",
        "if (x > 0 && y < 10) { z = x + y; }",
        "<html><body><p>Hello</p></body></html>",
        "#include <stdio.h>\nint main() { return 0; }",
        "import numpy as np\nx = np.array([1, 2, 3])",
        // ── JSON / structured data ──
        "{\"key\": \"value\", \"number\": 42, \"array\": [1, 2, 3]}",
        "[{\"id\": 1}, {\"id\": 2}]",
        // ── repeated patterns ──
        "aaaaaaaaaa",
        "abababababababab",
        "the the the the the the the the",
        "....",
        "----",
        "    ",
        "\n\n\n\n",
        // ── longer mixed content ──
        "This is a longer sentence with various elements: numbers (42, 3.14), \
         symbols (@#$), Unicode (caf\u{00e9}, \u{4f60}\u{597d}), and more.",
        "The year 2024 was notable for advances in AI. Models like GPT-4 and \
         Claude demonstrated remarkable capabilities in reasoning, coding, and \
         multilingual understanding.",
        // ── alphabet / character sequences ──
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789",
        // ── boundary / edge cases ──
        "a\nb\nc\n",
        "# Heading\n\n- item 1\n- item 2\n\n```code```",
        "\u{ffff}",  // max BMP non-character
        "\u{0080}",  // first non-ASCII
        "\u{07ff}",  // max 2-byte UTF-8
        "\u{0800}",  // first 3-byte UTF-8
        "\u{10000}", // first surrogate-pair range
        // ── unusual / invalid-ish Unicode ──
        "\u{fffd}",                                  // replacement character
        "\u{feff}Hello",                             // BOM prefix
        "\u{0000}",                                  // null
        "abc\u{0000}def",                            // embedded null
        "\u{fffe}",                                  // non-character
        "\u{fdd0}",                                  // non-character (FDD0 block)
        "\u{200b}\u{200c}\u{200d}",                  // zero-width space / ZWNJ / ZWJ
        "\u{202e}Hello\u{202c}",                     // RTL override + pop directional
        "\u{0001}\u{0002}\u{001f}\u{007f}",          // C0 controls + DEL
        "\u{0300}",                                  // lone combining grave (no base)
        "a\u{0300}\u{0301}\u{0302}\u{0303}\u{0304}", // 5 combining marks on one base
        "\u{e000}\u{f8ff}",                          // private use area
        "\u{01c5}\u{01c8}\u{01cb}",                  // titlecase letters (Dž Lj Nj)
        "\u{2028}\u{2029}",                          // line / paragraph separators
        "\u{fff9}\u{fffa}\u{fffb}",                  // interlinear annotation
        "\u{d7ff}\u{10ffff}",                        // last before surrogates + max codepoint
        // ── potential BPE merge edge cases ──
        "ab",
        "abc",
        "abcd",
        "aaa",
        "aaaa",
        "aaaaa",
        // ── markdown / formatting ──
        "**bold** *italic* ~~strikethrough~~ __underline__",
        "```rust\nfn main() {}\n```",
        "> blockquote\n>> nested",
        "| col1 | col2 |\n|------|------|\n| a    | b    |",
    ];

    /// Helper: compare both encoding and decoding of every input in `corpus`
    /// between our tokenizer and the HuggingFace tokenizer for a given model.
    /// Returns a list of failure descriptions (empty = all passed).
    fn compare_encode_decode(model_name: &str, corpus: &[&str]) -> Vec<String> {
        let hf = tokenizers::Tokenizer::from_pretrained(model_name, None)
            .unwrap_or_else(|e| panic!("{model_name}: HF load failed: {e}"));
        let ours = Tokenizer::from_model(model_name)
            .unwrap_or_else(|e| panic!("{model_name}: fastokens load failed: {e}"));

        let mut failures = Vec::new();
        for &input in corpus {
            let hf_enc = hf
                .encode(input, false)
                .unwrap_or_else(|e| panic!("{model_name}: HF encode({input:?}): {e}"));
            let hf_ids = hf_enc.get_ids().to_vec();
            let our_ids = match ours.encode(input) {
                Ok(ids) => ids,
                Err(e) => {
                    failures.push(format!("  encode error on {input:?}: {e}"));
                    continue;
                }
            };
            if our_ids != hf_ids {
                failures.push(format!(
                    "  encode mismatch on {input:?}: got {} tokens, expected {}\n\
                     \x20   ours: {:?}\n\
                     \x20   hf:   {:?}",
                    our_ids.len(),
                    hf_ids.len(),
                    &our_ids[..our_ids.len().min(20)],
                    &hf_ids[..hf_ids.len().min(20)],
                ));
            }

            // Decode comparison (skip empty inputs / empty token sequences).
            if input.is_empty() || hf_ids.is_empty() {
                continue;
            }
            let hf_decoded = match hf.decode(&hf_ids, false) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let our_decoded = match ours.decode(&hf_ids, false) {
                Ok(d) => d,
                Err(e) => {
                    failures.push(format!("  decode error on {input:?}: {e}"));
                    continue;
                }
            };
            if our_decoded != hf_decoded {
                failures.push(format!(
                    "  decode mismatch on {input:?}:\n\
                     \x20   ours: {:?}\n\
                     \x20   hf:   {:?}",
                    &our_decoded[..our_decoded.len().min(100)],
                    &hf_decoded[..hf_decoded.len().min(100)],
                ));
            }
        }
        failures
    }

    // ── Per-model encoding correctness ───────────────────────────────

    #[test]
    fn correctness_minimax_m2_1() {
        let f = compare_encode_decode("MiniMaxAI/MiniMax-M2.1", CORPUS);
        assert!(f.is_empty(), "MiniMaxAI/MiniMax-M2.1:\n{}", f.join("\n"));
    }

    #[test]
    fn correctness_nemotron() {
        let f = compare_encode_decode("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", CORPUS);
        assert!(
            f.is_empty(),
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16:\n{}",
            f.join("\n")
        );
    }

    #[test]
    fn correctness_deepseek_v3_2() {
        let f = compare_encode_decode("deepseek-ai/DeepSeek-V3.2", CORPUS);
        assert!(f.is_empty(), "deepseek-ai/DeepSeek-V3.2:\n{}", f.join("\n"));
    }

    #[test]
    fn correctness_gpt_oss() {
        let f = compare_encode_decode("openai/gpt-oss-120b", CORPUS);
        assert!(f.is_empty(), "openai/gpt-oss-120b:\n{}", f.join("\n"));
    }

    #[test]
    fn ignore_merges_glm47() {
        let model = "zai-org/GLM-4.7";
        let hf = tokenizers::Tokenizer::from_pretrained(model, None).unwrap();
        let ours = Tokenizer::from_model(model).unwrap();

        // " имущества" is a single token (140507) in GLM-4.7 vocab.
        // BPE merging alone produces 3 tokens — ignore_merges must
        // short-circuit to the vocab entry.
        let text = " имущества";
        let hf_ids = hf.encode(text, false).unwrap().get_ids().to_vec();
        let our_ids = ours.encode(text).unwrap();
        assert_eq!(
            our_ids, hf_ids,
            "ignore_merges mismatch on {text:?}: ours={our_ids:?} hf={hf_ids:?}"
        );

        // Also test with random-token-decoded text (the benchmark pattern).
        let vocab_size = hf.get_vocab_size(false) as u64;
        let random_ids: Vec<u32> = (0..5000)
            .map(|i| {
                ((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1) % vocab_size) as u32
            })
            .collect();
        let text = hf.decode(&random_ids, true).unwrap();
        let hf_enc = hf.encode(text.as_str(), false).unwrap().get_ids().to_vec();
        let our_enc = ours.encode(&text).unwrap();
        assert_eq!(
            our_enc,
            hf_enc,
            "ignore_merges random-decode mismatch: {} vs {} tokens",
            our_enc.len(),
            hf_enc.len()
        );
    }

    #[test]
    fn correctness_qwen3() {
        let f = compare_encode_decode("Qwen/Qwen3-0.6B", CORPUS);
        assert!(f.is_empty(), "Qwen/Qwen3-0.6B:\n{}", f.join("\n"));
    }

    #[test]
    fn correctness_mistral_nemo() {
        let f = compare_encode_decode("mistralai/Mistral-Nemo-Instruct-2407", CORPUS);
        assert!(
            f.is_empty(),
            "mistralai/Mistral-Nemo-Instruct-2407:\n{}",
            f.join("\n")
        );
    }

    #[test]
    fn correctness_qwen3_nemotron() {
        let f = compare_encode_decode("nvidia/Qwen3-Nemotron-235B-A22B-GenRM", CORPUS);
        assert!(
            f.is_empty(),
            "nvidia/Qwen3-Nemotron-235B-A22B-GenRM:\n{}",
            f.join("\n")
        );
    }

    #[test]
    fn correctness_kimi_k2_5() {
        let f = compare_encode_decode("hoangquan456/Kimi-K2.5", CORPUS);
        assert!(f.is_empty(), "hoangquan456/Kimi-K2.5:\n{}", f.join("\n"));
    }

    // ── Cache consistency ────────────────────────────────────────────

    /// Verify that encoding the same input twice produces identical results,
    /// exercising both the cold (cache miss) and warm (cache hit) paths.
    #[test]
    fn cache_consistency() {
        let model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16";
        let ours = Tokenizer::from_model(model).unwrap();

        let inputs = &[
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "caf\u{00e9} r\u{00e9}sum\u{00e9}",
            "\u{4f60}\u{597d}\u{4e16}\u{754c}",
            "fn main() { println!(\"hello\"); }",
            "a b c d e f g h i j k l m n o p",
            "aaaaaaaaaa bbbbbbbbbb cccccccccc",
        ];

        for &input in inputs {
            let first = ours.encode(input).unwrap();
            let second = ours.encode(input).unwrap();
            assert_eq!(first, second, "cache inconsistency for {input:?}");
            // Third call to exercise potential L1→L2 promotion paths.
            let third = ours.encode(input).unwrap();
            assert_eq!(first, third, "cache inconsistency (3rd call) for {input:?}");
        }
    }

    /// Same as above but for the fused byte-level path (Nemotron uses
    /// Sequence([Split, ByteLevel]) which triggers the fused code path).
    #[test]
    fn cache_consistency_fused() {
        let model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16";
        let ours = Tokenizer::from_model(model).unwrap();

        // Verify the fused path is active.
        assert!(ours.split_only.is_some(), "expected fused path for {model}",);

        // Run the same input many times to stress the fused cache.
        let input = "The year 2024 was notable for advances in AI. Models like \
                      GPT-4 and Claude demonstrated remarkable capabilities.";
        let baseline = ours.encode(input).unwrap();
        for i in 0..20 {
            let result = ours.encode(input).unwrap();
            assert_eq!(result, baseline, "fused cache drift on iteration {i}");
        }
    }

    // ── Added tokens (model-specific) ────────────────────────────────

    /// MiniMax-M2.1 has added tokens like <filename>, <reponame>, <think>,
    /// etc. Verify they are handled identically to HF.
    #[test]
    fn added_tokens_minimax() {
        let corpus = &[
            "<filename>",
            "open <filename> for reading",
            "<filename><reponame>",
            "printf(\"%s <filename>\\n\")",
            "<think>Let me reason about this.</think>",
            "<think>load <filename> from <reponame></think>",
            "<file> is not <filename>",
            "<fim_prefix>code here<fim_suffix>more code<fim_middle>",
        ];
        let f = compare_encode_decode("MiniMaxAI/MiniMax-M2.1", corpus);
        assert!(
            f.is_empty(),
            "MiniMaxAI/MiniMax-M2.1 added tokens:\n{}",
            f.join("\n")
        );
    }

    /// DeepSeek-V3.2 added tokens.
    #[test]
    fn added_tokens_deepseek() {
        let corpus = &[
            "<|begin▁of▁sentence|>Hello",
            "Hello<|end▁of▁sentence|>",
            "<|User|>What is 2+2?<|Assistant|>4<|end▁of▁sentence|>",
            "Normal text without special tokens",
            "<|tool▁calls▁begin|>call<|tool▁calls▁end|>",
        ];
        let f = compare_encode_decode("deepseek-ai/DeepSeek-V3.2", corpus);
        assert!(
            f.is_empty(),
            "deepseek-ai/DeepSeek-V3.2 added tokens:\n{}",
            f.join("\n")
        );
    }

    /// Qwen3 added tokens.
    #[test]
    fn added_tokens_qwen3() {
        let corpus = &[
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
            "<|im_start|>user\nHello!<|im_end|>",
            "<|endoftext|>",
            "Plain text with no special tokens at all.",
        ];
        let f = compare_encode_decode("Qwen/Qwen3-0.6B", corpus);
        assert!(
            f.is_empty(),
            "Qwen/Qwen3-0.6B added tokens:\n{}",
            f.join("\n")
        );
    }

    /// token_to_id must find added tokens, not just BPE model vocab entries.
    ///
    /// Root cause of the Qwen3VLProcessor._check_special_mm_tokens failure:
    /// `convert_tokens_to_ids("<|image_pad|>")` calls `token_to_id`, which
    /// previously only searched the BPE model vocabulary and returned None for
    /// added tokens, causing the processor to compare input_ids against
    /// unk_token_id (0) instead of the real image-pad token ID.
    #[test]
    fn token_to_id_searches_added_tokens() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        // These tokens live in added_tokens, not the BPE model vocab.
        for token in &[
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|im_start|>",
        ] {
            let id = tok.token_to_id(token);
            assert!(id.is_some(), "token_to_id({token:?}) returned None");
            // Round-trip: the ID must decode back to the same string.
            assert_eq!(tok.id_to_token(id.unwrap()), Some(*token));
        }
    }

    /// `encode_segments` honors the per-segment trust boundary and concatenates
    /// each segment's ids without flattening or crossing BPE boundaries.
    #[test]
    fn encode_segments_honors_trust_boundary() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        let special = "<|im_start|>";

        // Trusted segment: the control token is recognized as a single id.
        let trusted = tok
            .encode_segments(&[EncodeSegment::special(special)])
            .unwrap();
        assert_eq!(trusted, tok.encode(special).unwrap());
        assert_eq!(trusted.len(), 1, "control token should be one id");

        // Untrusted segment: the same text is encoded as ordinary content and
        // must NOT collapse to the special id.
        let untrusted = tok
            .encode_segments(&[EncodeSegment::ordinary(special)])
            .unwrap();
        assert_eq!(untrusted, tok.encode_ordinary(special).unwrap());
        assert_ne!(untrusted, trusted);

        // Mixed segments concatenate independently: a literal control token in
        // the untrusted content segment stays ordinary.
        let content = "hello <|im_start|> world";
        let got = tok
            .encode_segments(&[
                EncodeSegment::special(special),
                EncodeSegment::ordinary(content),
            ])
            .unwrap();
        let mut want = tok.encode(special).unwrap();
        want.extend(tok.encode_ordinary(content).unwrap());
        assert_eq!(got, want);

        // Empty input yields no tokens.
        assert!(tok.encode_segments(&[]).unwrap().is_empty());
        assert!(
            tok.encode_segments(&[EncodeSegment::ordinary("")])
                .unwrap()
                .is_empty()
        );
    }

    // Qwen2-VL's image token is located in the tokenizer_config.json's added_token_configs
    #[test]
    fn added_tokens_qwen2_vl_image_pad() {
        let model = "Qwen/Qwen2-VL-2B-Instruct";
        let api = make_api(None).unwrap();
        let repo = api.model(model.to_string());
        let tokenizer_config_path = repo.get("tokenizer_config.json").unwrap();
        let tokenizer_config: TokenizerConfig =
            serde_json::from_str(&fs::read_to_string(tokenizer_config_path).unwrap()).unwrap();

        let tok = Tokenizer::from_model(model).unwrap();
        let image_pad_id = tokenizer_config
            .added_token_configs()
            .unwrap()
            .into_iter()
            .find(|token| token.content == "<|image_pad|>")
            .map(|token| token.id)
            .expect("<|image_pad|> should exist in tokenizer_config.json");

        assert_eq!(tok.token_to_id("<|image_pad|>"), Some(image_pad_id));
        assert_eq!(tok.id_to_token(image_pad_id), Some("<|image_pad|>"));
        assert_eq!(tok.decode(&[image_pad_id], false).unwrap(), "<|image_pad|>");
    }

    /// Qwen3-VL vision tokens — the exact text that triggered:
    ///
    ///   ValueError: Failed to apply Qwen3VLProcessor on
    ///   data={'text': '<|vision_start|><|image_pad|><|vision_end|>'}
    ///   with kwargs={'truncation': False}
    ///
    /// Qwen3-0.6B ships with the full set of VL tokens in its added_tokens
    /// array.  A sequence that consists *entirely* of adjacent special tokens
    /// (no regular text in between) exercises the code path where
    /// build_pre_tokenized produces only zero-length Token splits.
    #[test]
    fn added_tokens_qwen3vl_vision_sequence() {
        let corpus = &[
            // Exact failing input from vLLM / Qwen3VLProcessor.
            "<|vision_start|><|image_pad|><|vision_end|>",
            // Bare image-pad token.
            "<|image_pad|>",
            // Multiple adjacent image-pad tokens (real prompts have dozens).
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>",
            // Mixed: VL tokens followed by regular text.
            "<|vision_start|><|image_pad|><|vision_end|>\nDescribe this image.",
        ];
        let f = compare_encode_decode("Qwen/Qwen3.5-27B", corpus);
        assert!(
            f.is_empty(),
            "Qwen/Qwen3.5-27B VL vision sequence:\n{}",
            f.join("\n")
        );
    }

    /// Nemotron added tokens.
    #[test]
    fn added_tokens_nemotron() {
        let corpus = &[
            "<|begin_of_text|>Hello world",
            "Hello<|end_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>",
            "No special tokens here.",
        ];
        let f = compare_encode_decode("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", corpus);
        assert!(
            f.is_empty(),
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 added tokens:\n{}",
            f.join("\n")
        );
    }

    // ── Long input stress test ───────────────────────────────────────

    /// Verify correctness on a longer input that exercises the parallel
    /// tokenization path (>128 splits).
    #[test]
    fn long_input_correctness() {
        let model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16";
        let hf = tokenizers::Tokenizer::from_pretrained(model_name, None).unwrap();
        let ours = Tokenizer::from_model(model_name).unwrap();

        // Build a ~10KB input from repeated varied content.
        let block = "The quick brown fox jumps over the lazy dog. \
                      Numbers: 42, 3.14, 1000. Code: fn main() {} \
                      Unicode: caf\u{00e9}, \u{4f60}\u{597d}. \
                      Special: @#$%^&*(). ";
        let input: String = block.repeat(100);
        assert!(input.len() > 8000);

        let hf_ids = hf.encode(input.as_str(), false).unwrap().get_ids().to_vec();
        let our_ids = ours.encode(&input).unwrap();
        assert_eq!(
            our_ids,
            hf_ids,
            "long input mismatch: {} vs {} tokens",
            our_ids.len(),
            hf_ids.len(),
        );
    }

    /// Same long-input test for a non-fused model.
    #[test]
    fn long_input_correctness_minimax() {
        let model_name = "MiniMaxAI/MiniMax-M2.1";
        let hf = tokenizers::Tokenizer::from_pretrained(model_name, None).unwrap();
        let ours = Tokenizer::from_model(model_name).unwrap();

        let block = "The quick brown fox jumps over the lazy dog. \
                      Numbers: 42, 3.14, 1000. Code: fn main() {} \
                      Unicode: caf\u{00e9}, \u{4f60}\u{597d}. \
                      Special: @#$%^&*(). ";
        let input: String = block.repeat(100);

        let hf_ids = hf.encode(input.as_str(), false).unwrap().get_ids().to_vec();
        let our_ids = ours.encode(&input).unwrap();
        assert_eq!(
            our_ids,
            hf_ids,
            "long input mismatch: {} vs {} tokens",
            our_ids.len(),
            hf_ids.len(),
        );
    }

    // ── Extended dataset tests (run with `cargo test -- --ignored`) ──

    use std::sync::OnceLock;

    struct ExtendedCorpus {
        longbench: Vec<String>,
        sharegpt: Vec<String>,
    }

    fn extended_corpus() -> &'static ExtendedCorpus {
        static CORPUS: OnceLock<ExtendedCorpus> = OnceLock::new();
        CORPUS.get_or_init(|| {
            let api = make_api(None).unwrap();

            // LongBench-v2: first 100 samples
            let lb_repo = api.dataset("zai-org/LongBench-v2".to_string());
            let lb_path = lb_repo.get("data.json").unwrap();
            let lb_data: Vec<serde_json::Value> =
                serde_json::from_str(&fs::read_to_string(lb_path).unwrap()).unwrap();
            let longbench: Vec<String> = lb_data
                .iter()
                .filter_map(|item| {
                    let ctx = item.get("context")?.as_str()?;
                    if ctx.is_empty() {
                        None
                    } else {
                        Some(ctx.to_string())
                    }
                })
                .collect();

            // ShareGPT52K: first 1000 samples
            let sg_repo = api.dataset("RyokoAI/ShareGPT52K".to_string());
            let sg_path = sg_repo.get("sg_90k_part1.json").unwrap();
            let sg_data: Vec<serde_json::Value> =
                serde_json::from_str(&fs::read_to_string(sg_path).unwrap()).unwrap();
            let sharegpt: Vec<String> = sg_data
                .iter()
                .filter_map(|item| {
                    let messages = item.get("conversations")?.as_array()?;
                    let parts: Vec<String> = messages
                        .iter()
                        .filter_map(|msg| {
                            let role = msg
                                .get("from")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            let value = msg.get("value").and_then(|v| v.as_str())?;
                            if value.is_empty() {
                                return None;
                            }
                            Some(format!("[{role}]: {value}"))
                        })
                        .collect();
                    if parts.is_empty() {
                        None
                    } else {
                        Some(parts.join("\n\n"))
                    }
                })
                .collect();

            ExtendedCorpus {
                longbench,
                sharegpt,
            }
        })
    }

    /// Compare encoding and decoding in batches using encode_batch.
    fn compare_encode_decode_batched(
        model_name: &str,
        corpus: &[String],
        batch_size: usize,
        progress: bool,
    ) -> Vec<String> {
        let hf = tokenizers::Tokenizer::from_pretrained(model_name, None)
            .unwrap_or_else(|e| panic!("{model_name}: HF load failed: {e}"));
        let ours = Tokenizer::from_model(model_name)
            .unwrap_or_else(|e| panic!("{model_name}: fastokens load failed: {e}"));

        let total = corpus.len();
        let mut processed = 0usize;
        let mut failures = Vec::new();
        for chunk in corpus.chunks(batch_size) {
            let hf_results: Vec<Vec<u32>> = chunk
                .iter()
                .map(|input| {
                    hf.encode(input.as_str(), false)
                        .unwrap_or_else(|e| panic!("{model_name}: HF encode: {e}"))
                        .get_ids()
                        .to_vec()
                })
                .collect();

            let our_results = match ours.encode_batch(chunk, false) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("  encode_batch error: {e}"));
                    continue;
                }
            };

            for (i, (hf_ids, our_ids)) in hf_results.iter().zip(our_results.iter()).enumerate() {
                let input = &chunk[i];
                let input_preview = {
                    let mut end = input.len().min(80);
                    while end < input.len() && !input.is_char_boundary(end) {
                        end += 1;
                    }
                    &input[..end]
                };

                if our_ids != hf_ids {
                    failures.push(format!(
                        "  encode mismatch on {:?}: got {} tokens, expected {}\n\
                         \x20   ours: {:?}\n\
                         \x20   hf:   {:?}",
                        input_preview,
                        our_ids.len(),
                        hf_ids.len(),
                        &our_ids[..our_ids.len().min(20)],
                        &hf_ids[..hf_ids.len().min(20)],
                    ));
                }

                // Decode comparison.
                if hf_ids.is_empty() || input.is_empty() {
                    continue;
                }
                let hf_decoded = match hf.decode(hf_ids, false) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let our_decoded = match ours.decode(hf_ids, false) {
                    Ok(d) => d,
                    Err(e) => {
                        failures.push(format!("  decode error on {input_preview:?}: {e}"));
                        continue;
                    }
                };
                if our_decoded != hf_decoded {
                    failures.push(format!(
                        "  decode mismatch on {input_preview:?}:\n\
                         \x20   ours: {:?}\n\
                         \x20   hf:   {:?}",
                        &our_decoded[..our_decoded.len().min(100)],
                        &hf_decoded[..hf_decoded.len().min(100)],
                    ));
                }
            }
            processed += chunk.len();
            if progress {
                eprint!(
                    "\r  {model_name}: {processed}/{total} ({:.0}%)",
                    processed as f64 / total as f64 * 100.0,
                );
            }
        }
        if progress {
            eprintln!();
        }
        failures
    }

    fn run_extended(model_name: &str) {
        let progress = std::env::var("EXTENDED_PROGRESS").is_ok();
        let corpus = extended_corpus();
        if progress {
            eprintln!(
                "  {model_name}: longbench ({} samples)",
                corpus.longbench.len()
            );
        }
        let mut failures =
            compare_encode_decode_batched(model_name, &corpus.longbench, 10, progress);
        if progress {
            eprintln!(
                "  {model_name}: sharegpt ({} samples)",
                corpus.sharegpt.len()
            );
        }
        failures.extend(compare_encode_decode_batched(
            model_name,
            &corpus.sharegpt,
            10,
            progress,
        ));
        assert!(
            failures.is_empty(),
            "{model_name} extended ({} failures):\n{}",
            failures.len(),
            failures.join("\n"),
        );
    }

    #[test]
    #[ignore]
    fn extended_minimax_m2_1() {
        run_extended("MiniMaxAI/MiniMax-M2.1");
    }

    #[test]
    #[ignore]
    fn extended_nemotron() {
        run_extended("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16");
    }

    #[test]
    #[ignore]
    fn extended_deepseek_v3_2() {
        run_extended("deepseek-ai/DeepSeek-V3.2");
    }

    #[test]
    #[ignore]
    fn extended_gpt_oss() {
        run_extended("openai/gpt-oss-120b");
    }

    #[test]
    #[ignore]
    fn extended_qwen3() {
        run_extended("Qwen/Qwen3-0.6B");
    }

    #[test]
    #[ignore]
    fn extended_mistral_nemo() {
        run_extended("mistralai/Mistral-Nemo-Instruct-2407");
    }

    #[test]
    #[ignore]
    fn extended_qwen3_nemotron() {
        run_extended("nvidia/Qwen3-Nemotron-235B-A22B-GenRM");
    }

    #[test]
    #[ignore]
    fn extended_mistral_large() {
        run_extended("mistralai/Mistral-Large-3-675B-Instruct-2512");
    }

    #[test]
    #[ignore]
    fn extended_qwen_small() {
        run_extended("Qwen/Qwen3-0.6B");
    }

    // ── encode / decode correctness ─────────────────────────────────────────

    /// Encode without special tokens → decode → original text, for all models.
    #[test]
    fn encode_decode_roundtrip_all_models() {
        let texts = &[
            "Hello, world!",
            "日本語テスト",
            "The quick brown fox jumps over the lazy dog.",
            "fn main() { println!(\"hello\"); }",
            "   leading and trailing spaces   ",
            "line1\nline2\ttabbed",
            "0123456789",
            "🌍🎉✨",
        ];
        let failures: Vec<String> = HF_MODELS
            .iter()
            .flat_map(|model| {
                let tok = match Tokenizer::from_model(model) {
                    Ok(t) => t,
                    Err(e) => return vec![format!("{model}: load error: {e}")],
                };
                texts
                    .iter()
                    .filter_map(|text| {
                        let ids = tok.encode_with_special_tokens(text, false).ok()?;
                        let decoded = tok.decode(&ids, false).ok()?;
                        if decoded != *text {
                            Some(format!("{model}: {text:?} → {decoded:?}"))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        assert!(
            failures.is_empty(),
            "encode→decode roundtrip failures:\n{}",
            failures.join("\n")
        );
    }

    /// Models with add_bos_token=true prepend BOS when add_special_tokens=true.
    ///
    /// In HuggingFace, `add_bos_token` in `tokenizer_config.json` gates whether
    /// the BOS token is inserted. Our Rust side implements this through the
    /// post-processor configured in `tokenizer.json`.  This test verifies the
    /// three key behaviours:
    ///
    /// 1. add_special_tokens=true  → BOS is the first token ID
    /// 2. add_special_tokens=false → BOS is absent
    /// 3. A model without a BOS post-processor (Qwen3) never adds BOS
    #[test]
    fn add_bos_token() {
        // ── model WITH add_bos_token (Mistral-Nemo, BOS = <s> id=1) ──────────
        let tok = Tokenizer::from_model("mistralai/Mistral-Nemo-Instruct-2407").unwrap();
        let bos_id = tok.token_to_id("<s>").expect("<s> not in vocabulary");

        let with_bos = tok.encode_with_special_tokens("hello world", true).unwrap();
        let without_bos = tok
            .encode_with_special_tokens("hello world", false)
            .unwrap();

        assert_eq!(
            with_bos.first().copied(),
            Some(bos_id),
            "first token should be BOS when add_special_tokens=true"
        );
        assert_ne!(
            without_bos.first().copied(),
            Some(bos_id),
            "BOS should be absent when add_special_tokens=false"
        );
        // The content tokens are identical in both cases.
        assert_eq!(&with_bos[1..], without_bos.as_slice());

        // ── model WITHOUT add_bos_token (Qwen3-0.6B) ─────────────────────────
        let tok_q = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        let with_flag = tok_q
            .encode_with_special_tokens("hello world", true)
            .unwrap();
        let without_flag = tok_q
            .encode_with_special_tokens("hello world", false)
            .unwrap();
        assert_eq!(
            with_flag, without_flag,
            "Qwen3 has no BOS post-processor — add_special_tokens should have no effect"
        );
    }

    /// decode(ids, skip=true) omits BOS/EOS; decode(ids, skip=false) includes them.
    #[test]
    fn decode_skip_special_tokens() {
        // Mistral-Nemo adds BOS (<s>, id=1) in basic encoding.
        let model = "mistralai/Mistral-Nemo-Instruct-2407";
        let tok = Tokenizer::from_model(model).unwrap();
        let text = "hello world";
        let ids_with = tok.encode_with_special_tokens(text, true).unwrap();
        let ids_without = tok.encode_with_special_tokens(text, false).unwrap();
        assert!(
            ids_with.len() > ids_without.len(),
            "expected BOS/EOS from {model}"
        );

        let skipped = tok.decode(&ids_with, true).unwrap();
        assert_eq!(skipped, text);

        let full = tok.decode(&ids_with, false).unwrap();
        assert_ne!(full, text);
        assert!(full.contains(text));
    }

    /// decode_batch produces the same results as sequential decode.
    #[test]
    fn decode_batch_matches_sequential() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        let sentences = &["first sentence", "second sentence", "日本語テスト", ""];
        let id_batches: Vec<Vec<u32>> = sentences
            .iter()
            .map(|s| tok.encode_with_special_tokens(s, false).unwrap())
            .collect();
        let refs: Vec<&[u32]> = id_batches.iter().map(Vec::as_slice).collect();
        let batch_out = tok.decode_batch(&refs, false).unwrap();
        for (out, expected) in batch_out.iter().zip(sentences.iter()) {
            assert_eq!(out, expected);
        }
    }

    /// decode_tokens(strings) == decode(ids) for the same sequence.
    #[test]
    fn decode_tokens_matches_decode_by_id() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        for text in &["Hello, world!", "The quick brown fox", "🌍 emoji"] {
            let ids = tok.encode_with_special_tokens(text, false).unwrap();
            let token_strings: Vec<String> = ids
                .iter()
                .map(|&id| tok.id_to_token(id).unwrap().to_string())
                .collect();
            let via_ids = tok.decode(&ids, false).unwrap();
            let via_tokens = tok.decode_tokens(token_strings).unwrap();
            assert_eq!(via_ids, via_tokens, "mismatch for {text:?}");
        }
    }

    /// Encoding an empty string produces an empty token list.
    #[test]
    fn empty_string_encode_decode() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        let ids = tok.encode_with_special_tokens("", false).unwrap();
        assert!(ids.is_empty(), "expected no tokens for empty string");
        assert_eq!(tok.decode(&[], false).unwrap(), "");
    }

    /// encode → decode → encode is stable (idempotent on second encode).
    #[test]
    fn encode_is_stable_after_decode() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        for text in &["hello world", "日本語テスト", "fn foo() {}"] {
            let ids1 = tok.encode_with_special_tokens(text, false).unwrap();
            let decoded = tok.decode(&ids1, false).unwrap();
            let ids2 = tok.encode_with_special_tokens(&decoded, false).unwrap();
            assert_eq!(ids1, ids2, "encode not stable after decode for {text:?}");
        }
    }

    /// post_process with add_special_tokens=false is the identity for all models.
    #[test]
    fn post_process_false_is_identity_all_models() {
        for model in HF_MODELS {
            let tok = Tokenizer::from_model(model).unwrap();
            let payload = vec![100u32, 200, 300];
            let out = tok.post_process(payload.clone(), false);
            assert_eq!(
                out, payload,
                "{model}: post_process(false) should be identity"
            );
        }
    }

    /// post_process(true) adds at least as many tokens as post_process(false).
    #[test]
    fn post_process_true_adds_special_tokens() {
        // Use Mistral-Nemo which has a post-processor that adds BOS.
        let tok = Tokenizer::from_model("mistralai/Mistral-Nemo-Instruct-2407").unwrap();
        let payload = vec![10u32, 20, 30];
        let without = tok.post_process(payload.clone(), false);
        let with_sp = tok.post_process(payload.clone(), true);
        assert_eq!(without, payload);
        assert!(
            with_sp.len() > without.len(),
            "expected special tokens to be added"
        );
        // The original payload IDs appear contiguously somewhere in the output.
        assert!(
            with_sp
                .windows(payload.len())
                .any(|w| w == payload.as_slice()),
            "payload should appear contiguously in post-processed output"
        );
    }

    /// decode of an unknown ID silently skips it, matching HuggingFace.
    #[test]
    fn decode_unknown_id_is_skipped() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        assert_eq!(tok.decode(&[u32::MAX], false).unwrap(), "");
    }

    /// decode interleaves valid tokens with unknown IDs, dropping only the bad ones.
    #[test]
    fn decode_mixed_valid_and_unknown_ids() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        let valid = tok.encode_with_special_tokens("hello", false).unwrap();
        let mut mixed = valid.clone();
        mixed.push(u32::MAX);
        mixed.extend(tok.encode_with_special_tokens(" world", false).unwrap());
        let expected = tok.decode(&valid, false).unwrap()
            + &tok
                .decode(
                    &tok.encode_with_special_tokens(" world", false).unwrap(),
                    false,
                )
                .unwrap();
        assert_eq!(tok.decode(&mixed, false).unwrap(), expected);
    }

    /// id_to_token / token_to_id round-trip for sampled IDs across all models.
    #[test]
    fn token_id_roundtrip_all_models() {
        let probe_ids = [0u32, 1, 2, 100, 1000, 10_000];
        let failures: Vec<String> = HF_MODELS
            .iter()
            .flat_map(|model| {
                let tok = match Tokenizer::from_model(model) {
                    Ok(t) => t,
                    Err(e) => return vec![format!("{model}: load error: {e}")],
                };
                probe_ids
                    .iter()
                    .filter_map(|&id| {
                        let token = tok.id_to_token(id)?;
                        let back = tok.token_to_id(token)?;
                        if back != id {
                            Some(format!("{model}: id {id} → {token:?} → {back}"))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        assert!(
            failures.is_empty(),
            "id↔token roundtrip failures:\n{}",
            failures.join("\n")
        );
    }

    // ── DecodeStream ────────────────────────────────────────────────────────

    const STREAM_MODEL: &str = "Qwen/Qwen3-0.6B";

    fn stream_tok() -> Tokenizer {
        Tokenizer::from_model(STREAM_MODEL).expect("failed to load tokenizer")
    }

    fn stream_collect(tok: &Tokenizer, ids: &[u32], skip: bool) -> (String, usize) {
        let mut buf = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let mut out = String::new();
        for &id in ids {
            let chunk: Option<String> = super::decode_stream_step(
                tok,
                vec![id],
                skip,
                &mut buf,
                &mut prefix,
                &mut prefix_index,
            )
            .unwrap();
            if let Some(c) = chunk {
                out.push_str(&c);
            }
        }
        (out, buf.len())
    }

    #[test]
    fn decode_stream_reconstructs_ascii() {
        let tok = stream_tok();
        let text = "Hello, world! This is a streaming decode test.";
        let ids = tok.encode_with_special_tokens(text, false).unwrap();
        let (decoded, _) = stream_collect(&tok, &ids, false);
        assert_eq!(decoded, text);
    }

    #[test]
    fn decode_stream_reconstructs_unicode() {
        let tok = stream_tok();
        let text = "日本語テスト: こんにちは 🌍 — привет мир";
        let ids = tok.encode_with_special_tokens(text, false).unwrap();
        let (decoded, _) = stream_collect(&tok, &ids, false);
        assert_eq!(decoded, text);
    }

    #[test]
    fn decode_stream_reconstructs_code() {
        let tok = stream_tok();
        let text = r#"fn main() { println!("hello"); }"#;
        let ids = tok.encode_with_special_tokens(text, false).unwrap();
        let (decoded, _) = stream_collect(&tok, &ids, false);
        assert_eq!(decoded, text);
    }

    #[test]
    fn decode_stream_empty_ids_no_output() {
        let tok = stream_tok();
        let (decoded, buf_len) = stream_collect(&tok, &[], false);
        assert!(decoded.is_empty());
        assert_eq!(buf_len, 0);
    }

    #[test]
    fn decode_stream_single_token() {
        let tok = stream_tok();
        let ids = tok.encode_with_special_tokens("hello", false).unwrap();
        assert!(!ids.is_empty());
        let (decoded, _) = stream_collect(&tok, &ids[..1], false);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn decode_stream_batch_step_matches_sequential() {
        let tok = stream_tok();
        let text = "The quick brown fox jumps over the lazy dog.";
        let ids = tok.encode_with_special_tokens(text, false).unwrap();
        let (sequential, _) = stream_collect(&tok, &ids, false);
        let mut buf = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let batch: String = super::decode_stream_step(
            &tok,
            ids.clone(),
            false,
            &mut buf,
            &mut prefix,
            &mut prefix_index,
        )
        .unwrap()
        .unwrap_or_default();
        assert_eq!(sequential, batch);
    }

    #[test]
    fn decode_stream_pre_seeded_only_returns_new_tokens() {
        let tok = stream_tok();
        let prompt = "The capital of France is";
        let cont = " Paris.";
        let prompt_ids = tok.encode_with_special_tokens(prompt, false).unwrap();
        let cont_ids = tok.encode_with_special_tokens(cont, false).unwrap();
        let mut buf = prompt_ids.clone();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let mut out = String::new();
        for &id in &cont_ids {
            let chunk: Option<String> = super::decode_stream_step(
                &tok,
                vec![id],
                false,
                &mut buf,
                &mut prefix,
                &mut prefix_index,
            )
            .unwrap();
            if let Some(c) = chunk {
                out.push_str(&c);
            }
        }
        assert_eq!(out, cont);
    }

    #[test]
    fn decode_stream_skip_special_tokens() {
        let tok = Tokenizer::from_model("mistralai/Mistral-Nemo-Instruct-2407").unwrap();
        let text = "hello";
        let ids_with = tok.encode_with_special_tokens(text, true).unwrap();
        let ids_without = tok.encode_with_special_tokens(text, false).unwrap();
        assert!(
            ids_with.len() > ids_without.len(),
            "expected BOS/EOS tokens"
        );
        let (with_sp, _) = stream_collect(&tok, &ids_with, false);
        let (no_sp, _) = stream_collect(&tok, &ids_with, true);
        assert_eq!(no_sp, text);
        assert!(with_sp.contains(&no_sp));
    }

    #[test]
    fn decode_stream_buffer_does_not_grow_unboundedly() {
        let tok = stream_tok();
        let text = "word ".repeat(80);
        let ids = tok.encode_with_special_tokens(text.trim(), false).unwrap();
        let (_, final_buf_len) = stream_collect(&tok, &ids, false);
        assert!(
            final_buf_len < 10,
            "buffer grew to {final_buf_len} entries after {} tokens",
            ids.len()
        );
    }

    #[test]
    fn decode_stream_chunks_are_non_empty_and_concatenate() {
        let tok = stream_tok();
        let text = "one two three four five six seven eight nine ten";
        let ids = tok.encode_with_special_tokens(text, false).unwrap();
        let mut buf = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let mut chunks: Vec<String> = Vec::new();
        for &id in &ids {
            let chunk: Option<String> = super::decode_stream_step(
                &tok,
                vec![id],
                false,
                &mut buf,
                &mut prefix,
                &mut prefix_index,
            )
            .unwrap();
            if let Some(c) = chunk {
                assert!(!c.is_empty(), "stream emitted an empty chunk");
                chunks.push(c);
            }
        }
        assert_eq!(chunks.concat(), text);
    }

    /// Streaming decode silently skips unknown IDs instead of erroring, so
    /// a single OOV token (e.g. emitted in the gap between tokenizer vocab
    /// and embedding matrix on some Qwen FP8 checkpoints) doesn't kill the
    /// whole generation. Matches HuggingFace DecodeStream behavior.
    #[test]
    fn decode_stream_unknown_id_does_not_error() {
        let tok = stream_tok();
        let mut buf = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let result = super::decode_stream_step(
            &tok,
            vec![u32::MAX],
            false,
            &mut buf,
            &mut prefix,
            &mut prefix_index,
        );
        assert!(result.is_ok(), "expected Ok, got {result:?}");
    }

    #[test]
    fn decode_stream_invalid_prefix_error_message() {
        let tok = stream_tok();
        let ids = tok.encode_with_special_tokens("hello", false).unwrap();
        let mut buf = ids.clone();
        let mut prefix = "ZZZZZZZ".to_string();
        let mut prefix_index = 0usize;
        let result: Result<Option<String>, String> = super::decode_stream_step(
            &tok,
            vec![*ids.last().unwrap()],
            false,
            &mut buf,
            &mut prefix,
            &mut prefix_index,
        );
        if let Err(msg) = result {
            assert!(
                msg.starts_with("Invalid prefix encountered"),
                "unexpected error: {msg:?}"
            );
        }
    }
}

#[cfg(test)]
mod ordinary_tests;
