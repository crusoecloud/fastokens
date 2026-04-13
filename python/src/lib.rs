
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

// ---------------------------------------------------------------------------
// PyEncoding
// ---------------------------------------------------------------------------

/// Minimal stand-in for `tokenizers.Encoding`.
///
/// Returned directly by `Tokenizer.encode` and `Tokenizer.encode_batch` so
/// no Python-side wrapping is needed.  Fields that `fastokens` does not
/// track (`tokens`, `offsets`, `sequence_ids`, `word_ids`) have getters that
/// raise `NotImplementedError` to match the HuggingFace API surface.
#[pyclass(name = "Encoding")]
pub struct PyEncoding {
    #[pyo3(get, set)]
    pub ids: Vec<u32>,
    #[pyo3(get, set)]
    pub attention_mask: Vec<u32>,
    #[pyo3(get, set)]
    pub type_ids: Vec<u32>,
    #[pyo3(get, set)]
    pub special_tokens_mask: Vec<u32>,
    #[pyo3(get, set)]
    pub n_sequences: usize,
    // Backing storage for set-only properties.
    _sequence_ids: Vec<Option<i64>>,
    _word_ids: Vec<Option<i64>>,
}

impl PyEncoding {
    pub fn make(ids: Vec<u32>, attention_mask: Vec<u32>) -> Self {
        let n = ids.len();
        Self {
            type_ids: vec![0u32; n],
            special_tokens_mask: vec![0u32; n],
            n_sequences: 1,
            _sequence_ids: vec![Some(0); n],
            _word_ids: vec![None; n],
            ids,
            attention_mask,
        }
    }

    fn apply_slice(&mut self, start: usize, end: usize) {
        self.ids = self.ids[start..end].to_vec();
        self.attention_mask = self.attention_mask[start..end].to_vec();
        self.type_ids = self.type_ids[start..end].to_vec();
        self.special_tokens_mask = self.special_tokens_mask[start..end].to_vec();
        self._sequence_ids = self._sequence_ids[start..end].to_vec();
        self._word_ids = self._word_ids[start..end].to_vec();
    }

    fn extend_right(&mut self, pad_id: u32, pad_type_id: u32, count: usize) {
        self.ids.extend(vec![pad_id; count]);
        self.attention_mask.extend(vec![0u32; count]);
        self.type_ids.extend(vec![pad_type_id; count]);
        self.special_tokens_mask.extend(vec![0u32; count]);
        self._sequence_ids.extend(vec![None; count]);
        self._word_ids.extend(vec![None; count]);
    }

    fn extend_left(&mut self, pad_id: u32, pad_type_id: u32, count: usize) {
        let mut ids = vec![pad_id; count];
        ids.extend_from_slice(&self.ids);
        let mut mask = vec![0u32; count];
        mask.extend_from_slice(&self.attention_mask);
        let mut type_ids = vec![pad_type_id; count];
        type_ids.extend_from_slice(&self.type_ids);
        let mut special = vec![0u32; count];
        special.extend_from_slice(&self.special_tokens_mask);
        let mut seq_ids = vec![None; count];
        seq_ids.extend_from_slice(&self._sequence_ids);
        let mut word_ids = vec![None; count];
        word_ids.extend_from_slice(&self._word_ids);
        self.ids = ids;
        self.attention_mask = mask;
        self.type_ids = type_ids;
        self.special_tokens_mask = special;
        self._sequence_ids = seq_ids;
        self._word_ids = word_ids;
    }
}

#[pymethods]
impl PyEncoding {
    #[new]
    #[pyo3(signature = (ids, attention_mask = None))]
    fn new(ids: Vec<u32>, attention_mask: Option<Vec<u32>>) -> Self {
        let n = ids.len();
        let mask = attention_mask.unwrap_or_else(|| vec![1u32; n]);
        Self::make(ids, mask)
    }

    fn __len__(&self) -> usize {
        self.ids.len()
    }

    fn __repr__(&self) -> String {
        format!("Encoding(num_tokens={})", self.ids.len())
    }

    // -- Properties that raise NotImplementedError ----------------------

    #[getter]
    fn tokens(&self) -> PyResult<Vec<String>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track token strings; \
             use Tokenizer.id_to_token() to convert individual IDs",
        ))
    }
    #[setter]
    fn set_tokens(&mut self, _v: &Bound<'_, PyAny>) {}

    #[getter]
    fn offsets(&self) -> PyResult<Vec<(usize, usize)>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }
    #[setter]
    fn set_offsets(&mut self, _v: &Bound<'_, PyAny>) {}

    #[getter]
    fn sequence_ids(&self) -> PyResult<Vec<Option<i64>>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track sequence IDs",
        ))
    }
    #[setter]
    fn set_sequence_ids(&mut self, value: Vec<Option<i64>>) {
        self._sequence_ids = value;
    }

    #[getter]
    fn word_ids(&self) -> PyResult<Vec<Option<i64>>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }
    #[setter]
    fn set_word_ids(&mut self, value: Vec<Option<i64>>) {
        self._word_ids = value;
    }

    #[getter]
    fn words(&self) -> PyResult<Vec<Option<i64>>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }
    #[setter]
    fn set_words(&mut self, value: Vec<Option<i64>>) {
        self._word_ids = value;
    }

    /// Always empty — fastokens does not produce overflowing sequences.
    #[getter]
    fn overflowing<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }
    #[setter]
    fn set_overflowing(&mut self, _v: &Bound<'_, PyAny>) {}

    // -- Sequence ID helper ---------------------------------------------

    fn set_sequence_id(&mut self, sequence_id: i64) {
        let n = self.ids.len();
        self._sequence_ids = vec![Some(sequence_id); n];
    }

    // -- Positional mapping (all raise NotImplementedError) -------------

    #[pyo3(signature = (char_pos, sequence_index = 0))]
    fn char_to_token(
        &self,
        char_pos: usize,
        sequence_index: usize,
    ) -> PyResult<Option<usize>> {
        let _ = (char_pos, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }

    #[pyo3(signature = (char_pos, sequence_index = 0))]
    fn char_to_word(
        &self,
        char_pos: usize,
        sequence_index: usize,
    ) -> PyResult<Option<usize>> {
        let _ = (char_pos, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }

    fn token_to_chars(&self, token_index: usize) -> PyResult<Option<(usize, usize)>> {
        let _ = token_index;
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }

    fn token_to_sequence(&self, token_index: usize) -> PyResult<Option<usize>> {
        let _ = token_index;
        Err(PyNotImplementedError::new_err(
            "fastokens does not track sequence IDs",
        ))
    }

    fn token_to_word(&self, token_index: usize) -> PyResult<Option<usize>> {
        let _ = token_index;
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }

    #[pyo3(signature = (word_index, sequence_index = 0))]
    fn word_to_chars(
        &self,
        word_index: usize,
        sequence_index: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        let _ = (word_index, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }

    #[pyo3(signature = (word_index, sequence_index = 0))]
    fn word_to_tokens(
        &self,
        word_index: usize,
        sequence_index: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        let _ = (word_index, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }

    // -- Truncate / pad -------------------------------------------------

    #[pyo3(signature = (max_length, stride = 0, direction = "right"))]
    fn truncate(&mut self, max_length: usize, stride: usize, direction: &str) {
        let _ = stride;
        let n = self.ids.len();
        if n <= max_length {
            return;
        }
        if direction == "left" {
            self.apply_slice(n - max_length, n);
        } else {
            self.apply_slice(0, max_length);
        }
    }

    #[pyo3(signature = (length, direction = "right", pad_id = 0, pad_type_id = 0, pad_token = "[PAD]"))]
    fn pad(
        &mut self,
        length: usize,
        direction: &str,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
    ) {
        let _ = pad_token;
        let n = self.ids.len();
        if length <= n {
            return;
        }
        let deficit = length - n;
        if direction == "left" {
            self.extend_left(pad_id, pad_type_id, deficit);
        } else {
            self.extend_right(pad_id, pad_type_id, deficit);
        }
    }

    // -- Merge ----------------------------------------------------------

    #[staticmethod]
    #[pyo3(signature = (encodings, growing_offsets = true))]
    fn merge(
        py: Python<'_>,
        encodings: Vec<Py<PyEncoding>>,
        growing_offsets: bool,
    ) -> PyEncoding {
        let _ = growing_offsets;
        let mut ids: Vec<u32> = vec![];
        let mut attention_mask: Vec<u32> = vec![];
        let mut type_ids: Vec<u32> = vec![];
        let mut special_tokens_mask: Vec<u32> = vec![];
        let mut n_sequences: usize = 0;
        let mut seq_ids: Vec<Option<i64>> = vec![];
        let mut word_ids: Vec<Option<i64>> = vec![];

        for enc_py in &encodings {
            let enc = enc_py.borrow(py);
            ids.extend_from_slice(&enc.ids);
            attention_mask.extend_from_slice(&enc.attention_mask);
            type_ids.extend_from_slice(&enc.type_ids);
            special_tokens_mask.extend_from_slice(&enc.special_tokens_mask);
            n_sequences += enc.n_sequences;
            seq_ids.extend_from_slice(&enc._sequence_ids);
            word_ids.extend_from_slice(&enc._word_ids);
        }

        PyEncoding {
            ids,
            attention_mask,
            type_ids,
            special_tokens_mask,
            n_sequences,
            _sequence_ids: seq_ids,
            _word_ids: word_ids,
        }
    }
}

// ---------------------------------------------------------------------------
// TruncationParams / PaddingParams
// ---------------------------------------------------------------------------

struct TruncationParams {
    max_length: usize,
    stride: usize,
    strategy: String,
    direction: String,
}

struct PaddingParams {
    direction: String,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: String,
    length: Option<usize>,
    pad_to_multiple_of: Option<usize>,
}

// ---------------------------------------------------------------------------
// PyPostProcessor
// ---------------------------------------------------------------------------

/// Python-facing post-processor object — mirrors `tokenizers.processors.*`.
///
/// Holds the JSON representation of the post-processor so that:
/// - `str(pp)` returns JSON (the setter calls `str()` on whatever it receives)
/// - the object round-trips correctly through the getter/setter pair
#[pyclass(name = "PostProcessor")]
#[derive(Clone)]
struct PyPostProcessor {
    json: String,
}

#[pymethods]
impl PyPostProcessor {
    fn __str__(&self) -> &str {
        &self.json
    }
    fn __repr__(&self) -> &str {
        &self.json
    }
}

// ---------------------------------------------------------------------------
// PyTokenizer
// ---------------------------------------------------------------------------

/// An LLM tokenizer backed by `tokenizer.json`.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: fastokens::Tokenizer,
    trunc: Option<TruncationParams>,
    pad: Option<PaddingParams>,
    /// Cached JSON of the current post-processor (for the getter).
    post_processor_json: Option<String>,
}

impl PyTokenizer {
    /// Build from a raw JSON string, extracting the post-processor field so
    /// the getter can return it without needing to re-serialize.
    fn build_from_str(json: &str, py: Python<'_>) -> PyResult<Self> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let post_processor_json = value
            .get("post_processor")
            .filter(|v| !v.is_null())
            .map(|v| v.to_string());
        let inner = py
            .allow_threads(|| fastokens::Tokenizer::from_json(value).map_err(|e| e.to_string()))
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner, trunc: None, pad: None, post_processor_json })
    }
}

impl PyTokenizer {
    fn do_truncate(&self, ids: &mut Vec<u32>) {
        let Some(ref t) = self.trunc else { return };
        if ids.len() <= t.max_length {
            return;
        }
        if t.direction == "left" {
            ids.drain(..ids.len() - t.max_length);
        } else {
            ids.truncate(t.max_length);
        }
    }

    /// Pad `ids` to `target` length and return the attention mask.
    fn pad_to(&self, ids: &mut Vec<u32>, target: usize) -> Vec<u32> {
        let n_real = ids.len();
        if target <= n_real {
            return vec![1u32; n_real];
        }
        let Some(ref p) = self.pad else {
            return vec![1u32; n_real];
        };
        let deficit = target - n_real;
        if p.direction == "left" {
            let mut new_ids = vec![p.pad_id; deficit];
            new_ids.extend_from_slice(ids);
            *ids = new_ids;
            let mut mask = vec![0u32; deficit];
            mask.extend(vec![1u32; n_real]);
            mask
        } else {
            ids.extend(vec![p.pad_id; deficit]);
            let mut mask = vec![1u32; n_real];
            mask.extend(vec![0u32; deficit]);
            mask
        }
    }

    fn single_pad_target(&self, n: usize) -> usize {
        let Some(ref p) = self.pad else { return n };
        let base = p.length.unwrap_or(n).max(n);
        match p.pad_to_multiple_of {
            Some(m) if m > 0 => (base + m - 1) / m * m,
            _ => base,
        }
    }
}

#[pymethods]
impl PyTokenizer {
    /// Download `tokenizer.json` from HuggingFace Hub for the given model
    /// (e.g. `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    ///
    /// (This is an alias for Tokenizer.from_model)
    #[new]
    fn new(model: &str, py: Python<'_>) -> PyResult<Self> {
        Self::from_model(model, py)
    }

    /// Create a tokenizer from a `tokenizer.json` file.
    #[staticmethod]
    fn from_file(path: &str, py: Python<'_>) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("cannot read {path}: {e}")))?;
        Self::build_from_str(&json, py)
    }

    /// Create a tokenizer from a raw JSON string for `tokenizer.json`.
    #[staticmethod]
    fn from_json_str(json: &str, py: Python<'_>) -> PyResult<Self> {
        Self::build_from_str(json, py)
    }

    /// Download `tokenizer.json` from HuggingFace Hub for the given model
    /// (e.g. `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    #[staticmethod]
    fn from_model(model: &str, py: Python<'_>) -> PyResult<Self> {
        let json = py
            .allow_threads(|| {
                fastokens::Tokenizer::download_tokenizer_json(model).map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Self::build_from_str(&json, py)
    }

    // ── Post-processor ────────────────────────────────────────────────

    /// The current post-processor, or ``None`` if none is configured.
    ///
    /// The returned object's ``__str__`` yields its JSON representation,
    /// so ``str(tokenizer.post_processor)`` round-trips through the setter.
    #[getter]
    fn post_processor(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.post_processor_json {
            None => Ok(py.None()),
            Some(json) => Py::new(py, PyPostProcessor { json: json.clone() })
                .map(|p| p.into_any()),
        }
    }

    /// Set the post-processor.
    ///
    /// Accepts anything whose ``str()`` yields a valid post-processor JSON —
    /// including our own ``PostProcessor`` objects and ``tokenizers.processors.*``
    /// objects from the HuggingFace tokenizers library.
    #[setter]
    fn set_post_processor(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.set_post_processor(None);
            self.post_processor_json = None;
            return Ok(());
        }
        // `tokenizers.processors.*` objects expose `__getstate__` returning JSON
        // bytes — this is the reliable path across all tokenizers versions.
        // For our own `PyPostProcessor` (no `__getstate__`), fall back to
        // `__str__` which returns the JSON string directly.
        let json_str = if let Ok(state) = value.call_method0("__getstate__") {
            if let Ok(bytes) = state.extract::<Vec<u8>>() {
                String::from_utf8(bytes)
                    .map_err(|e| PyValueError::new_err(format!("non-UTF-8 processor state: {e}")))?
            } else {
                value.str()?.to_cow()?.to_string()
            }
        } else {
            value.str()?.to_cow()?.to_string()
        };
        self.update_post_processor_json(&json_str)
    }

    // ── Truncation ────────────────────────────────────────────────────

    #[pyo3(signature = (max_length, stride = 0, strategy = "longest_first", direction = "right"))]
    fn enable_truncation(
        &mut self,
        max_length: usize,
        stride: usize,
        strategy: &str,
        direction: &str,
    ) {
        self.trunc = Some(TruncationParams {
            max_length,
            stride,
            strategy: strategy.to_string(),
            direction: direction.to_string(),
        });
    }

    fn no_truncation(&mut self) {
        self.trunc = None;
    }

    #[getter]
    fn truncation(&self, py: Python<'_>) -> PyObject {
        match &self.trunc {
            None => py.None(),
            Some(t) => {
                let d = PyDict::new(py);
                d.set_item("max_length", t.max_length).unwrap();
                d.set_item("stride", t.stride).unwrap();
                d.set_item("strategy", &t.strategy).unwrap();
                d.set_item("direction", &t.direction).unwrap();
                d.into()
            }
        }
    }

    // ── Padding ───────────────────────────────────────────────────────

    #[pyo3(signature = (direction = "right", pad_id = 0, pad_type_id = 0, pad_token = "[PAD]", length = None, pad_to_multiple_of = None))]
    fn enable_padding(
        &mut self,
        direction: &str,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
        length: Option<usize>,
        pad_to_multiple_of: Option<usize>,
    ) {
        self.pad = Some(PaddingParams {
            direction: direction.to_string(),
            pad_id,
            pad_type_id,
            pad_token: pad_token.to_string(),
            length,
            pad_to_multiple_of,
        });
    }

    fn no_padding(&mut self) {
        self.pad = None;
    }

    #[getter]
    fn padding(&self, py: Python<'_>) -> PyObject {
        match &self.pad {
            None => py.None(),
            Some(p) => {
                let d = PyDict::new(py);
                d.set_item("direction", &p.direction).unwrap();
                d.set_item("pad_id", p.pad_id).unwrap();
                d.set_item("pad_type_id", p.pad_type_id).unwrap();
                d.set_item("pad_token", &p.pad_token).unwrap();
                match p.length {
                    Some(l) => d.set_item("length", l).unwrap(),
                    None => d.set_item("length", py.None()).unwrap(),
                }
                match p.pad_to_multiple_of {
                    Some(m) => d.set_item("pad_to_multiple_of", m).unwrap(),
                    None => d.set_item("pad_to_multiple_of", py.None()).unwrap(),
                }
                d.into()
            }
        }
    }

    // ── Encoding ──────────────────────────────────────────────────────

    /// Run the full encoding pipeline.
    ///
    /// Truncation and padding configured via `enable_truncation` /
    /// `enable_padding` are applied before returning.
    #[pyo3(signature = (input, add_special_tokens = false))]
    fn encode(
        &self,
        input: &str,
        add_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        let (ids, mask) = py
            .allow_threads(|| -> Result<(Vec<u32>, Vec<u32>), String> {
                let mut ids = self
                    .inner
                    .encode_with_special_tokens(input, add_special_tokens)
                    .map_err(|e| e.to_string())?;
                self.do_truncate(&mut ids);
                let target = self.single_pad_target(ids.len());
                let mask = self.pad_to(&mut ids, target);
                Ok((ids, mask))
            })
            .map_err(PyValueError::new_err)?;

        Py::new(py, PyEncoding::make(ids, mask))
    }

    /// Encode a batch of inputs in parallel.
    ///
    /// Truncation is applied per-sequence; padding (if enabled) pads the
    /// batch to a uniform length.
    #[pyo3(signature = (inputs, add_special_tokens = false))]
    fn encode_batch(
        &self,
        inputs: Vec<String>,
        add_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyEncoding>>> {
        use rayon::prelude::*;

        let mut batch: Vec<Vec<u32>> = py
            .allow_threads(|| {
                inputs
                    .par_iter()
                    .map(|s| {
                        self.inner
                            .encode_with_special_tokens(s.as_str(), add_special_tokens)
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .map_err(PyValueError::new_err)?;

        for ids in &mut batch {
            self.do_truncate(ids);
        }

        let pad_target: Option<usize> = self.pad.as_ref().map(|p| {
            let max_len = batch.iter().map(|ids| ids.len()).max().unwrap_or(0);
            let base = p.length.unwrap_or(max_len).max(max_len);
            match p.pad_to_multiple_of {
                Some(m) if m > 0 => (base + m - 1) / m * m,
                _ => base,
            }
        });

        batch
            .into_iter()
            .map(|mut ids| {
                let mask = match pad_target {
                    Some(target) => self.pad_to(&mut ids, target),
                    None => vec![1u32; ids.len()],
                };
                Py::new(py, PyEncoding::make(ids, mask))
            })
            .collect()
    }

    // ── Post-processing ───────────────────────────────────────────────

    /// Apply the post-processor to an existing encoding.
    ///
    /// When `add_special_tokens` is true the post-processor inserts special
    /// tokens (BOS/EOS/etc.).  Pair encodings are not supported.
    #[pyo3(signature = (encoding, pair = None, add_special_tokens = true))]
    fn post_process(
        &self,
        encoding: Py<PyEncoding>,
        pair: Option<Py<PyEncoding>>,
        add_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        if pair.is_some() {
            return Err(PyNotImplementedError::new_err(
                "pair post-processing is not supported by fastokens",
            ));
        }
        if !add_special_tokens {
            return Ok(encoding);
        }
        let ids = encoding.borrow(py).ids.clone();
        let new_ids = py.allow_threads(|| self.inner.post_process(ids, true));
        let n = new_ids.len();
        Py::new(py, PyEncoding::make(new_ids, vec![1u32; n]))
    }

    /// Return the number of special tokens added for a single or pair sequence.
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        if is_pair {
            return 0; // pair not supported
        }
        // Probe: encode empty IDs with and without special tokens.
        let with_special = self.inner.post_process(vec![], true);
        with_special.len()
    }

    // ── Decoding ──────────────────────────────────────────────────────

    /// Decode a list of token strings back into text using the decoder pipeline.
    ///
    /// This is what `convert_tokens_to_string` needs: token strings (e.g.
    /// "Ġhello") → decoded text (" hello").  The decoder (e.g. ByteLevel)
    /// is applied exactly as during normal `decode`.
    fn decode_tokens(&self, tokens: Vec<String>, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| self.inner.decode_tokens(tokens).map_err(|e| e.to_string()))
            .map_err(PyValueError::new_err)
    }

    /// Decode token IDs back into text.
    #[pyo3(signature = (ids, skip_special_tokens = false))]
    fn decode(
        &self,
        ids: Vec<u32>,
        skip_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<String> {
        py.allow_threads(|| {
            self.inner
                .decode(&ids, skip_special_tokens)
                .map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)
    }

    /// Decode a batch of token ID sequences.
    #[pyo3(signature = (sentences, skip_special_tokens = false))]
    fn decode_batch(
        &self,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<String>> {
        py.allow_threads(|| {
            let refs: Vec<&[u32]> = sentences.iter().map(Vec::as_slice).collect();
            self.inner
                .decode_batch(&refs, skip_special_tokens)
                .map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)
    }

    // ── Vocabulary ────────────────────────────────────────────────────

    /// Look up the token ID for a string.
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Look up the string for a token ID.
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id).map(String::from)
    }

    /// Return the vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

}

impl PyTokenizer {
    /// Parse `json`, update the Rust post-processor in place, and cache the JSON.
    fn update_post_processor_json(&mut self, json: &str) -> PyResult<()> {
        use fastokens::json_structs::PostProcessorConfig;
        use fastokens::post_processors::PostProcessor;

        let value: Value = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("invalid post-processor JSON: {e}")))?;
        let config: PostProcessorConfig = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(format!("cannot parse post-processor: {e}")))?;
        let pp = PostProcessor::from_config(config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.set_post_processor(Some(pp));
        self.post_processor_json = Some(json.to_string());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `PyEncoding::pad` correctly fills `type_ids` with `pad_type_id` for
    /// padded positions.  This is the expected behaviour.
    #[test]
    fn encoding_pad_applies_pad_type_id() {
        let mut enc = PyEncoding::new(vec![10u32, 20, 30], None);
        // 3 real tokens → pad to length 5 with pad_type_id = 1
        enc.pad(5, "right", 0u32, 1u32, "[PAD]");

        assert_eq!(enc.ids,       vec![10u32, 20, 30, 0, 0]);
        assert_eq!(enc.attention_mask, vec![1u32, 1, 1, 0, 0]);
        assert_eq!(enc.type_ids,  vec![0u32, 0, 0, 1, 1],
            "padded positions should carry pad_type_id=1 in type_ids");
    }

    /// `encode_batch` goes through `pad_to`, which only extends `ids` and
    /// returns the attention mask.  `PyEncoding::make` is then called with the
    /// padded `ids`, and it initialises `type_ids` to all-zeros — so
    /// `pad_type_id` is silently ignored.
    ///
    /// This test reproduces that exact code-path and asserts the *correct*
    /// behaviour; it currently **fails**, documenting the bug.
    #[test]
    fn encode_batch_pad_type_id_applied_to_type_ids() {
        let pad_id = 0u32;
        let pad_type_id = 1u32;
        let n_real = 3usize;
        let target = 5usize;
        let deficit = target - n_real;

        // Reproduce what encode_batch + pad_to does today:
        //   1. encode → ids (3 real tokens)
        //   2. pad_to: extend ids with pad_id, build attention mask
        //   3. PyEncoding::make(ids, mask)   ← type_ids comes from here as all-zeros
        let mut ids = vec![10u32, 20, 30];
        ids.extend(vec![pad_id; deficit]);
        let mut mask = vec![1u32; n_real];
        mask.extend(vec![0u32; deficit]);
        let enc = PyEncoding::make(ids, mask);

        // The padded positions should carry pad_type_id — but pad_to never
        // passes pad_type_id into PyEncoding::make, so they are 0.
        assert_eq!(
            enc.type_ids[n_real..].to_vec(),
            vec![pad_type_id; deficit],
            "encode_batch must propagate pad_type_id into type_ids of padded positions"
        );
    }
}

// ---------------------------------------------------------------------------
// DecodeStream
// ---------------------------------------------------------------------------

/// Python binding for [`fastokens::DecodeStream`].
///
/// Drop-in replacement for `tokenizers.decoders.DecodeStream`. Accepts both a
/// bare `fastokens.Tokenizer` and any shim that stores one in `._fast`
/// (e.g. `_TokenizerShim`).
#[pyclass(name = "DecodeStream")]
struct PyDecodeStream {
    inner: fastokens::DecodeStream,
}

#[pymethods]
impl PyDecodeStream {
    #[new]
    #[pyo3(signature = (ids = None, skip_special_tokens = false))]
    fn new(ids: Option<Vec<u32>>, skip_special_tokens: bool) -> Self {
        Self {
            inner: fastokens::DecodeStream::new(ids.unwrap_or_default(), skip_special_tokens),
        }
    }

    #[pyo3(signature = (tokenizer, id))]
    fn step(
        &mut self,
        tokenizer: &Bound<'_, PyAny>,
        id: &Bound<'_, PyAny>,
        py: Python<'_>,
    ) -> PyResult<Option<String>> {
        let new_ids: Vec<u32> = if let Ok(single) = id.extract::<u32>() {
            vec![single]
        } else {
            id.extract::<Vec<u32>>()?
        };

        // Accept a PyTokenizer directly or any shim that stores one in ._fast.
        let py_tok: Py<PyTokenizer> = tokenizer
            .extract::<Py<PyTokenizer>>()
            .or_else(|_| tokenizer.getattr("_fast")?.extract::<Py<PyTokenizer>>())?;

        let tok = py_tok.borrow(py);
        self.inner.step(&tok.inner, new_ids).map_err(PyValueError::new_err)
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyPostProcessor>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyDecodeStream>()?;
    Ok(())
}
