use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

/// An LLM tokenizer backed by `tokenizer.json`.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: fastokens::Tokenizer,
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
        let inner = py
            .allow_threads(|| {
                fastokens::Tokenizer::from_file(Path::new(path)).map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Create a tokenizer from a raw JSON string for `tokenizer.json`.
    #[staticmethod]
    fn from_json_str(json: &str, py: Python<'_>) -> PyResult<Self> {
        let inner = py
            .allow_threads(|| {
                let value: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
                fastokens::Tokenizer::from_json(value).map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Download `tokenizer.json` from HuggingFace Hub for the given model
    /// (e.g. `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    #[staticmethod]
    fn from_model(model: &str, py: Python<'_>) -> PyResult<Self> {
        let inner = py
            .allow_threads(|| fastokens::Tokenizer::from_model(model).map_err(|e| e.to_string()))
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Run the full encoding pipeline.
    #[pyo3(signature = (input, add_special_tokens=false))]
    fn encode(
        &self,
        input: &str,
        add_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<u32>> {
        py.allow_threads(|| {
            self.inner
                .encode_with_special_tokens(input, add_special_tokens)
                .map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)
    }

    /// Encode a batch of inputs in parallel.
    #[pyo3(signature = (inputs, add_special_tokens=false))]
    fn encode_batch(
        &self,
        inputs: Vec<String>,
        add_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Vec<u32>>> {
        py.allow_threads(|| {
            self.inner
                .encode_batch(&inputs, add_special_tokens)
                .map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)
    }

    /// Decode token IDs back into text.
    #[pyo3(signature = (ids, skip_special_tokens=false))]
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
    #[pyo3(signature = (sentences, skip_special_tokens=false))]
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
