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

    /// Run the full encoding pipeline: split added tokens, normalize,
    /// pre-tokenize, tokenize and post-process the input string.
    fn encode(&self, input: &str, py: Python<'_>) -> PyResult<Vec<u32>> {
        py.allow_threads(|| self.inner.encode(input).map_err(|e| e.to_string()))
            .map_err(PyValueError::new_err)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
