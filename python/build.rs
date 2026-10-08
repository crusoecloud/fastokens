fn main() {
    // The `Py_LIMITED_API` / `Py_3_*` / `PyPy` cfgs pyo3 itself is built with, so
    // the bindings can pick the zero-copy string access the target allows.
    pyo3_build_config::use_pyo3_cfgs();
}
