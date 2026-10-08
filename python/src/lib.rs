use std::borrow::Cow;
use std::sync::RwLock;

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Fast conversions
// ---------------------------------------------------------------------------

/// Inputs at least this long are encoded with the GIL released. Shorter ones take
/// microseconds, less than a contended GIL handoff could cost them.
const GIL_RELEASE_MIN_BYTES: usize = 32 * 1024;

// `abi3-py39` builds (the published wheels): pyo3 has no borrowing `str` access
// there, because `PyUnicode_AsUTF8AndSize` only joined the stable ABI in 3.10.
// CPython has exported it since 3.3, so call it directly when the running
// interpreter is 3.10+ — except on Windows, where abi3 modules link against
// `python3.dll`, which exports it only from 3.10 (a 3.9 import would fail).
#[cfg(all(Py_LIMITED_API, not(Py_3_10), not(windows)))]
unsafe extern "C" {
    fn PyUnicode_AsUTF8AndSize(
        unicode: *mut ffi::PyObject,
        size: *mut ffi::Py_ssize_t,
    ) -> *const std::os::raw::c_char;
}

/// The UTF-8 text of a Python `str`, borrowed from the string's own UTF-8 buffer
/// wherever the build and interpreter allow; otherwise pyo3 encodes it into a
/// temporary `bytes` and copies that into a `String`, on every call.
fn utf8<'a>(s: &'a Bound<'_, PyString>) -> PyResult<Cow<'a, str>> {
    #[cfg(any(Py_3_10, not(Py_LIMITED_API)))]
    return s.to_str().map(Cow::Borrowed);

    #[cfg(all(Py_LIMITED_API, not(Py_3_10)))]
    {
        #[cfg(not(windows))]
        {
            static ZERO_COPY: GILOnceCell<bool> = GILOnceCell::new();
            let py = s.py();
            if *ZERO_COPY.get_or_init(py, || py.version_info() >= (3, 10)) {
                let mut size: ffi::Py_ssize_t = 0;
                // SAFETY: `s` is a live `str`; the returned buffer is owned by it and
                // immutable for its lifetime, which outlives the `'a` borrow.
                let data = unsafe { PyUnicode_AsUTF8AndSize(s.as_ptr(), &mut size) };
                if data.is_null() {
                    return Err(PyErr::fetch(py));
                }
                // SAFETY: CPython guarantees the buffer is valid UTF-8 of `size` bytes.
                return Ok(Cow::Borrowed(unsafe {
                    std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                        data.cast(),
                        size as usize,
                    ))
                }));
            }
        }
        s.to_cow()
    }
}

/// Python `int` objects for token ids `0..len`, created on first use and shared
/// by every tokenizer's encodings for the life of the process, so materializing
/// `Encoding.ids` bumps one refcount per token instead of allocating an `int` for
/// each. Holds owned references, never released (a bounded table: one entry per
/// id of the largest vocabulary loaded), as raw pointers only touched with the GIL
/// held.
static ID_OBJECTS: RwLock<Vec<usize>> = RwLock::new(Vec::new());

/// Extend [`ID_OBJECTS`] to cover ids `0..len`.
fn ensure_id_objects(py: Python<'_>, len: usize) -> PyResult<()> {
    if ID_OBJECTS.read().unwrap().len() >= len {
        return Ok(());
    }
    let mut table = ID_OBJECTS.write().unwrap();
    while table.len() < len {
        // SAFETY: the GIL is held (`py`).
        let p = unsafe { ffi::PyLong_FromUnsignedLong(table.len() as std::os::raw::c_ulong) };
        if p.is_null() {
            return Err(PyErr::fetch(py));
        }
        table.push(p as usize);
    }
    Ok(())
}

/// Where a list's item array lives, if this interpreter lays `PyListObject` out
/// as CPython does (`PyObject_VAR_HEAD` then `PyObject **ob_item`), which is
/// verified once against a real list. `None` when the check fails.
fn list_items_offset(py: Python<'_>) -> Option<usize> {
    static OFFSET: GILOnceCell<Option<usize>> = GILOnceCell::new();
    *OFFSET.get_or_init(py, || {
        let off = 3 * std::mem::size_of::<usize>();
        // SAFETY: plain C-API calls on objects created here; the read at `off` stays
        // inside the list object, whose size is at least that of `PyVarObject`
        // plus the item pointer on every CPython.
        unsafe {
            let probe = ffi::PyLong_FromLong(123_456_789);
            let list = ffi::PyList_New(1);
            if probe.is_null() || list.is_null() || ffi::PyList_SetItem(list, 0, probe) != 0 {
                ffi::PyErr_Clear();
                return None;
            }
            let items = *((list as *const u8).add(off) as *const *const *mut ffi::PyObject);
            let ok = !items.is_null() && *items == probe;
            ffi::Py_DecRef(list);
            ok.then_some(off)
        }
    })
}

/// `values` as a Python list of ints, taking the int objects from
/// [`ID_OBJECTS`] where it covers them.
///
/// The fast path stores straight into the list's item array (see
/// [`list_items_offset`]) and bumps the cached ints' refcounts in place, as C
/// extensions built against the Python 3.9 limited API do with `Py_INCREF`; that
/// avoids two C-API calls per token. Ids below 257 are CPython's shared small ints
/// (immortal from 3.12), so they take the regular `Py_INCREF`.
fn u32_list<'py>(py: Python<'py>, values: &[u32]) -> PyResult<Bound<'py, PyList>> {
    let table = ID_OBJECTS.read().unwrap();
    let cached: &[usize] = &table;
    if let (Some(off), false) = (list_items_offset(py), cached.is_empty()) {
        if values.len() >= PARALLEL_LIST_MIN && fastokens::fanout::threads() > 1 {
            return u32_list_parallel(py, values, cached, off);
        }
        // SAFETY: a fresh list of `values.len()` NULL slots; each slot is written
        // exactly once with an owned reference, and on error the list is dropped
        // (NULL slots are allowed). `cached` holds live ints we own references to.
        unsafe {
            let list = ffi::PyList_New(values.len() as ffi::Py_ssize_t);
            if list.is_null() {
                return Err(PyErr::fetch(py));
            }
            let list = Bound::from_owned_ptr(py, list);
            let items = *((list.as_ptr() as *const u8).add(off) as *const *mut *mut ffi::PyObject);
            for (i, &v) in values.iter().enumerate() {
                let item = match cached.get(v as usize) {
                    Some(&o) if v > 256 => {
                        let p = o as *mut ffi::PyObject;
                        *(p as *mut ffi::Py_ssize_t) += 1;
                        p
                    }
                    Some(&o) => {
                        let p = o as *mut ffi::PyObject;
                        ffi::Py_INCREF(p);
                        p
                    }
                    None => {
                        let p = ffi::PyLong_FromUnsignedLong(v as std::os::raw::c_ulong);
                        if p.is_null() {
                            return Err(PyErr::fetch(py));
                        }
                        p
                    }
                };
                *items.add(i) = item;
            }
            return Ok(list.downcast_into_unchecked());
        }
    }
    // SAFETY: a fresh list of `values.len()` NULL slots, each filled exactly once
    // with an owned reference (`PyList_SetItem` steals it) before it is returned;
    // on an early error the list is dropped, which tolerates NULL slots.
    unsafe {
        let list = ffi::PyList_New(values.len() as ffi::Py_ssize_t);
        if list.is_null() {
            return Err(PyErr::fetch(py));
        }
        let list = Bound::from_owned_ptr(py, list);
        for (i, &v) in values.iter().enumerate() {
            let item = match cached.get(v as usize) {
                Some(&o) => {
                    let p = o as *mut ffi::PyObject;
                    ffi::Py_INCREF(p);
                    p
                }
                None => {
                    let p = ffi::PyLong_FromUnsignedLong(v as std::os::raw::c_ulong);
                    if p.is_null() {
                        return Err(PyErr::fetch(py));
                    }
                    p
                }
            };
            ffi::PyList_SetItem(list.as_ptr(), i as ffi::Py_ssize_t, item);
        }
        Ok(list.downcast_into_unchecked())
    }
}

/// Lists of at least this many ids are filled by the worker pool.
const PARALLEL_LIST_MIN: usize = 64 * 1024;

thread_local! {
    /// Per-thread occurrence counters for [`u32_list_parallel`], indexed by id,
    /// and the ids they hold nonzero counts for (to reset only those).
    static ID_COUNTS: std::cell::RefCell<(Vec<u32>, Vec<u32>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// [`u32_list`]'s fast path for a long list, spread over the worker pool (see
/// [`fill_lists_parallel`]).
fn u32_list_parallel<'py>(
    py: Python<'py>,
    values: &[u32],
    table: &[usize],
    off: usize,
) -> PyResult<Bound<'py, PyList>> {
    // SAFETY: a fresh list of `values.len()` NULL slots, which the fill below
    // makes exactly the list of `values`; on error the list is dropped, which
    // tolerates the slots still NULL.
    unsafe {
        let list = ffi::PyList_New(values.len() as ffi::Py_ssize_t);
        if list.is_null() {
            return Err(PyErr::fetch(py));
        }
        let list = Bound::from_owned_ptr(py, list);
        let items = *((list.as_ptr() as *const u8).add(off) as *const *mut *mut ffi::PyObject);
        fill_lists_parallel(py, table, &[(values, items)])?;
        Ok(list.downcast_into_unchecked())
    }
}

/// Encodings together holding at least this many ids get their `ids` lists
/// built by `encode_batch`, on the pool, instead of one by one on access.
const PREBUILD_LISTS_MIN: usize = 64 * 1024;

/// The `ids` lists of `encodings`, built together by the worker pool, or `None`s
/// when the batch is too small for that to pay (or the fast path is off).
fn prebuild_lists(py: Python<'_>, encodings: &[EncodingData]) -> PyResult<Vec<Option<Py<PyList>>>> {
    let total: usize = encodings.iter().map(|e| e.ids.len()).sum();
    let table = ID_OBJECTS.read().unwrap();
    let off = list_items_offset(py);
    let (Some(off), false, true) = (
        off,
        table.is_empty(),
        total >= PREBUILD_LISTS_MIN && fastokens::fanout::threads() > 1,
    ) else {
        return Ok(encodings.iter().map(|_| None).collect());
    };
    // SAFETY: fresh lists of NULL slots, each filled exactly by the fill below;
    // on error they are dropped, which tolerates the slots still NULL.
    unsafe {
        let mut lists = Vec::with_capacity(encodings.len());
        let mut segs = Vec::with_capacity(encodings.len());
        for e in encodings {
            let list = ffi::PyList_New(e.ids.len() as ffi::Py_ssize_t);
            if list.is_null() {
                return Err(PyErr::fetch(py));
            }
            let list: Py<PyList> = Bound::from_owned_ptr(py, list)
                .downcast_into_unchecked()
                .unbind();
            segs.push((
                &e.ids[..],
                *((list.as_ptr() as *const u8).add(off) as *const *mut *mut ffi::PyObject),
            ));
            lists.push(list);
        }
        fill_lists_parallel(py, &table, &segs)?;
        Ok(lists.into_iter().map(Some).collect())
    }
}

/// Fill fresh lists — each `(values, item array)`, the array having
/// `values.len()` NULL slots — with the ints of `values`, on the worker pool.
///
/// Each task takes an equal share of all the ids and stores the shared int
/// objects of [`ID_OBJECTS`] straight into the item arrays — plain memory writes
/// into objects no Python code can reach yet; the workers call no Python API —
/// counting how often it used each one, and then adds those counts to the
/// refcounts: one atomic update per distinct id per task instead of one per
/// token. Atomic because the tasks run concurrently; nothing else can touch these
/// refcounts meanwhile, as the caller holds the GIL. The caller settles CPython's
/// shared small ints and fills in any id the table does not cover.
///
/// # Safety
/// Every item array must be a fresh list's, of exactly its values' length.
unsafe fn fill_lists_parallel(
    py: Python<'_>,
    table: &[usize],
    segs: &[(&[u32], *mut *mut ffi::PyObject)],
) -> PyResult<()> {
    /// The segments, shared by the tasks, which write disjoint slots.
    struct Segs<'a>(&'a [(&'a [u32], *mut *mut ffi::PyObject)]);
    // SAFETY: tasks write disjoint slots and nothing reads them until all finish.
    unsafe impl Sync for Segs<'_> {}
    impl Segs<'_> {
        fn get(&self, k: usize) -> (&[u32], *mut *mut ffi::PyObject) {
            self.0[k]
        }
    }

    let mut starts = Vec::with_capacity(segs.len() + 1);
    let mut total = 0;
    for (v, _) in segs {
        starts.push(total);
        total += v.len();
    }
    starts.push(total);
    let tasks = fastokens::fanout::threads()
        .min(total / (PARALLEL_LIST_MIN / 8))
        .max(1);
    let per = total.div_ceil(tasks);
    let shared = Segs(segs);
    let shared = &shared;
    let starts = &starts;
    // Per task: (id, count) of the small ints it stored, and the (segment, index)
    // slots it could not fill.
    let parts = fastokens::fanout::map(tasks, |t| {
        let (lo, hi) = (t * per, ((t + 1) * per).min(total));
        ID_COUNTS.with(|c| {
            let (counts, touched) = &mut *c.borrow_mut();
            if counts.len() < table.len() {
                counts.resize(table.len(), 0);
            }
            let mut missing = Vec::new();
            // The segment holding global index `lo`, then on through `hi`.
            let mut k = starts.partition_point(|&s| s <= lo) - 1;
            let mut g = lo;
            while g < hi {
                while starts[k + 1] <= g {
                    k += 1;
                }
                let (values, items) = shared.get(k);
                let end = hi.min(starts[k + 1]);
                for i in (g - starts[k])..(end - starts[k]) {
                    let v = values[i] as usize;
                    match table.get(v) {
                        Some(&o) => {
                            // SAFETY: `i < values.len()`, the item array's length.
                            unsafe { *items.add(i) = o as *mut ffi::PyObject };
                            // SAFETY: `counts` covers the table.
                            let c = unsafe { counts.get_unchecked_mut(v) };
                            if *c == 0 {
                                touched.push(v as u32);
                            }
                            *c += 1;
                        }
                        None => missing.push((k, i)),
                    }
                }
                g = end;
            }
            let mut small = Vec::new();
            for &v in touched.iter() {
                let c = std::mem::take(&mut counts[v as usize]);
                if v > 256 {
                    // SAFETY: a live int we own a reference to; see above.
                    unsafe {
                        std::sync::atomic::AtomicIsize::from_ptr(table[v as usize] as *mut isize)
                            .fetch_add(c as isize, std::sync::atomic::Ordering::Relaxed);
                    }
                } else {
                    small.push((v, c));
                }
            }
            touched.clear();
            (small, missing)
        })
    });
    // Every stored slot's reference first, so the lists are consistent (NULL or
    // owned) even if creating a missing int fails below.
    for (small, _) in &parts {
        for &(v, c) in small {
            // SAFETY: ids up to 256 are CPython's shared small ints.
            unsafe { incref_small_int(py, table[v as usize] as *mut ffi::PyObject, c as usize) };
        }
    }
    for (_, missing) in parts {
        for (k, i) in missing {
            let (values, items) = segs[k];
            // SAFETY: the GIL is held; `i` indexes segment `k`'s item array.
            unsafe {
                let p = ffi::PyLong_FromUnsignedLong(values[i] as std::os::raw::c_ulong);
                if p.is_null() {
                    return Err(PyErr::fetch(py));
                }
                *items.add(i) = p;
            }
        }
    }
    Ok(())
}

/// A Python list of `n` copies of the int `value` (the side arrays' defaults:
/// all ones, all zeros).
fn repeat_list<'py>(py: Python<'py>, value: u32, n: usize) -> PyResult<Bound<'py, PyList>> {
    let obj = value.into_pyobject(py)?;
    // SAFETY: a fresh list of `n` NULL slots, each set to `obj` below with the
    // reference it needs; on an early error the list is dropped, which tolerates
    // NULL slots.
    unsafe {
        let list = ffi::PyList_New(n as ffi::Py_ssize_t);
        if list.is_null() {
            return Err(PyErr::fetch(py));
        }
        let list = Bound::from_owned_ptr(py, list);
        let p = obj.as_ptr();
        if let Some(off) = list_items_offset(py) {
            let items = *((list.as_ptr() as *const u8).add(off) as *const *mut *mut ffi::PyObject);
            std::slice::from_raw_parts_mut(items, n).fill(p);
            incref_small_int(py, p, n);
        } else {
            for i in 0..n {
                ffi::Py_INCREF(p);
                ffi::PyList_SetItem(list.as_ptr(), i as ffi::Py_ssize_t, p);
            }
        }
        Ok(list.downcast_into_unchecked())
    }
}

/// Take `n` references to one of CPython's shared small ints (`-5..=256`): from
/// 3.12 they are immortal and need none; before, they are ordinary objects and
/// the count is simply added (the GIL is held).
///
/// # Safety
/// `p` must be a small int.
unsafe fn incref_small_int(py: Python<'_>, p: *mut ffi::PyObject, n: usize) {
    static IMMORTAL: GILOnceCell<bool> = GILOnceCell::new();
    if !*IMMORTAL.get_or_init(py, || py.version_info() >= (3, 12)) {
        // SAFETY: a live object whose refcount only GIL holders touch.
        unsafe { *(p as *mut ffi::Py_ssize_t) += n as ffi::Py_ssize_t };
    }
}

// ---------------------------------------------------------------------------
// PyEncoding
// ---------------------------------------------------------------------------

/// Minimal stand-in for `tokenizers.Encoding`.
///
/// Returned directly by `Tokenizer.encode` and `Tokenizer.encode_batch` so
/// no Python-side wrapping is needed.  Fields that `fastokens` does not
/// track (`tokens`, `offsets`, `sequence_ids`, `word_ids`) have getters that
/// raise `NotImplementedError` to match the HuggingFace API surface.
///
/// Only `ids` is stored eagerly. The per-token side arrays are `None` while they
/// hold their defaults for the current length — all ones for `attention_mask`,
/// zeros for `type_ids` / `special_tokens_mask` — and are materialized only when
/// set, padded, or merged.
#[pyclass(name = "Encoding")]
pub struct PyEncoding {
    data: EncodingData,
    /// `ids` as a list built ahead of time (by `encode_batch`, in parallel with
    /// the others), handed out by the first `ids` access. Cleared by any edit.
    prebuilt_ids: Option<Py<PyList>>,
}

/// An encoding's contents (see [`PyEncoding`]), free of Python objects.
pub struct EncodingData {
    pub ids: Vec<u32>,
    attention_mask: Option<Vec<u32>>,
    type_ids: Option<Vec<u32>>,
    special_tokens_mask: Option<Vec<u32>>,
    pub n_sequences: usize,
    // Backing storage for set-only properties (`None`: the defaults).
    _sequence_ids: Option<Vec<Option<i64>>>,
    _word_ids: Option<Vec<Option<i64>>>,
}

impl From<EncodingData> for PyEncoding {
    fn from(data: EncodingData) -> Self {
        Self {
            data,
            prebuilt_ids: None,
        }
    }
}

impl std::ops::Deref for PyEncoding {
    type Target = EncodingData;
    fn deref(&self) -> &EncodingData {
        &self.data
    }
}

impl std::ops::DerefMut for PyEncoding {
    fn deref_mut(&mut self) -> &mut EncodingData {
        self.prebuilt_ids = None;
        &mut self.data
    }
}

impl EncodingData {
    pub fn make(ids: Vec<u32>, attention_mask: Option<Vec<u32>>) -> Self {
        Self {
            ids,
            attention_mask,
            type_ids: None,
            special_tokens_mask: None,
            n_sequences: 1,
            _sequence_ids: None,
            _word_ids: None,
        }
    }

    fn attention_mask_vec(&self) -> Cow<'_, [u32]> {
        self.attention_mask
            .as_deref()
            .map_or_else(|| Cow::Owned(vec![1u32; self.ids.len()]), Cow::Borrowed)
    }

    fn type_ids_vec(&self) -> Cow<'_, [u32]> {
        self.type_ids
            .as_deref()
            .map_or_else(|| Cow::Owned(vec![0u32; self.ids.len()]), Cow::Borrowed)
    }

    fn special_tokens_mask_vec(&self) -> Cow<'_, [u32]> {
        self.special_tokens_mask
            .as_deref()
            .map_or_else(|| Cow::Owned(vec![0u32; self.ids.len()]), Cow::Borrowed)
    }

    fn sequence_ids_vec(&self) -> Cow<'_, [Option<i64>]> {
        self._sequence_ids
            .as_deref()
            .map_or_else(|| Cow::Owned(vec![Some(0); self.ids.len()]), Cow::Borrowed)
    }

    fn word_ids_vec(&self) -> Cow<'_, [Option<i64>]> {
        self._word_ids
            .as_deref()
            .map_or_else(|| Cow::Owned(vec![None; self.ids.len()]), Cow::Borrowed)
    }

    /// Replace every side array defaulted to `None` by its explicit value, before
    /// an edit that makes them diverge from the defaults.
    fn materialize(&mut self) {
        self.attention_mask = Some(self.attention_mask_vec().into_owned());
        self.type_ids = Some(self.type_ids_vec().into_owned());
        self.special_tokens_mask = Some(self.special_tokens_mask_vec().into_owned());
        self._sequence_ids = Some(self.sequence_ids_vec().into_owned());
        self._word_ids = Some(self.word_ids_vec().into_owned());
    }

    fn apply_slice(&mut self, start: usize, end: usize) {
        self.ids = self.ids[start..end].to_vec();
        // Defaults stay defaults for the new length.
        for v in [
            &mut self.attention_mask,
            &mut self.type_ids,
            &mut self.special_tokens_mask,
        ]
        .into_iter()
        .flatten()
        {
            *v = v[start..end].to_vec();
        }
        for v in [&mut self._sequence_ids, &mut self._word_ids]
            .into_iter()
            .flatten()
        {
            *v = v[start..end].to_vec();
        }
    }

    fn extend_right(&mut self, pad_id: u32, pad_type_id: u32, count: usize) {
        self.materialize();
        self.ids.extend(vec![pad_id; count]);
        let (mask, type_ids, special, seq_ids, word_ids) = self.sides_mut();
        mask.extend(vec![0u32; count]);
        type_ids.extend(vec![pad_type_id; count]);
        special.extend(vec![0u32; count]);
        seq_ids.extend(vec![None; count]);
        word_ids.extend(vec![None; count]);
    }

    fn extend_left(&mut self, pad_id: u32, pad_type_id: u32, count: usize) {
        self.materialize();
        fn prepend<T: Clone>(v: &mut Vec<T>, value: T, count: usize) {
            v.splice(0..0, std::iter::repeat_n(value, count));
        }
        prepend(&mut self.ids, pad_id, count);
        let (mask, type_ids, special, seq_ids, word_ids) = self.sides_mut();
        prepend(mask, 0u32, count);
        prepend(type_ids, pad_type_id, count);
        prepend(special, 0u32, count);
        prepend(seq_ids, None, count);
        prepend(word_ids, None, count);
    }

    fn truncate_to(&mut self, max_length: usize, direction: &str) {
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

    fn pad_to(&mut self, length: usize, direction: &str, pad_id: u32, pad_type_id: u32) {
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

    /// The materialized side arrays (call [`Self::materialize`] first).
    #[allow(clippy::type_complexity)]
    fn sides_mut(
        &mut self,
    ) -> (
        &mut Vec<u32>,
        &mut Vec<u32>,
        &mut Vec<u32>,
        &mut Vec<Option<i64>>,
        &mut Vec<Option<i64>>,
    ) {
        (
            self.attention_mask.as_mut().expect("materialized"),
            self.type_ids.as_mut().expect("materialized"),
            self.special_tokens_mask.as_mut().expect("materialized"),
            self._sequence_ids.as_mut().expect("materialized"),
            self._word_ids.as_mut().expect("materialized"),
        )
    }
}

#[pymethods]
impl PyEncoding {
    #[new]
    #[pyo3(signature = (ids, attention_mask = None))]
    fn new(ids: Vec<u32>, attention_mask: Option<Vec<u32>>) -> Self {
        EncodingData::make(ids, attention_mask).into()
    }

    #[getter(ids)]
    fn py_ids<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match self.prebuilt_ids.take() {
            Some(list) => Ok(list.into_bound(py)),
            None => u32_list(py, &self.data.ids),
        }
    }

    #[getter(n_sequences)]
    fn py_n_sequences(&self) -> usize {
        self.n_sequences
    }
    #[setter(n_sequences)]
    fn set_py_n_sequences(&mut self, value: usize) {
        self.data.n_sequences = value;
    }
    #[setter(ids)]
    fn set_py_ids(&mut self, value: Vec<u32>) {
        self.ids = value;
    }

    #[getter(attention_mask)]
    fn py_attention_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match &self.attention_mask {
            Some(v) => u32_list(py, v),
            None => repeat_list(py, 1, self.ids.len()),
        }
    }
    #[setter(attention_mask)]
    fn set_py_attention_mask(&mut self, value: Vec<u32>) {
        self.attention_mask = Some(value);
    }

    #[getter(type_ids)]
    fn py_type_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match &self.type_ids {
            Some(v) => u32_list(py, v),
            None => repeat_list(py, 0, self.ids.len()),
        }
    }
    #[setter(type_ids)]
    fn set_py_type_ids(&mut self, value: Vec<u32>) {
        self.type_ids = Some(value);
    }

    #[getter(special_tokens_mask)]
    fn py_special_tokens_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        match &self.special_tokens_mask {
            Some(v) => u32_list(py, v),
            None => repeat_list(py, 0, self.ids.len()),
        }
    }
    #[setter(special_tokens_mask)]
    fn set_py_special_tokens_mask(&mut self, value: Vec<u32>) {
        self.special_tokens_mask = Some(value);
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
        self._sequence_ids = Some(value);
    }

    #[getter]
    fn word_ids(&self) -> PyResult<Vec<Option<i64>>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }
    #[setter]
    fn set_word_ids(&mut self, value: Vec<Option<i64>>) {
        self._word_ids = Some(value);
    }

    #[getter]
    fn words(&self) -> PyResult<Vec<Option<i64>>> {
        Err(PyNotImplementedError::new_err(
            "fastokens does not track word IDs",
        ))
    }
    #[setter]
    fn set_words(&mut self, value: Vec<Option<i64>>) {
        self._word_ids = Some(value);
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
        self._sequence_ids = Some(vec![Some(sequence_id); n]);
    }

    // -- Positional mapping (all raise NotImplementedError) -------------

    #[pyo3(signature = (char_pos, sequence_index = 0))]
    fn char_to_token(&self, char_pos: usize, sequence_index: usize) -> PyResult<Option<usize>> {
        let _ = (char_pos, sequence_index);
        Err(PyNotImplementedError::new_err(
            "fastokens does not track character offsets",
        ))
    }

    #[pyo3(signature = (char_pos, sequence_index = 0))]
    fn char_to_word(&self, char_pos: usize, sequence_index: usize) -> PyResult<Option<usize>> {
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
        self.truncate_to(max_length, direction);
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
        self.pad_to(length, direction, pad_id, pad_type_id);
    }

    // -- Merge ----------------------------------------------------------

    #[staticmethod]
    #[pyo3(signature = (encodings, growing_offsets = true))]
    fn merge(py: Python<'_>, encodings: Vec<Py<PyEncoding>>, growing_offsets: bool) -> PyEncoding {
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
            attention_mask.extend_from_slice(&enc.attention_mask_vec());
            type_ids.extend_from_slice(&enc.type_ids_vec());
            special_tokens_mask.extend_from_slice(&enc.special_tokens_mask_vec());
            n_sequences += enc.n_sequences;
            seq_ids.extend_from_slice(&enc.sequence_ids_vec());
            word_ids.extend_from_slice(&enc.word_ids_vec());
        }

        EncodingData {
            ids,
            attention_mask: Some(attention_mask),
            type_ids: Some(type_ids),
            special_tokens_mask: Some(special_tokens_mask),
            n_sequences,
            _sequence_ids: Some(seq_ids),
            _word_ids: Some(word_ids),
        }
        .into()
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

fn build_encoding(ids: Vec<u32>, pad: Option<&PaddingParams>, target: usize) -> EncodingData {
    let mut enc = EncodingData::make(ids, None);
    if let Some(p) = pad {
        enc.pad_to(target, &p.direction, p.pad_id, p.pad_type_id);
    }
    enc
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

/// Mutable state guarded by `PyTokenizer::state`.
///
/// All read paths (encode/decode/getters) hold a read lock; mutators
/// (`enable_truncation`, `set_post_processor`, …) hold a write lock so they
/// cannot race with concurrent reads when the GIL is released.
struct TokenizerState {
    inner: fastokens::Tokenizer,
    trunc: Option<TruncationParams>,
    pad: Option<PaddingParams>,
    /// Cached JSON of the current post-processor (for the getter).
    post_processor_json: Option<String>,
}

fn tokenizer_options(
    pcre2_match_limit: Option<u32>,
    pcre2_depth_limit: Option<u32>,
    pcre2_heap_limit: Option<u32>,
    pcre2_max_jit_stack_size: Option<usize>,
) -> fastokens::TokenizerOptions {
    fastokens::TokenizerOptions {
        pcre2_limits: fastokens::Pcre2Limits {
            match_limit: pcre2_match_limit,
            depth_limit: pcre2_depth_limit,
            heap_limit: pcre2_heap_limit,
            max_jit_stack_size: pcre2_max_jit_stack_size,
        },
    }
}

impl TokenizerState {
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

    fn single_pad_target(&self, n: usize) -> usize {
        let Some(ref p) = self.pad else { return n };
        let base = p.length.unwrap_or(n).max(n);
        match p.pad_to_multiple_of {
            Some(m) if m > 0 => base.div_ceil(m) * m,
            _ => base,
        }
    }

    fn build_single_encoding(&self, mut ids: Vec<u32>) -> EncodingData {
        self.do_truncate(&mut ids);
        let target = self.single_pad_target(ids.len());
        build_encoding(ids, self.pad.as_ref(), target)
    }

    /// Parse `json`, update the Rust post-processor in place, and cache the JSON.
    fn update_post_processor_json(&mut self, json: &str) -> PyResult<()> {
        use fastokens::json_structs::PostProcessorConfig;
        use fastokens::post_processors::PostProcessor;

        let value: Value = serde_json::from_str(json)
            .map_err(|e| PyValueError::new_err(format!("invalid post-processor JSON: {e}")))?;
        let config: PostProcessorConfig = serde_json::from_value(value)
            .map_err(|e| PyValueError::new_err(format!("cannot parse post-processor: {e}")))?;
        let pp =
            PostProcessor::from_config(config).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.set_post_processor(Some(pp));
        self.post_processor_json = Some(json.to_string());
        Ok(())
    }
}

/// A token to add to the vocabulary, extracted from any object exposing the
/// attributes of a `tokenizers.AddedToken` (which `transformers` hands to
/// `add_tokens`).
#[derive(FromPyObject)]
struct PyNewToken {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

impl From<PyNewToken> for fastokens::NewToken {
    fn from(token: PyNewToken) -> Self {
        Self {
            content: token.content,
            single_word: token.single_word,
            lstrip: token.lstrip,
            rstrip: token.rstrip,
            normalized: token.normalized,
            special: token.special,
        }
    }
}

fn added_token_policy(split_special_tokens: bool) -> fastokens::AddedTokenPolicy {
    if split_special_tokens {
        fastokens::AddedTokenPolicy::SkipSpecial
    } else {
        fastokens::AddedTokenPolicy::All
    }
}

/// An LLM tokenizer backed by `tokenizer.json`.
// Python aligns object memory to 16 bytes only; a pyclass needing more would be
// misaligned (see `PyTokenizer::state`).
const _: () = {
    assert!(std::mem::align_of::<PyTokenizer>() <= 16);
    assert!(std::mem::align_of::<PyEncoding>() <= 16);
    assert!(std::mem::align_of::<PyPostProcessor>() <= 16);
    assert!(std::mem::align_of::<PyDecodeStream>() <= 16);
};

#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    /// Boxed: the tokenizer holds over-aligned values (e.g. `memchr`'s AVX2
    /// searchers, 32-byte aligned), but Python allocates a pyclass object with
    /// only 16-byte alignment, so storing them inline would misplace them — and
    /// an aligned SIMD load of one then faults. The Rust heap honors alignment.
    state: Box<RwLock<TokenizerState>>,
    /// Set once [`ID_OBJECTS`] covers this tokenizer's ids.
    id_objs_ready: GILOnceCell<()>,
}

impl PyTokenizer {
    fn with_state(state: TokenizerState) -> Self {
        Self {
            state: Box::new(RwLock::new(state)),
            id_objs_ready: GILOnceCell::new(),
        }
    }

    /// Make [`ID_OBJECTS`] cover the vocabulary plus headroom for id gaps and
    /// added tokens (ids past its end are simply converted one by one).
    fn ensure_id_objects(&self, py: Python<'_>) -> PyResult<()> {
        self.id_objs_ready
            .get_or_try_init(py, || {
                ensure_id_objects(py, self.read().inner.vocab_size() + 4096)
            })
            .map(|_| ())
    }

    /// Wrap encoded ids, applying truncation / padding, for Python.
    fn single_encoding(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<Py<PyEncoding>> {
        self.ensure_id_objects(py)?;
        let data = self.read().build_single_encoding(ids);
        // A long result's list is built now, while the pool is still warm from
        // encoding it, rather than on the first `ids` read.
        let prebuilt_ids = prebuild_lists(py, std::slice::from_ref(&data))?
            .pop()
            .flatten();
        Py::new(py, PyEncoding { data, prebuilt_ids })
    }

    /// Collect the elements of a Python id sequence that are integers
    /// representable as `i64`, skipping the rest.
    ///
    /// The fast path in [`Self::extract_decode_ids`] handles a clean list of
    /// in-range `u32`s. This is the fallback: it drops anything that cannot be
    /// a token id — negatives, values above `u32::MAX`, ints too large for
    /// `i64`, and non-integers such as floats (`inf`/`-inf`/`nan`), any of
    /// which would otherwise abort the whole decode.
    fn extract_valid_decode_ids(ids: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
        let iter = ids.try_iter()?;
        let mut out = Vec::with_capacity(ids.len().unwrap_or(0));
        for item in iter {
            if let Ok(value) = item?.extract::<i64>()
                && (0..=u32::MAX as i64).contains(&value)
            {
                out.push(value as u32);
            }
        }
        Ok(out)
    }

    fn extract_decode_ids(ids: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
        // Fast path: a clean list of in-range u32s extracts in one bulk call.
        // Any rejected element (negative, > u32::MAX, > i64, or a non-integer
        // such as a float) drops to the element-wise fallback, which skips the
        // offending ids instead of failing the whole decode.
        match ids.extract::<Vec<u32>>() {
            Ok(ids) => Ok(ids),
            Err(_) => Self::extract_valid_decode_ids(ids),
        }
    }

    fn extract_decode_batch_ids(sentences: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<u32>>> {
        match sentences.extract::<Vec<Vec<u32>>>() {
            Ok(sentences) => Ok(sentences),
            Err(_) => {
                let iter = sentences.try_iter()?;
                let mut out = Vec::with_capacity(sentences.len().unwrap_or(0));
                for seq in iter {
                    out.push(Self::extract_valid_decode_ids(&seq?)?);
                }
                Ok(out)
            }
        }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, TokenizerState> {
        self.state.read().expect("PyTokenizer state lock poisoned")
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, TokenizerState> {
        self.state.write().expect("PyTokenizer state lock poisoned")
    }

    /// Build from a raw JSON string, extracting the post-processor field so
    /// the getter can return it without needing to re-serialize.
    fn build_from_str(
        json: &str,
        py: Python<'_>,
        options: fastokens::TokenizerOptions,
    ) -> PyResult<Self> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let post_processor_json = value
            .get("post_processor")
            .filter(|v| !v.is_null())
            .map(|v| v.to_string());
        let inner = py
            .allow_threads(|| {
                fastokens::Tokenizer::from_json_with_options(value, options)
                    .map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self::with_state(TokenizerState {
            inner,
            trunc: None,
            pad: None,
            post_processor_json,
        }))
    }
}

#[pymethods]
impl PyTokenizer {
    /// Download `tokenizer.json` from HuggingFace Hub for the given model
    /// (e.g. `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    ///
    /// (This is an alias for Tokenizer.from_model)
    #[new]
    #[pyo3(signature = (model, pcre2_match_limit = None, pcre2_depth_limit = None, pcre2_heap_limit = None, pcre2_max_jit_stack_size = None))]
    fn new(
        model: &str,
        pcre2_match_limit: Option<u32>,
        pcre2_depth_limit: Option<u32>,
        pcre2_heap_limit: Option<u32>,
        pcre2_max_jit_stack_size: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        Self::from_model(
            model,
            pcre2_match_limit,
            pcre2_depth_limit,
            pcre2_heap_limit,
            pcre2_max_jit_stack_size,
            py,
        )
    }

    /// Create a tokenizer from a `tokenizer.json` file.
    #[staticmethod]
    #[pyo3(signature = (path, pcre2_match_limit = None, pcre2_depth_limit = None, pcre2_heap_limit = None, pcre2_max_jit_stack_size = None))]
    fn from_file(
        path: &str,
        pcre2_match_limit: Option<u32>,
        pcre2_depth_limit: Option<u32>,
        pcre2_heap_limit: Option<u32>,
        pcre2_max_jit_stack_size: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("cannot read {path}: {e}")))?;
        Self::build_from_str(
            &json,
            py,
            tokenizer_options(
                pcre2_match_limit,
                pcre2_depth_limit,
                pcre2_heap_limit,
                pcre2_max_jit_stack_size,
            ),
        )
    }

    /// Create a tokenizer from a raw JSON string for `tokenizer.json`.
    #[staticmethod]
    #[pyo3(signature = (json, pcre2_match_limit = None, pcre2_depth_limit = None, pcre2_heap_limit = None, pcre2_max_jit_stack_size = None))]
    fn from_json_str(
        json: &str,
        pcre2_match_limit: Option<u32>,
        pcre2_depth_limit: Option<u32>,
        pcre2_heap_limit: Option<u32>,
        pcre2_max_jit_stack_size: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        Self::build_from_str(
            json,
            py,
            tokenizer_options(
                pcre2_match_limit,
                pcre2_depth_limit,
                pcre2_heap_limit,
                pcre2_max_jit_stack_size,
            ),
        )
    }

    /// Download a tokenizer from HuggingFace Hub for the given model (e.g.
    /// `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    ///
    /// Uses `tokenizer.json` when the repository has one. Repositories that
    /// ship a bare `tiktoken.model` instead (e.g. Moonshot's Kimi) are resolved
    /// through the tiktoken loader.
    ///
    /// The ``pcre2_*`` limits do not apply to a tiktoken model. Those patterns use
    /// character-class intersection (``&&``), which PCRE2 cannot compile, so
    /// pre-tokenization runs on ``fancy-regex`` and there is no PCRE2 matcher to
    /// bound. They are accepted and ignored on that path rather than rejected.
    #[staticmethod]
    #[pyo3(signature = (model, pcre2_match_limit = None, pcre2_depth_limit = None, pcre2_heap_limit = None, pcre2_max_jit_stack_size = None))]
    fn from_model(
        model: &str,
        pcre2_match_limit: Option<u32>,
        pcre2_depth_limit: Option<u32>,
        pcre2_heap_limit: Option<u32>,
        pcre2_max_jit_stack_size: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let options = tokenizer_options(
            pcre2_match_limit,
            pcre2_depth_limit,
            pcre2_heap_limit,
            pcre2_max_jit_stack_size,
        );

        // `tokenizer.json` is fetched here rather than deferred to
        // `Tokenizer::from_model_with_options` because the `post_processor`
        // property needs the raw JSON *text* to round-trip through `str()`.
        // `Tokenizer::post_processor()` does expose the parsed value, but not the
        // source it was built from. Only a genuinely absent file falls through —
        // a transport or auth failure must propagate.
        let json = py
            .allow_threads(
                || match fastokens::Tokenizer::download_tokenizer_json(model) {
                    Ok(json) => Ok(Some(json)),
                    Err(e) if e.is_not_found() => Ok(None),
                    Err(e) => Err(e.to_string()),
                },
            )
            .map_err(PyValueError::new_err)?;

        let Some(json) = json else {
            // No `tokenizer.json`: let the Rust loader resolve the repository as
            // a tiktoken model. It re-probes `tokenizer.json` (one extra 404 on
            // this path only) in exchange for keeping the common path untouched.
            // A tiktoken pipeline has no post-processor, so nothing is lost.
            let inner = py
                .allow_threads(|| {
                    fastokens::Tokenizer::from_model_with_options(model, options)
                        .map_err(|e| e.to_string())
                })
                .map_err(PyValueError::new_err)?;
            return Ok(Self::with_state(TokenizerState {
                inner,
                trunc: None,
                pad: None,
                post_processor_json: None,
            }));
        };

        Self::build_from_str(&json, py, options)
    }

    /// Create a tokenizer from a tiktoken model file (e.g. `tiktoken.model`,
    /// or OpenAI's `.tiktoken` files).
    ///
    /// A tiktoken model file contains only the byte-level BPE ranks. The
    /// pre-tokenization regex (`pattern`) and `special_tokens` are not in the
    /// file and must be supplied here — or pass `encoding="cl100k_base"` /
    /// `"o200k_base"` to use the corresponding OpenAI defaults. An explicit
    /// `pattern` / `special_tokens` overrides the preset.
    #[staticmethod]
    #[pyo3(signature = (path, pattern = None, special_tokens = None, encoding = None))]
    fn from_tiktoken(
        path: &str,
        pattern: Option<String>,
        special_tokens: Option<std::collections::HashMap<String, u32>>,
        encoding: Option<String>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        use fastokens::TiktokenConfig;

        let preset = match encoding.as_deref() {
            Some(name) => Some(TiktokenConfig::from_preset(name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown tiktoken encoding preset {name:?}; \
                     expected 'cl100k_base' or 'o200k_base'"
                ))
            })?),
            None => None,
        };

        let pattern = pattern
            .or_else(|| preset.as_ref().map(|p| p.pattern.clone()))
            .ok_or_else(|| {
                PyValueError::new_err(
                    "a `pattern` (pre-tokenization regex) is required unless \
                     `encoding` names a known preset",
                )
            })?;

        let special_tokens: Vec<(String, u32)> = match special_tokens {
            Some(map) => map.into_iter().collect(),
            None => preset.map(|p| p.special_tokens).unwrap_or_default(),
        };

        let config = TiktokenConfig::new(pattern, special_tokens);

        let contents = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("cannot read {path}: {e}")))?;

        let inner = py
            .allow_threads(|| {
                fastokens::Tokenizer::from_tiktoken_str(&contents, config)
                    .map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;

        Ok(Self::with_state(TokenizerState {
            inner,
            trunc: None,
            pad: None,
            post_processor_json: None,
        }))
    }

    // ── Post-processor ────────────────────────────────────────────────

    /// The current post-processor, or ``None`` if none is configured.
    ///
    /// The returned object's ``__str__`` yields its JSON representation,
    /// so ``str(tokenizer.post_processor)`` round-trips through the setter.
    #[getter]
    fn post_processor(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.read().post_processor_json {
            None => Ok(py.None()),
            Some(json) => Py::new(py, PyPostProcessor { json: json.clone() }).map(|p| p.into_any()),
        }
    }

    /// Set the post-processor.
    ///
    /// Accepts anything whose ``str()`` yields a valid post-processor JSON —
    /// including our own ``PostProcessor`` objects and ``tokenizers.processors.*``
    /// objects from the HuggingFace tokenizers library.
    #[setter]
    fn set_post_processor(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            let mut state = self.write();
            state.inner.set_post_processor(None);
            state.post_processor_json = None;
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
        self.write().update_post_processor_json(&json_str)
    }

    // ── Truncation ────────────────────────────────────────────────────

    #[pyo3(signature = (max_length, stride = 0, strategy = "longest_first", direction = "right"))]
    fn enable_truncation(&self, max_length: usize, stride: usize, strategy: &str, direction: &str) {
        self.write().trunc = Some(TruncationParams {
            max_length,
            stride,
            strategy: strategy.to_string(),
            direction: direction.to_string(),
        });
    }

    fn no_truncation(&self) {
        self.write().trunc = None;
    }

    #[getter]
    fn truncation(&self, py: Python<'_>) -> PyObject {
        match &self.read().trunc {
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
        &self,
        direction: &str,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
        length: Option<usize>,
        pad_to_multiple_of: Option<usize>,
    ) {
        self.write().pad = Some(PaddingParams {
            direction: direction.to_string(),
            pad_id,
            pad_type_id,
            pad_token: pad_token.to_string(),
            length,
            pad_to_multiple_of,
        });
    }

    fn no_padding(&self) {
        self.write().pad = None;
    }

    #[getter]
    fn padding(&self, py: Python<'_>) -> PyObject {
        match &self.read().pad {
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
    /// With `split_special_tokens`, special added tokens in `input` are encoded
    /// as ordinary text instead of as control-token IDs; the rest of the added
    /// vocabulary still matches. This is what `transformers` asks for when a
    /// caller passes `split_special_tokens=True` to keep untrusted text from
    /// producing control tokens.
    ///
    /// Truncation and padding configured via `enable_truncation` /
    /// `enable_padding` are applied before returning.
    #[pyo3(signature = (input, add_special_tokens = false, split_special_tokens = false))]
    fn encode(
        &self,
        input: &Bound<'_, PyString>,
        add_special_tokens: bool,
        split_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        let text = utf8(input)?;
        let policy = added_token_policy(split_special_tokens);
        let run = || {
            self.read()
                .inner
                .encode_with_policy(&text, add_special_tokens, policy)
                .map_err(|e| e.to_string())
        };
        let ids = if text.len() >= GIL_RELEASE_MIN_BYTES {
            py.allow_threads(run)
        } else {
            run()
        }
        .map_err(PyValueError::new_err)?;
        self.single_encoding(py, ids)
    }

    /// Encode through the base tokenizer pipeline without recognizing added
    /// vocabulary entries.
    ///
    /// Truncation and padding configured via `enable_truncation` /
    /// `enable_padding` are applied before returning.
    fn encode_ordinary(
        &self,
        input: &Bound<'_, PyString>,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        let text = utf8(input)?;
        let run = || {
            self.read()
                .inner
                .encode_ordinary(&text)
                .map_err(|e| e.to_string())
        };
        let ids = if text.len() >= GIL_RELEASE_MIN_BYTES {
            py.allow_threads(run)
        } else {
            run()
        }
        .map_err(PyValueError::new_err)?;
        self.single_encoding(py, ids)
    }

    /// Encode a pre-segmented input, concatenating each segment's token ids.
    ///
    /// `segments` is a list of `(text, allow_special)` pairs. Each segment is
    /// tokenized independently; special/added tokens are recognized only in
    /// segments with `allow_special=True` (trusted chat-template output), so a
    /// literal control token in an `allow_special=False` segment stays ordinary
    /// content. Mirrors legacy tiktoken / Dynamo segmented encoding. No
    /// post-processor special tokens are inserted. Truncation and padding (if
    /// enabled) are applied to the concatenated result before returning.
    fn encode_segments(
        &self,
        segments: Vec<(String, bool)>,
        py: Python<'_>,
    ) -> PyResult<Py<PyEncoding>> {
        let state = self.read();
        let segs: Vec<fastokens::EncodeSegment<'_>> = segments
            .iter()
            .map(|(text, allow_special)| fastokens::EncodeSegment {
                text: text.as_str(),
                allow_special: *allow_special,
            })
            .collect();
        let ids = state
            .inner
            .encode_segments(&segs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        drop(state);
        self.single_encoding(py, ids)
    }

    /// Encode a batch of inputs in parallel.
    ///
    /// `split_special_tokens` has the same meaning as in [`Self::encode`].
    ///
    /// Truncation is applied per-sequence; padding (if enabled) pads the
    /// batch to a uniform length.
    #[pyo3(signature = (inputs, add_special_tokens = false, split_special_tokens = false))]
    fn encode_batch(
        &self,
        inputs: Vec<Bound<'_, PyString>>,
        add_special_tokens: bool,
        split_special_tokens: bool,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyEncoding>>> {
        let policy = added_token_policy(split_special_tokens);
        let texts = inputs.iter().map(utf8).collect::<PyResult<Vec<_>>>()?;
        self.ensure_id_objects(py)?;
        let state = self.read();
        let mut batch: Vec<Vec<u32>> = py
            .allow_threads(|| {
                state
                    .inner
                    .encode_batch_with_policy(&texts, add_special_tokens, policy)
                    .map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;

        for ids in &mut batch {
            state.do_truncate(ids);
        }

        let pad_target: Option<usize> = state.pad.as_ref().map(|p| {
            let max_len = batch.iter().map(|ids| ids.len()).max().unwrap_or(0);
            let base = p.length.unwrap_or(max_len).max(max_len);
            match p.pad_to_multiple_of {
                Some(m) if m > 0 => base.div_ceil(m) * m,
                _ => base,
            }
        });

        let encodings: Vec<EncodingData> = batch
            .into_iter()
            .map(|ids| {
                let target = pad_target.unwrap_or(ids.len());
                build_encoding(ids, state.pad.as_ref(), target)
            })
            .collect();
        drop(state);
        // The `ids` lists every caller goes on to read, built together on the pool.
        let mut lists = prebuild_lists(py, &encodings)?.into_iter();
        encodings
            .into_iter()
            .map(|data| {
                Py::new(
                    py,
                    PyEncoding {
                        data,
                        prebuilt_ids: lists.next().flatten(),
                    },
                )
            })
            .collect()
    }

    /// Encode a batch into a single flat token buffer, for high-throughput bulk
    /// tokenization (e.g. building a training corpus). Returns
    /// `(ids, offsets)` where `ids` is the concatenated token ids of every input
    /// as little-endian `uint32` bytes, and `offsets` is `len(inputs) + 1`
    /// little-endian `uint64` values: input `i`'s tokens are
    /// `ids[offsets[i]:offsets[i+1]]`. Decode with
    /// `np.frombuffer(ids, np.uint32)` / `np.frombuffer(offsets, np.uint64)`.
    ///
    /// Unlike [`encode_batch`], this materializes no per-token Python objects —
    /// the whole result is two buffers — which is what makes bulk encoding fast
    /// from Python. Truncation is applied per input; padding does not apply
    /// (the result is ragged, addressed by `offsets`).
    ///
    /// `split_special_tokens` has the same meaning as in [`Self::encode`], so
    /// the bulk path can suppress control-token IDs for untrusted input the
    /// same way the per-encoding paths do.
    #[pyo3(signature = (inputs, add_special_tokens = false, split_special_tokens = false))]
    fn encode_batch_flat<'py>(
        &self,
        inputs: Vec<Bound<'py, PyString>>,
        add_special_tokens: bool,
        split_special_tokens: bool,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let policy = added_token_policy(split_special_tokens);
        let texts = inputs.iter().map(utf8).collect::<PyResult<Vec<_>>>()?;
        let state = self.read();
        let mut batch: Vec<Vec<u32>> = py
            .allow_threads(|| {
                state
                    .inner
                    .encode_batch_with_policy(&texts, add_special_tokens, policy)
                    .map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        for ids in &mut batch {
            state.do_truncate(ids);
        }

        // Concatenate into one flat id buffer and build cumulative offsets.
        let total: usize = batch.iter().map(Vec::len).sum();
        let mut flat: Vec<u32> = Vec::with_capacity(total);
        let mut offsets: Vec<u64> = Vec::with_capacity(batch.len() + 1);
        offsets.push(0);
        for ids in &batch {
            flat.extend_from_slice(ids);
            offsets.push(flat.len() as u64);
        }

        // Reinterpret the id/offset buffers as bytes (same allocation) and copy
        // them into Python `bytes` — no per-element Python objects.
        // SAFETY: `u32`/`u64` slices reinterpret as byte slices of the same
        // length in bytes; both are `Copy` with no padding.
        let ids_bytes =
            unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
        let off_bytes =
            unsafe { std::slice::from_raw_parts(offsets.as_ptr() as *const u8, offsets.len() * 8) };
        Ok((PyBytes::new(py, ids_bytes), PyBytes::new(py, off_bytes)))
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
        let new_ids = self.read().inner.post_process(ids, true);
        self.ensure_id_objects(py)?;
        Py::new(py, PyEncoding::from(EncodingData::make(new_ids, None)))
    }

    /// Return the number of special tokens added for a single or pair sequence.
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        if is_pair {
            return 0; // pair not supported
        }
        // Probe: encode empty IDs with and without special tokens.
        let with_special = self.read().inner.post_process(vec![], true);
        with_special.len()
    }

    // ── Decoding ──────────────────────────────────────────────────────

    /// Decode a list of token strings back into text using the decoder pipeline.
    ///
    /// This is what `convert_tokens_to_string` needs: token strings (e.g.
    /// "Ġhello") → decoded text (" hello").  The decoder (e.g. ByteLevel)
    /// is applied exactly as during normal `decode`.
    fn decode_tokens(&self, tokens: Vec<String>) -> PyResult<String> {
        self.read()
            .inner
            .decode_tokens(tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs back into text.
    #[pyo3(signature = (ids, skip_special_tokens = false))]
    fn decode(&self, ids: &Bound<'_, PyAny>, skip_special_tokens: bool) -> PyResult<String> {
        let ids = Self::extract_decode_ids(ids)?;
        self.read()
            .inner
            .decode(&ids, skip_special_tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode a batch of token ID sequences.
    #[pyo3(signature = (sentences, skip_special_tokens = false))]
    fn decode_batch(
        &self,
        sentences: &Bound<'_, PyAny>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        let sentences = Self::extract_decode_batch_ids(sentences)?;
        let state = self.read();
        let refs: Vec<&[u32]> = sentences.iter().map(Vec::as_slice).collect();
        state
            .inner
            .decode_batch(&refs, skip_special_tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // ── Vocabulary ────────────────────────────────────────────────────

    /// Look up the token ID for a string.
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.read().inner.token_to_id(token)
    }

    /// Look up the string for a token ID.
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.read().inner.id_to_token(id).map(String::from)
    }

    /// Return the vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.read().inner.vocab_size()
    }

    /// The added-vocabulary entries, as `tokenizer.json` would serialize them.
    ///
    /// Reflects tokens added since construction, so callers that re-serialize
    /// the tokenizer do not lose them.
    fn added_tokens<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.read()
            .inner
            .added_token_configs()
            .iter()
            .map(|config| {
                let entry = PyDict::new(py);
                entry.set_item("id", config.id)?;
                entry.set_item("content", &config.content)?;
                entry.set_item("single_word", config.single_word)?;
                entry.set_item("lstrip", config.lstrip)?;
                entry.set_item("rstrip", config.rstrip)?;
                entry.set_item("normalized", config.normalized)?;
                entry.set_item("special", config.special)?;
                Ok(entry)
            })
            .collect()
    }

    /// Extend the vocabulary, returning the `(content, id)` of every entry
    /// created or changed.
    ///
    /// `tokens` are objects carrying the attributes of a
    /// `tokenizers.AddedToken`. IDs are assigned exactly as HuggingFace
    /// `tokenizers` assigns them: a content already in the vocabulary keeps its
    /// ID (only its flags can change), anything else is appended above the
    /// vocabulary. Adding a content that is already present with the same flags
    /// changes nothing and is not reported.
    fn add_tokens(&self, tokens: Vec<PyNewToken>) -> PyResult<Vec<(String, u32)>> {
        let tokens: Vec<fastokens::NewToken> = tokens.into_iter().map(Into::into).collect();
        let added = self
            .write()
            .inner
            .add_tokens(&tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(added
            .into_iter()
            .map(|config| (config.content, config.id))
            .collect())
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
        let mut enc = EncodingData::make(vec![10u32, 20, 30], None);
        // 3 real tokens → pad to length 5 with pad_type_id = 1
        enc.pad_to(5, "right", 0u32, 1u32);

        assert_eq!(enc.ids, vec![10u32, 20, 30, 0, 0]);
        assert_eq!(*enc.attention_mask_vec(), [1u32, 1, 1, 0, 0]);
        assert_eq!(
            *enc.type_ids_vec(),
            [0u32, 0, 0, 1, 1],
            "padded positions should carry pad_type_id=1 in type_ids"
        );
    }

    /// The tokenizer encode paths build returned encodings through the same
    /// padding owner as `PyEncoding::pad`, preserving `pad_type_id` metadata.
    #[test]
    fn encode_batch_pad_type_id_applied_to_type_ids() {
        let pad = PaddingParams {
            direction: "right".to_string(),
            pad_id: 0,
            pad_type_id: 1,
            pad_token: "[PAD]".to_string(),
            length: None,
            pad_to_multiple_of: None,
        };
        let enc = build_encoding(vec![10u32, 20, 30], Some(&pad), 5);

        assert_eq!(enc.ids, vec![10u32, 20, 30, 0, 0]);
        assert_eq!(*enc.attention_mask_vec(), [1u32, 1, 1, 0, 0]);
        assert_eq!(*enc.type_ids_vec(), [0u32, 0, 0, 1, 1]);
    }

    #[test]
    fn build_encoding_left_padding_applies_pad_type_id() {
        let pad = PaddingParams {
            direction: "left".to_string(),
            pad_id: 0,
            pad_type_id: 7,
            pad_token: "[PAD]".to_string(),
            length: None,
            pad_to_multiple_of: None,
        };
        let enc = build_encoding(vec![10u32, 20, 30], Some(&pad), 5);

        assert_eq!(enc.ids, vec![0u32, 0, 10, 20, 30]);
        assert_eq!(*enc.attention_mask_vec(), [0u32, 0, 1, 1, 1]);
        assert_eq!(*enc.type_ids_vec(), [7u32, 7, 0, 0, 0]);
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
        let state = tok.read();
        self.inner
            .step(&state.inner, new_ids)
            .map_err(PyValueError::new_err)
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
