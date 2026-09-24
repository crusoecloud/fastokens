//! Fixed-size tables for the encode hot paths, backed by memory that may use
//! transparent huge pages.
//!
//! The pretoken cache and the merge tables are megabytes to tens of megabytes
//! probed at random: on 4 KiB pages nearly every probe also misses the TLB, and
//! a page walk is paid on top of the cache miss (a software prefetch that misses
//! the TLB may even be dropped). On Linux a table of at least one huge page is
//! its own anonymous mapping, 2 MiB-aligned and `madvise(MADV_HUGEPAGE)`d before
//! its first touch, so under the common `madvise` THP policy it faults in as huge
//! pages — the whole table then fits in a few TLB entries. (Through `malloc` it
//! would often be carved from heap memory an earlier phase already faulted in as
//! 4 KiB pages, which the advice cannot change.) Elsewhere, or when the kernel
//! declines, it is ordinary memory.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

const HUGE_PAGE: usize = 2 << 20;

/// A boxed slice of `T` whose allocation is huge-page-friendly (see the module
/// docs); it dereferences to `[T]`.
pub(crate) struct HugeTable<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    /// Whether the memory is a mapping of its own (else from the allocator).
    mapped: bool,
}

// SAFETY: `HugeTable` owns its elements exclusively, like `Box<[T]>`.
unsafe impl<T: Copy + Send> Send for HugeTable<T> {}
// SAFETY: as above; shared access only hands out `&[T]`.
unsafe impl<T: Copy + Sync> Sync for HugeTable<T> {}

impl<T: Copy> HugeTable<T> {
    /// `len` copies of `fill`.
    pub(crate) fn new(len: usize, fill: T) -> Self {
        let t = Self::alloc(len);
        for i in 0..len {
            // SAFETY: `i < len`, within the fresh allocation.
            unsafe { t.ptr.as_ptr().add(i).write(fill) };
        }
        t
    }

    /// A copy of `src`.
    pub(crate) fn from_slice(src: &[T]) -> Self {
        let t = Self::alloc(src.len());
        // SAFETY: the fresh allocation holds `src.len()` elements and cannot
        // overlap `src`.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), t.ptr.as_ptr(), src.len()) };
        t
    }

    /// Room for `len` elements, advised for huge pages but not yet touched.
    fn alloc(len: usize) -> Self {
        let layout = Self::layout(len);
        if layout.size() == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len,
                mapped: false,
            };
        }
        if layout.size() >= HUGE_PAGE
            && let Some(p) = map_huge(layout.size())
        {
            return Self {
                ptr: p.cast(),
                len,
                mapped: true,
            };
        }
        // SAFETY: the layout has a nonzero size.
        let raw = unsafe { alloc(layout) };
        let Some(ptr) = NonNull::new(raw as *mut T) else {
            handle_alloc_error(layout)
        };
        Self {
            ptr,
            len,
            mapped: false,
        }
    }

    fn layout(len: usize) -> Layout {
        let size = len
            .checked_mul(size_of::<T>())
            .expect("table size overflow");
        // Huge-page alignment once the table spans a huge page (the allocator
        // path where no mapping is made); else whole cache lines, so line-sized
        // entries never straddle two.
        let align = if size >= HUGE_PAGE {
            HUGE_PAGE
        } else {
            align_of::<T>().max(64)
        };
        Layout::from_size_align(size, align).expect("table layout")
    }
}

impl<T: Copy> Drop for HugeTable<T> {
    fn drop(&mut self) {
        let layout = Self::layout(self.len);
        if layout.size() == 0 {
            return;
        }
        if self.mapped {
            unmap_huge(self.ptr.as_ptr() as *mut u8, layout.size());
            return;
        }
        // SAFETY: allocated in `alloc` with this same layout.
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) };
    }
}

impl<T: Copy> Deref for HugeTable<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: `len` initialized elements (`new` / `from_slice` wrote them all).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for HugeTable<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> Clone for HugeTable<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl<T: Copy> From<Vec<T>> for HugeTable<T> {
    fn from(v: Vec<T>) -> Self {
        Self::from_slice(&v)
    }
}

/// A fresh anonymous mapping of `size` bytes (a multiple of 4 KiB is not
/// required), 2 MiB-aligned and advised for huge pages; `None` if the mapping
/// fails (the caller then uses the allocator).
#[cfg(target_os = "linux")]
fn map_huge(size: usize) -> Option<NonNull<u8>> {
    let span = size.next_multiple_of(HUGE_PAGE) + HUGE_PAGE;
    // SAFETY: a new private anonymous mapping; no existing memory is affected.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            span,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        return None;
    }
    let base = base as usize;
    let start = base.next_multiple_of(HUGE_PAGE);
    let end = start + size.next_multiple_of(HUGE_PAGE);
    // SAFETY: the trimmed head and tail lie inside the mapping made above, and
    // the kept part is advised before anything touches it; failure to trim or
    // advise only costs memory or huge pages.
    unsafe {
        if start > base {
            libc::munmap(base as *mut libc::c_void, start - base);
        }
        if base + span > end {
            libc::munmap(end as *mut libc::c_void, base + span - end);
        }
        libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
    }
    NonNull::new(start as *mut u8)
}

#[cfg(not(target_os = "linux"))]
fn map_huge(_size: usize) -> Option<NonNull<u8>> {
    None
}

/// Unmap a table `map_huge` made (`size` as passed to it).
#[cfg(target_os = "linux")]
fn unmap_huge(p: *mut u8, size: usize) {
    // SAFETY: `p` is the 2 MiB-aligned start of a mapping of this rounded size.
    unsafe { libc::munmap(p as *mut libc::c_void, size.next_multiple_of(HUGE_PAGE)) };
}

#[cfg(not(target_os = "linux"))]
fn unmap_huge(_p: *mut u8, _size: usize) {
    unreachable!("no mappings are made here")
}

#[cfg(test)]
mod tests {
    use super::HugeTable;

    #[test]
    fn tables_hold_their_elements() {
        for len in [0usize, 1, 7, 1 << 16, (2 << 20) / 8 + 3] {
            let mut t = HugeTable::new(len, 7u64);
            assert_eq!(t.len(), len);
            assert!(t.iter().all(|&x| x == 7));
            if len > 0 {
                t[len - 1] = 9;
                let c = t.clone();
                assert_eq!(c[len - 1], 9);
                assert_eq!(c.as_ptr() as usize % 64, 0);
            }
            let v: Vec<u32> = (0..len as u32).collect();
            assert_eq!(&*HugeTable::from(v.clone()), &v[..]);
        }
    }
}
