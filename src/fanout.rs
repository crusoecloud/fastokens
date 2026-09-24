//! A small fork-join executor for the encode hot paths.
//!
//! One encode splits its input into chunks and needs them all tokenized as soon as
//! possible. A general work-stealing pool is a poor fit for that shape: rayon
//! wakes sleeping workers roughly one at a time as work is split, and on a
//! virtualized many-core host every idle-core wake costs tens to hundreds of
//! microseconds, so the last chunk of a job can start milliseconds after the
//! first — longer than tokenizing the whole chunk. This executor instead:
//!
//! - lets the **caller participate**: it claims chunk indices from a shared
//!   counter like any worker, so a job is never slower than running it inline —
//!   if workers are slow to arrive, the caller simply takes more chunks;
//! - **wakes all the workers a job needs at once** (one `unpark` each) instead of
//!   in a chain;
//! - keeps workers **spinning briefly** after a job ([`spin_duration`]) so a burst
//!   of requests skips the wake-up entirely, then parks them so an idle process
//!   burns no CPU.
//!
//! Jobs are claimed first-come, first-served, so concurrent callers (Python
//! threads with the GIL released) and nested jobs (a batch item that is itself
//! long) share the workers without deadlock: a caller always makes progress on its
//! own job and only ever waits for chunks a worker is already running.
//!
//! Public so the bindings can run their own bulk work (building Python lists) on
//! the same threads; it is not otherwise part of the tokenizer API.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::Thread;
use std::time::{Duration, Instant};

/// One published job: run `f(i)` for every `i in 0..n`.
struct Job {
    /// The task body. Its real lifetime is the [`for_each`] call that owns the
    /// job, which does not return until no worker can reach it (`active == 0`
    /// after it has left the board).
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    /// Next task index to claim.
    next: AtomicUsize,
    /// Tasks finished (successfully or by panic).
    done: AtomicUsize,
    /// Workers currently attached to this job. Only incremented under the board
    /// lock while the job is on the board.
    active: AtomicUsize,
    /// The first panic payload raised by a task, re-raised on the caller.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

/// Raw job pointer on the board. The owning [`for_each`] keeps the job alive
/// while it is listed and until every attached worker has detached.
#[derive(Clone, Copy, PartialEq, Eq)]
struct JobPtr(*const Job);
// SAFETY: a `Job` is only shared through the protocol described on `Job::f`; its
// fields are atomics / a mutex, and `f` is `Sync`.
unsafe impl Send for JobPtr {}

struct Pool {
    /// Jobs with (possibly) unclaimed tasks, oldest first.
    board: Mutex<Vec<JobPtr>>,
    /// Parked workers, available to be woken.
    idle: Mutex<Vec<Thread>>,
    /// Bumped whenever a job is published; spinning workers watch it.
    epoch: AtomicU64,
    /// Workers currently spinning (they will pick up a new job unprompted).
    spinning: AtomicUsize,
    /// Number of worker threads (the caller is one more participant).
    workers: usize,
}

impl Pool {
    /// Whether any listed job still has unclaimed tasks.
    fn has_open_work(&self) -> bool {
        self.board.lock().unwrap().iter().any(|&j| {
            // SAFETY: jobs on the board are alive (see `Job::f`).
            let job = unsafe { &*j.0 };
            job.next.load(Ordering::Relaxed) < job.n
        })
    }

    /// Attach to the oldest job that still has unclaimed tasks.
    fn attach(&self) -> Option<JobPtr> {
        let board = self.board.lock().unwrap();
        for &j in board.iter() {
            // SAFETY: jobs on the board are alive (see `Job::f`).
            let job = unsafe { &*j.0 };
            if job.next.load(Ordering::Relaxed) < job.n {
                job.active.fetch_add(1, Ordering::AcqRel);
                return Some(j);
            }
        }
        None
    }
}

/// Default participant count: every logical core up to [`MAX_DEFAULT_THREADS`];
/// on Apple Silicon the performance-core count (efficiency cores only add
/// straggler latency to a barrier-synchronized job). Overridable with
/// `FASTOKENS_BPE_THREADS`.
fn default_threads() -> usize {
    #[cfg(target_os = "macos")]
    if let Some(p) = perf_core_count() {
        return p.min(MAX_DEFAULT_THREADS);
    }
    std::thread::available_parallelism().map_or(1, |n| n.get().min(MAX_DEFAULT_THREADS))
}

/// Past this many participants an encode gets slower, not faster: every extra
/// thread brings a cold pretoken cache (a cache miss costs a full BPE merge), and
/// on big hosts the extra threads land on SMT siblings and remote NUMA nodes. On a
/// 2x44-core host 16–32 threads were fastest for both long single documents and
/// batches, and 176 (all logical cores) was ~2x slower than 32.
const MAX_DEFAULT_THREADS: usize = 32;

/// On Apple Silicon, the number of performance (P) cores (`hw.perflevel0.logicalcpu`).
#[cfg(target_os = "macos")]
fn perf_core_count() -> Option<usize> {
    let name = c"hw.perflevel0.logicalcpu";
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut libc::c_int as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value as usize)
}

/// How long a worker keeps polling for new work after finishing a job before it
/// parks: long enough to bridge the gap between back-to-back encodes, short
/// enough that an idle process stops burning CPU almost at once. Overridable in
/// microseconds with `FASTOKENS_SPIN_US` (`0` parks immediately).
fn spin_duration() -> Duration {
    static SPIN: OnceLock<Duration> = OnceLock::new();
    *SPIN.get_or_init(|| {
        let us = std::env::var("FASTOKENS_SPIN_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_SPIN_US);
        Duration::from_micros(us)
    })
}

/// Default for [`spin_duration`]. Longer spins measured no faster and, with many
/// workers, slower: spinning threads compete with working SMT siblings.
const DEFAULT_SPIN_US: u64 = 50;

/// Number of participants (workers + the caller) a job can use.
pub fn threads() -> usize {
    pool().map_or(1, |p| p.workers + 1)
}

fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<&'static Pool>> = OnceLock::new();
    *POOL.get_or_init(|| {
        let avail = std::thread::available_parallelism().map_or(1, |n| n.get());
        let n = std::env::var("FASTOKENS_BPE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(default_threads)
            .min(avail);
        if n < 2 {
            return None;
        }
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            board: Mutex::new(Vec::new()),
            idle: Mutex::new(Vec::with_capacity(n)),
            epoch: AtomicU64::new(0),
            spinning: AtomicUsize::new(0),
            workers: n - 1,
        }));
        for i in 0..pool.workers {
            std::thread::Builder::new()
                .name(format!("fastokens-{i}"))
                .spawn(move || worker(pool))
                .expect("failed to spawn fastokens worker thread");
        }
        Some(pool)
    })
}

thread_local! {
    /// Whether this thread is running a task of a multi-threaded job.
    static IN_TASK: Cell<bool> = const { Cell::new(false) };
}

/// Whether the calling thread is running a task of a job spread over the pool —
/// where work shared between its threads (a cross-thread cache) pays off.
pub(crate) fn in_parallel_task() -> bool {
    IN_TASK.with(Cell::get)
}

/// Claim and run tasks of `job` until none are left.
fn run_tasks(job: &Job) {
    // SAFETY: the job (and so `f`) outlives every attached participant.
    let f = unsafe { &*job.f };
    let outer = IN_TASK.with(|t| t.replace(true));
    loop {
        let i = job.next.fetch_add(1, Ordering::AcqRel);
        if i >= job.n {
            break;
        }
        if let Err(p) = catch_unwind(AssertUnwindSafe(|| f(i))) {
            job.panic.lock().unwrap().get_or_insert(p);
        }
        job.done.fetch_add(1, Ordering::Release);
    }
    IN_TASK.with(|t| t.set(outer));
}

fn worker(pool: &'static Pool) {
    let me = std::thread::current();
    let spin = spin_duration();
    loop {
        if let Some(j) = pool.attach() {
            // SAFETY: attached, so the job stays alive until we detach below.
            let job = unsafe { &*j.0 };
            run_tasks(job);
            job.active.fetch_sub(1, Ordering::AcqRel);
            continue;
        }
        // Spin for a new job, then park.
        pool.spinning.fetch_add(1, Ordering::AcqRel);
        let seen = pool.epoch.load(Ordering::Acquire);
        // A job published before `seen` was read may have been counted on our
        // spinning to cover it: look once more before waiting for the next epoch.
        if pool.has_open_work() {
            pool.spinning.fetch_sub(1, Ordering::AcqRel);
            continue;
        }
        let start = Instant::now();
        let mut woke = false;
        while start.elapsed() < spin {
            for _ in 0..64 {
                std::hint::spin_loop();
            }
            if pool.epoch.load(Ordering::Acquire) != seen {
                woke = true;
                break;
            }
        }
        pool.spinning.fetch_sub(1, Ordering::AcqRel);
        if woke {
            continue;
        }
        // Register as idle *before* re-checking the board: a job published after
        // the re-check finds us on the idle list and unparks us (see `for_each`).
        // A listed job whose tasks are all claimed is no reason to stay awake.
        pool.idle.lock().unwrap().push(me.clone());
        if !pool.has_open_work() {
            std::thread::park();
        }
        // Leave the idle list if a waker did not already take us off it.
        let mut idle = pool.idle.lock().unwrap();
        if let Some(k) = idle.iter().position(|t| t.id() == me.id()) {
            idle.swap_remove(k);
        }
    }
}

/// Run `f(i)` for every `i in 0..n`, spread over the worker pool with the caller
/// participating; returns when all have run. A panic in any task is re-raised
/// here after every task has finished.
pub fn for_each<F: Fn(usize) + Sync>(n: usize, f: F) {
    let pool = match pool() {
        Some(p) if n >= 2 => p,
        _ => {
            (0..n).for_each(f);
            return;
        }
    };
    let dynf: &(dyn Fn(usize) + Sync) = &f;
    let job = Job {
        // SAFETY: erases the borrow's lifetime; `job` never outlives this call
        // (see the wait below), so `f` is alive whenever it is reached.
        f: unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync)>(
                dynf,
            )
        },
        n,
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        panic: Mutex::new(None),
    };
    let ptr = JobPtr(&job);
    pool.board.lock().unwrap().push(ptr);
    pool.epoch.fetch_add(1, Ordering::AcqRel);

    // Wake as many parked workers as there are tasks the caller and the already
    // spinning workers will not cover — all at once, not in a chain.
    let want = (n - 1).saturating_sub(pool.spinning.load(Ordering::Acquire));
    if want > 0 {
        let wake: Vec<Thread> = {
            let mut idle = pool.idle.lock().unwrap();
            let k = want.min(idle.len());
            let at = idle.len() - k;
            idle.drain(at..).collect()
        };
        for t in wake {
            t.unpark();
        }
    }

    run_tasks(&job);

    // No task is left to claim: take the job off the board so no new worker
    // attaches, then wait for the workers still finishing claimed tasks.
    {
        let mut board = pool.board.lock().unwrap();
        if let Some(k) = board.iter().position(|&j| j == ptr) {
            board.remove(k);
        }
    }
    let mut spins = 0u32;
    while job.active.load(Ordering::Acquire) != 0 || job.done.load(Ordering::Acquire) != n {
        if spins < 1 << 14 {
            std::hint::spin_loop();
            spins += 1;
        } else {
            std::thread::yield_now();
        }
    }
    if let Some(p) = job.panic.into_inner().unwrap() {
        resume_unwind(p);
    }
}

/// Write-once result slots, one per task, filled concurrently by [`map`].
struct Slots<T>(Vec<UnsafeCell<Option<T>>>);
// SAFETY: each slot is written by exactly one task (its index) and read only after
// every task has finished (`for_each` returned), so there is no concurrent access
// to any slot.
unsafe impl<T: Send> Sync for Slots<T> {}

/// `(0..n).map(f).collect()`, computed in parallel via [`for_each`].
pub fn map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    if n < 2 || pool().is_none() {
        return (0..n).map(f).collect();
    }
    let slots = Slots((0..n).map(|_| UnsafeCell::new(None)).collect());
    {
        // Borrow the whole `Slots` (which is `Sync`), not just its field.
        let slots = &slots;
        for_each(n, |i| {
            let v = f(i);
            // SAFETY: slot `i` is written only by task `i` (see `Slots`).
            unsafe { *slots.0[i].get() = Some(v) };
        });
    }
    slots
        .0
        .into_iter()
        .map(|c| c.into_inner().expect("every task ran"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn map_preserves_order_and_runs_every_task() {
        for n in [0usize, 1, 2, 3, 17, 1000] {
            let v = map(n, |i| i * 3);
            assert_eq!(v, (0..n).map(|i| i * 3).collect::<Vec<_>>());
        }
    }

    #[test]
    fn concurrent_and_nested_jobs() {
        let total = AtomicU32::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..50 {
                        let inner: Vec<u32> = map(16, |i| {
                            // Nested job from inside a task.
                            map(8, |j| (i * 8 + j) as u32).iter().sum()
                        });
                        total.fetch_add(inner.iter().sum::<u32>(), Ordering::Relaxed);
                    }
                });
            }
        });
        let per_call: u32 = (0..128u32).sum();
        assert_eq!(total.load(Ordering::Relaxed), per_call * 8 * 50);
    }

    #[test]
    fn task_panic_is_reraised_after_all_tasks() {
        let ran = AtomicU32::new(0);
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
            for_each(64, |i| {
                ran.fetch_add(1, Ordering::Relaxed);
                if i == 5 {
                    panic!("boom");
                }
            })
        }));
        assert!(r.is_err());
        assert_eq!(ran.load(Ordering::Relaxed), 64);
        // The pool still works afterwards.
        assert_eq!(map(10, |i| i).len(), 10);
    }
}
