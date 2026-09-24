//! Bit-parallel pre-tokenizer scanner — the dependency-free port of the
//! `bitcannon` approach (see `docs/hf-tokenizers-v1-comparison-and-port-plan.md`,
//! P1b). It is staged; this module is **milestone 1**: the SIMD/SWAR *classify*
//! kernel (the shared foundation for every grammar's boundary program) and the
//! first cl100k boundary logic built on it.
//!
//! # Why bit-parallel
//!
//! The scalar [`super::scan`] scanner works one pre-token at a time — on prose
//! that is ~4 bytes per pre-token, i.e. millions of per-token dispatches. This
//! engine instead classifies **64 bytes per block** into per-class *bitmaps*
//! (one bit per byte) and derives every token boundary from those bitmaps with
//! whole-register bitwise ops, regardless of how short the tokens are. Only the
//! final "emit one span per set start-bit" step is per-token, and it is a bare
//! `trailing_zeros` + callback with no re-classification.
//!
//! # Portability
//!
//! The classify kernel is plain `u64` SWAR (8 bytes per step, then a bit-gather),
//! so it is correct and reasonably fast on every target with no `unsafe` arch
//! intrinsics. A NEON/AVX2 kernel is a later, drop-in optimisation behind the
//! same [`Classes`] interface.

/// Per-class bitmaps for a block of up to 64 bytes: bit `j` is set when byte `j`
/// of the block belongs to that class. The classes are exactly the ones the
/// tiktoken-family grammars test over ASCII; `nonascii` marks bytes `>= 0x80`,
/// which the caller resolves through the Unicode tables (a run can cross into a
/// non-ASCII `\p{L}`/`\p{M}`), so no ASCII class bit is set for them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Classes {
    /// `[A-Za-z]`
    pub alpha: u64,
    /// `[0-9]`
    pub digit: u64,
    /// `\s` — whitespace including newline: tab, LF, VT, FF, CR, space.
    pub ws: u64,
    /// `[\r\n]` — the newline subset of `ws`.
    pub nl: u64,
    /// ASCII and none of the above: exactly `[^\s\p{L}\p{N}]` restricted to
    /// ASCII (this includes the apostrophe `'`).
    pub other: u64,
    /// byte `>= 0x80`.
    pub nonascii: u64,
    /// `' '` (space, 0x20) — the one whitespace byte the grammars treat specially
    /// (the `? ` punct prefix). Subset of `ws`.
    pub sp: u64,
    /// `'` (apostrophe, 0x27) — where contractions start. Subset of `other`.
    pub ap: u64,
}

#[allow(dead_code)]
const HI: u64 = 0x8080_8080_8080_8080;
#[allow(dead_code)]
const LO: u64 = 0x0101_0101_0101_0101;

/// Gather the high bit of each of the 8 bytes of `x` into the low 8 bits of the
/// result, byte `j` → bit `j` (little-endian). The classic SWAR movemask.
#[inline(always)]
#[allow(dead_code)]
fn movemask8(x: u64) -> u64 {
    (((x >> 7) & LO).wrapping_mul(0x0102_0408_1020_4080)) >> 56
}

/// High bit set per lane whose byte is in `[lo, hi]` (bytes must be `< 0x80`).
/// Same range-test form as the scalar scanner's `ascii_lower_run_end`.
#[inline(always)]
#[allow(dead_code)]
fn in_range(word: u64, lo: u8, hi: u8) -> u64 {
    let lob = LO.wrapping_mul(lo as u64);
    let hib = LO.wrapping_mul(0x80 + hi as u64);
    (word | HI).wrapping_sub(lob) & hib.wrapping_sub(word) & HI
}

/// Classify up to 64 bytes starting at `off` into [`Classes`]. Bytes past the end
/// of `bytes` (when the final block is short) are left unset in every map.
///
/// Dispatches to a NEON kernel on aarch64 (16 bytes/op) and the portable SWAR
/// kernel elsewhere; both produce identical bitmaps (`classify_matches_scalar`,
/// and `neon_matches_swar` on aarch64).
#[inline]
pub(crate) fn classify_block(bytes: &[u8], off: usize) -> Classes {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { classify_block_neon(bytes, off) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if avx2::available() {
            // SAFETY: AVX2 support was just detected.
            return unsafe { avx2::classify_block(bytes, off) };
        }
        classify_block_swar(bytes, off)
    }
}

/// Portable `u64` SWAR classify (8 bytes/step + a movemask gather).
#[inline]
#[allow(dead_code)]
pub(crate) fn classify_block_swar(bytes: &[u8], off: usize) -> Classes {
    let mut c = Classes::default();
    let end = (off + 64).min(bytes.len());
    let mut j = 0usize; // bit position within the block
    let mut p = off;
    while p < end {
        let take = (end - p).min(8);
        let word = read_word(bytes, p, take);
        let na = word & HI; // high bit set per non-ASCII lane
        let ascii = !na & HI; // high bit set per ASCII lane
        // Clear each lane's high bit before the range tests so a non-ASCII lane's
        // subtraction cannot borrow into an adjacent lane (all lanes now < 0x80).
        // Non-ASCII lanes may match a class here spuriously; `& ascii` drops them.
        let a = word & !HI;
        let lower = a | 0x2020_2020_2020_2020;
        let alpha = in_range(lower, b'a', b'z') & ascii;
        let digit = in_range(a, b'0', b'9') & ascii;
        let ws_ctrl = in_range(a, 0x09, 0x0D) & ascii; // \t \n \v \f \r
        let ws_sp = in_range(a, b' ', b' ') & ascii;
        let ws = ws_ctrl | ws_sp;
        let nl = (in_range(a, b'\n', b'\n') | in_range(a, b'\r', b'\r')) & ascii;
        let other = ascii & !(alpha | digit | ws);
        let sp = ws_sp; // space only (0x20), already `& ascii` via ws_sp components
        let ap = in_range(a, b'\'', b'\'') & ascii;
        // Only the low `take` lanes are valid; mask the gathered bits.
        let valid: u64 = if take == 8 { !0 } else { (1u64 << take) - 1 };
        c.alpha |= (movemask8(alpha) & valid) << j;
        c.digit |= (movemask8(digit) & valid) << j;
        c.ws |= (movemask8(ws) & valid) << j;
        c.nl |= (movemask8(nl) & valid) << j;
        c.other |= (movemask8(other) & valid) << j;
        c.nonascii |= (movemask8(na) & valid) << j;
        c.sp |= (movemask8(sp) & valid) << j;
        c.ap |= (movemask8(ap) & valid) << j;
        j += take;
        p += take;
    }
    c
}

/// Read up to 8 bytes at `p` into a `u64` (little-endian), zero-filling lanes past
/// `take`. Zero-filled lanes classify as `other` (NUL), but they are masked out by
/// `valid` in [`classify_block`], so they never contribute a bit.
#[inline(always)]
#[allow(dead_code)]
fn read_word(bytes: &[u8], p: usize, take: usize) -> u64 {
    if take == 8 {
        // SAFETY: caller guarantees p + 8 <= bytes.len().
        unsafe { (bytes.as_ptr().add(p) as *const u64).read_unaligned() }
    } else {
        let mut buf = [0u8; 8];
        buf[..take].copy_from_slice(&bytes[p..p + take]);
        u64::from_le_bytes(buf)
    }
}

/// NEON classify (aarch64): 16 bytes/op. Byte-identical to [`classify_block_swar`].
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)] // NEON intrinsics; the whole fn is an unsafe contract
unsafe fn classify_block_neon(bytes: &[u8], off: usize) -> Classes {
    use std::arch::aarch64::*;

    /// 16 lanes of 0xFF/0x00 → a 16-bit mask, lane `j` → bit `j`.
    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn movemask(v: uint8x16_t) -> u64 {
        const POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vandq_u8(v, vld1q_u8(POWERS.as_ptr()));
        let lo = vaddv_u8(vget_low_u8(bits)) as u64;
        let hi = vaddv_u8(vget_high_u8(bits)) as u64;
        lo | (hi << 8)
    }

    let mut c = Classes::default();
    let end = (off + 64).min(bytes.len());
    let mut p = off;
    let mut j = 0usize; // bit position within the block
    let hi80 = vdupq_n_u8(0x80);
    while p < end {
        let take = (end - p).min(16);
        let v = if take == 16 && p + 16 <= bytes.len() {
            vld1q_u8(bytes.as_ptr().add(p))
        } else {
            let mut buf = [0u8; 16];
            buf[..take].copy_from_slice(&bytes[p..p + take]);
            vld1q_u8(buf.as_ptr())
        };
        let ascii = vcltq_u8(v, hi80);
        let lower = vorrq_u8(v, vdupq_n_u8(0x20));
        let alpha = vandq_u8(
            vandq_u8(
                vcgeq_u8(lower, vdupq_n_u8(b'a')),
                vcleq_u8(lower, vdupq_n_u8(b'z')),
            ),
            ascii,
        );
        let digit = vandq_u8(
            vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'0')), vcleq_u8(v, vdupq_n_u8(b'9'))),
            ascii,
        );
        let ws_sp = vceqq_u8(v, vdupq_n_u8(0x20));
        let ws = vandq_u8(
            vorrq_u8(
                vandq_u8(vcgeq_u8(v, vdupq_n_u8(0x09)), vcleq_u8(v, vdupq_n_u8(0x0D))),
                ws_sp,
            ),
            ascii,
        );
        let nl = vandq_u8(
            vorrq_u8(vceqq_u8(v, vdupq_n_u8(0x0A)), vceqq_u8(v, vdupq_n_u8(0x0D))),
            ascii,
        );
        let sp = vandq_u8(ws_sp, ascii);
        let ap = vandq_u8(vceqq_u8(v, vdupq_n_u8(b'\'')), ascii);
        let na = vmvnq_u8(ascii);
        let other = vandq_u8(ascii, vmvnq_u8(vorrq_u8(alpha, vorrq_u8(digit, ws))));

        let valid: u64 = if take == 16 {
            0xFFFF
        } else {
            (1u64 << take) - 1
        };
        c.alpha |= (movemask(alpha) & valid) << j;
        c.digit |= (movemask(digit) & valid) << j;
        c.ws |= (movemask(ws) & valid) << j;
        c.nl |= (movemask(nl) & valid) << j;
        c.other |= (movemask(other) & valid) << j;
        c.nonascii |= (movemask(na) & valid) << j;
        c.sp |= (movemask(sp) & valid) << j;
        c.ap |= (movemask(ap) & valid) << j;
        j += take;
        p += take;
    }
    c
}

/// The class words the bulk cl100k path needs from one 64-byte block: `other` is
/// derived at the bitmap level (`!(alpha|digit|ws)`), which the pure-ASCII bulk
/// path can do because `nonascii == 0`, so it needs no `other`/`nonascii`/`ap`
/// movemasks — only a single "any bad byte?" reduction. `bad` is set when the
/// block holds a non-ASCII byte or an apostrophe (both send the caller to the
/// scalar fallback).
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct BulkClasses {
    pub alpha: u64,
    pub digit: u64,
    pub ws: u64,
    pub nl: u64,
    pub sp: u64,
    /// Apostrophes (`'`). Classed as "other" for every run rule (so a word prefix or
    /// punct run absorbs one naturally); the separate bitmap only drives the
    /// contraction post-pass, which needs the few apostrophe positions.
    pub ap: u64,
    /// Uppercase ASCII letters (`A`..=`Z`). Only computed for the `CASED` (o200k)
    /// classify — o200k splits letter runs at each lower→upper transition.
    pub up: u64,
    /// ASCII `/`. Only computed for the `CASED` (o200k) classify — o200k's punct
    /// trailing is `[\r\n/]*` (cl100k's is `[\r\n]*`).
    pub slash: u64,
    /// Kimi's transcoded Han stand-in byte (`0x80`), only computed for `HAN`
    /// classifies (where `0x80` is then not "bad").
    pub han: u64,
    /// Only non-ASCII now — apostrophes are handled, not bailed on.
    pub bad: bool,
}

/// NEON bulk classify (aarch64): 5 movemasks + one "any bad" reduction, vs the
/// full kernel's 8 movemasks.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn classify_block_bulk<const CASED: bool, const HAN: bool>(bytes: &[u8], off: usize) -> BulkClasses {
    use std::arch::aarch64::*;
    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn movemask(v: uint8x16_t) -> u64 {
        const POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vandq_u8(v, vld1q_u8(POWERS.as_ptr()));
        (vaddv_u8(vget_low_u8(bits)) as u64) | ((vaddv_u8(vget_high_u8(bits)) as u64) << 8)
    }
    let mut c = BulkClasses::default();
    if off + 64 <= bytes.len() {
        return classify_full_block_bulk::<CASED, HAN>(bytes.as_ptr().add(off));
    }
    let end = (off + 64).min(bytes.len());
    let mut p = off;
    let mut j = 0usize;
    let hi80 = vdupq_n_u8(0x80);
    let mut bad_acc = vdupq_n_u8(0);
    while p < end {
        let take = (end - p).min(16);
        let v = if take == 16 && p + 16 <= bytes.len() {
            vld1q_u8(bytes.as_ptr().add(p))
        } else {
            let mut buf = [0u8; 16];
            buf[..take].copy_from_slice(&bytes[p..p + take]);
            vld1q_u8(buf.as_ptr())
        };
        let ascii = vcltq_u8(v, hi80);
        let lower = vorrq_u8(v, vdupq_n_u8(0x20));
        let alpha = vandq_u8(
            vandq_u8(
                vcgeq_u8(lower, vdupq_n_u8(b'a')),
                vcleq_u8(lower, vdupq_n_u8(b'z')),
            ),
            ascii,
        );
        let digit = vandq_u8(
            vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'0')), vcleq_u8(v, vdupq_n_u8(b'9'))),
            ascii,
        );
        let ws_sp = vceqq_u8(v, vdupq_n_u8(0x20));
        let ws = vandq_u8(
            vorrq_u8(
                vandq_u8(vcgeq_u8(v, vdupq_n_u8(0x09)), vcleq_u8(v, vdupq_n_u8(0x0D))),
                ws_sp,
            ),
            ascii,
        );
        let nl = vandq_u8(
            vorrq_u8(vceqq_u8(v, vdupq_n_u8(0x0A)), vceqq_u8(v, vdupq_n_u8(0x0D))),
            ascii,
        );
        let sp = vandq_u8(ws_sp, ascii);
        // "bad" = non-ASCII only; accumulate, reduce once at the end.
        // Non-ASCII is bad — except Kimi's Han stand-in `0x80` when `HAN`.
        let na = if HAN { vcgtq_u8(v, hi80) } else { vmvnq_u8(ascii) };
        let ap = vceqq_u8(v, vdupq_n_u8(b'\''));
        bad_acc = vorrq_u8(bad_acc, na);

        let valid: u64 = if take == 16 {
            0xFFFF
        } else {
            (1u64 << take) - 1
        };
        c.alpha |= (movemask(alpha) & valid) << j;
        c.digit |= (movemask(digit) & valid) << j;
        c.ws |= (movemask(ws) & valid) << j;
        c.nl |= (movemask(nl) & valid) << j;
        c.sp |= (movemask(sp) & valid) << j;
        if HAN {
            let h = vceqq_u8(v, hi80);
            if vmaxvq_u8(h) != 0 {
                c.han |= (movemask(h) & valid) << j;
            }
        }
        // Apostrophes are ~0.1–0.7 % of bytes, so most 16-byte chunks have none: a
        // cheap horizontal any-check skips the movemask entirely on those chunks.
        if vmaxvq_u8(ap) != 0 {
            c.ap |= (movemask(ap) & valid) << j;
        }
        if CASED {
            // o200k only: uppercase `A`..=`Z` (before the `| 0x20` lowercase fold) and
            // `/`. `lower` already folded case, so recover the upper set from `v`.
            let up = vandq_u8(
                vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'A')), vcleq_u8(v, vdupq_n_u8(b'Z'))),
                ascii,
            );
            let slash = vceqq_u8(v, vdupq_n_u8(b'/'));
            c.up |= (movemask(up) & valid) << j;
            c.slash |= (movemask(slash) & valid) << j;
        }
        j += take;
        p += take;
    }
    c.bad = vmaxvq_u8(bad_acc) != 0;
    c
}

/// [`classify_block_bulk`] for a full 64-byte block: the four 16-byte vectors are
/// classified side by side and each class's 64-bit mask is built in one
/// pairwise-add reduction (`vpaddq` ×4), instead of two horizontal adds per
/// 16-byte chunk — the movemasks were half the cost of the whole bulk scan.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn classify_full_block_bulk<const CASED: bool, const HAN: bool>(ptr: *const u8) -> BulkClasses {
    use std::arch::aarch64::*;
    const POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let pw = vld1q_u8(POWERS.as_ptr());
    // Bit `i` of the result = lane `i % 16` of vector `i / 16` is set.
    let mm = |a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t| -> u64 {
        let s0 = vpaddq_u8(vandq_u8(a, pw), vandq_u8(b, pw));
        let s1 = vpaddq_u8(vandq_u8(c, pw), vandq_u8(d, pw));
        let s = vpaddq_u8(s0, s1);
        vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(s, s)), 0)
    };
    let v = [
        vld1q_u8(ptr),
        vld1q_u8(ptr.add(16)),
        vld1q_u8(ptr.add(32)),
        vld1q_u8(ptr.add(48)),
    ];
    let hi80 = vdupq_n_u8(0x80);
    let classes = |v: uint8x16_t| {
        let ascii = vcltq_u8(v, hi80);
        let lower = vorrq_u8(v, vdupq_n_u8(0x20));
        let alpha = vandq_u8(
            vandq_u8(
                vcgeq_u8(lower, vdupq_n_u8(b'a')),
                vcleq_u8(lower, vdupq_n_u8(b'z')),
            ),
            ascii,
        );
        let digit = vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'0')), vcleq_u8(v, vdupq_n_u8(b'9')));
        let sp = vceqq_u8(v, vdupq_n_u8(0x20));
        let ws = vorrq_u8(
            vandq_u8(vcgeq_u8(v, vdupq_n_u8(0x09)), vcleq_u8(v, vdupq_n_u8(0x0D))),
            sp,
        );
        let nl = vorrq_u8(vceqq_u8(v, vdupq_n_u8(0x0A)), vceqq_u8(v, vdupq_n_u8(0x0D)));
        let ap = vceqq_u8(v, vdupq_n_u8(b'\''));
        (alpha, digit, ws, nl, sp, ap)
    };
    let (a0, d0, w0, n0, s0, p0) = classes(v[0]);
    let (a1, d1, w1, n1, s1, p1) = classes(v[1]);
    let (a2, d2, w2, n2, s2, p2) = classes(v[2]);
    let (a3, d3, w3, n3, s3, p3) = classes(v[3]);
    let any_hi = vmaxq_u8(vmaxq_u8(v[0], v[1]), vmaxq_u8(v[2], v[3]));
    let any_ap = vorrq_u8(vorrq_u8(p0, p1), vorrq_u8(p2, p3));
    let mut c = BulkClasses {
        alpha: mm(a0, a1, a2, a3),
        digit: mm(d0, d1, d2, d3),
        ws: mm(w0, w1, w2, w3),
        nl: mm(n0, n1, n2, n3),
        sp: mm(s0, s1, s2, s3),
        ap: if vmaxvq_u8(any_ap) != 0 {
            mm(p0, p1, p2, p3)
        } else {
            0
        },
        up: 0,
        slash: 0,
        han: 0,
        // Any non-ASCII byte is bad — except Kimi's Han stand-in `0x80` when `HAN`.
        bad: vmaxvq_u8(any_hi) >= if HAN { 0x81 } else { 0x80 },
    };
    if HAN && vmaxvq_u8(any_hi) == 0x80 {
        let h = |v: uint8x16_t| vceqq_u8(v, hi80);
        c.han = mm(h(v[0]), h(v[1]), h(v[2]), h(v[3]));
    }
    if CASED {
        let up =
            |v: uint8x16_t| vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'A')), vcleq_u8(v, vdupq_n_u8(b'Z')));
        let sl = |v: uint8x16_t| vceqq_u8(v, vdupq_n_u8(b'/'));
        c.up = mm(up(v[0]), up(v[1]), up(v[2]), up(v[3]));
        c.slash = mm(sl(v[0]), sl(v[1]), sl(v[2]), sl(v[3]));
    }
    c
}

/// Non-aarch64 bulk classify: the AVX2 kernel where the CPU has it, else the
/// portable one.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn classify_block_bulk<const CASED: bool, const HAN: bool>(bytes: &[u8], off: usize) -> BulkClasses {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512::available() {
            // SAFETY: AVX-512BW support was just detected.
            return unsafe { avx512::classify_block_bulk::<CASED, HAN>(bytes, off) };
        }
        if avx2::available() {
            // SAFETY: AVX2 support was just detected.
            return unsafe { avx2::classify_block_bulk::<CASED, HAN>(bytes, off) };
        }
    }
    classify_block_bulk_portable::<CASED, HAN>(bytes, off)
}

/// Portable fallback: derive `BulkClasses` from the full SWAR classify.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn classify_block_bulk_portable<const CASED: bool, const HAN: bool>(
    bytes: &[u8],
    off: usize,
) -> BulkClasses {
    let c = classify_block_swar(bytes, off);
    let (mut up, mut slash) = (0u64, 0u64);
    if CASED {
        // Portable path: recover uppercase `A`..=`Z` and `/` scalar-wise.
        let end = (off + 64).min(bytes.len());
        for (bit, &b) in bytes[off..end].iter().enumerate() {
            if b.wrapping_sub(b'A') < 26 {
                up |= 1u64 << bit;
            }
            if b == b'/' {
                slash |= 1u64 << bit;
            }
        }
    }
    BulkClasses {
        alpha: c.alpha,
        digit: c.digit,
        ws: c.ws,
        nl: c.nl,
        sp: c.sp,
        ap: c.ap,
        up,
        slash,
        han: if HAN { han_mask(bytes, off) } else { 0 },
        bad: if HAN { c.nonascii & !han_mask(bytes, off) != 0 } else { c.nonascii != 0 },
    }
}

/// Portable: bits of the `0x80` bytes (Kimi's Han stand-in) of the block at `off`.
#[cfg(not(target_arch = "aarch64"))]
fn han_mask(bytes: &[u8], off: usize) -> u64 {
    let end = (off + 64).min(bytes.len());
    bytes[off..end].iter().enumerate().fold(0, |m, (i, &b)| m | ((b == 0x80) as u64) << i)
}

/// Arch-dispatching wrapper for [`classify_block_bulk`]. `CASED` requests the o200k
/// extra masks (uppercase, `/`).
#[inline]
fn unsafe_bulk_classify<const CASED: bool, const HAN: bool>(bytes: &[u8], off: usize) -> BulkClasses {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { classify_block_bulk::<CASED, HAN>(bytes, off) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        classify_block_bulk::<CASED, HAN>(bytes, off)
    }
}

// ── Milestone 2: cl100k boundary scanner over the class bitmaps ──────────────

/// A memoized 64-byte-aligned classified block, so consecutive token boundary
/// queries within the same block reuse one `classify_block`.
struct BlockCache {
    base: usize,
    classes: Classes,
    valid: bool,
}

impl BlockCache {
    #[inline]
    fn new() -> Self {
        Self {
            base: 0,
            classes: Classes::default(),
            valid: false,
        }
    }
    /// The classes for the 64-aligned block containing byte `p`.
    #[inline]
    fn at(&mut self, bytes: &[u8], p: usize) -> &Classes {
        let base = p & !63;
        if !self.valid || self.base != base {
            self.classes = classify_block(bytes, base);
            self.base = base;
            self.valid = true;
        }
        &self.classes
    }
}

/// First position `>= q` where `sel`'s class bit is 0 (i.e. the end of a run of
/// that class starting at `q`), capped at `n`. Scans whole 64-bit class words.
#[inline]
fn run_end(
    cache: &mut BlockCache,
    bytes: &[u8],
    mut q: usize,
    n: usize,
    sel: fn(&Classes) -> u64,
) -> usize {
    while q < n {
        let base = q & !63;
        let m = sel(cache.at(bytes, base));
        let bit = q - base;
        let zeros = !m & (u64::MAX << bit); // positions >= bit where m is 0
        if zeros != 0 {
            return (base + zeros.trailing_zeros() as usize).min(n);
        }
        q = base + 64;
    }
    n
}

/// Position of the last newline in `[p, e)`, or `usize::MAX` if none.
#[inline]
fn last_nl_in(cache: &mut BlockCache, bytes: &[u8], p: usize, e: usize) -> usize {
    if e == p {
        return usize::MAX;
    }
    let mut base = (e - 1) & !63;
    loop {
        let nl = cache.at(bytes, base).nl;
        // Restrict to bits within [max(p,base), min(e, base+64)).
        let lo = p.max(base) - base;
        let hi = (e.min(base + 64)) - base;
        let mask = if hi >= 64 { u64::MAX } else { (1u64 << hi) - 1 } & !((1u64 << lo) - 1);
        let m = nl & mask;
        if m != 0 {
            return base + (63 - m.leading_zeros() as usize);
        }
        if base <= p {
            return usize::MAX;
        }
        base -= 64;
    }
}

/// Length of a cl100k contraction at `bytes[p]` (which must be `'`), or 0 —
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`. Mirrors `scan::contraction_len`.
#[inline]
fn contraction_len(bytes: &[u8], p: usize) -> usize {
    let n = bytes.len();
    if p + 1 >= n || bytes[p + 1] >= 0x80 {
        return 0;
    }
    match bytes[p + 1] | 0x20 {
        b's' | b't' | b'm' | b'd' => 2,
        b'r' | b'v' if p + 2 < n && (bytes[p + 2] | 0x20) == b'e' => 3,
        b'l' if p + 2 < n && (bytes[p + 2] | 0x20) == b'l' => 3,
        _ => 0,
    }
}

#[inline]
fn class_bit(cache: &mut BlockCache, bytes: &[u8], p: usize, sel: fn(&Classes) -> u64) -> bool {
    let base = p & !63;
    (sel(cache.at(bytes, base)) >> (p - base)) & 1 != 0
}

/// Scan `text` (**assumed pure ASCII**) as cl100k, calling `emit(start, end)` for
/// each pretoken. Boundary rules mirror `scan::scan_core`'s cl100k path exactly;
/// this drives them from the SIMD class bitmaps instead of per-byte `char_at` +
/// table lookups. Verified byte-exact against `scan_core` (`cl100k_simd_matches_scalar`).
///
/// Non-ASCII handling (a letter run can straddle the ASCII↔non-ASCII boundary) is
/// milestone 3 (fused-path wiring); callers must pass ASCII text for now.
pub(crate) fn scan_cl100k_ascii<F>(text: &str, mut emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let b = text.as_bytes();
    let n = b.len();
    let cache = &mut BlockCache::new();
    let sel_a = |c: &Classes| c.alpha;
    let sel_d = |c: &Classes| c.digit;
    let sel_o = |c: &Classes| c.other;
    let sel_ws = |c: &Classes| c.ws;

    let mut i = 0usize;
    while i < n {
        let is_a = class_bit(cache, b, i, sel_a);
        let is_d = class_bit(cache, b, i, sel_d);
        let is_ws = class_bit(cache, b, i, sel_ws);
        let is_o = class_bit(cache, b, i, sel_o);

        // 1. Contraction (highest priority), only at a fresh `'`.
        if b[i] == b'\'' {
            let cl = contraction_len(b, i);
            if cl > 0 {
                emit(i, i + cl)?;
                i += cl;
                continue;
            }
        }

        // 2. Word: `[^\r\n\p{L}\p{N}]?\p{L}+`. Prefix-eligible = other | ws-non-newline.
        let next_a = i + 1 < n && class_bit(cache, b, i + 1, sel_a);
        let prefix_eligible = is_o || (is_ws && b[i] != b'\n' && b[i] != b'\r');
        if is_a {
            let e = run_end(cache, b, i, n, sel_a);
            emit(i, e)?;
            i = e;
            continue;
        }
        if prefix_eligible && next_a {
            let e = run_end(cache, b, i + 1, n, sel_a);
            emit(i, e)?;
            i = e;
            continue;
        }

        // 3. Number: `\p{N}{1,3}`.
        if is_d {
            let run = run_end(cache, b, i, n, sel_d);
            let e = run.min(i + 3);
            emit(i, e)?;
            i = e;
            continue;
        }

        // 4. Punct: ` ?[^\s\p{L}\p{N}]+[\r\n]*`.
        let next_o = i + 1 < n && class_bit(cache, b, i + 1, sel_o);
        let pstart = if b[i] == b' ' && next_o { i + 1 } else { i };
        if class_bit(cache, b, pstart, sel_o) {
            let mut e = run_end(cache, b, pstart, n, sel_o);
            // Trailing `[\r\n]*`.
            e = run_end_trailing_nl(cache, b, e, n);
            emit(i, e)?;
            i = e;
            continue;
        }

        // 5. Whitespace: `\s*[\r\n]+ | \s+(?!\S) | \s+`.
        let we = run_end(cache, b, i, n, sel_ws);
        let last_nl = last_nl_in(cache, b, i, we);
        let end = if last_nl != usize::MAX {
            last_nl + 1
        } else if we == n {
            we
        } else if we - 1 > i {
            we - 1 // leave the last whitespace codepoint (all ASCII, 1 byte) for the next token
        } else {
            we
        };
        emit(i, end)?;
        i = end;
    }
    Ok(())
}

/// Advance past a trailing `[\r\n]*` run from `e`.
#[inline]
fn run_end_trailing_nl(cache: &mut BlockCache, bytes: &[u8], mut e: usize, n: usize) -> usize {
    while e < n {
        let base = e & !63;
        let nl = cache.at(bytes, base).nl;
        let bit = e - base;
        // positions >= bit that are NOT nl end the trailing run
        let non_nl = !nl & (u64::MAX << bit);
        if non_nl != 0 {
            let z = non_nl.trailing_zeros() as usize;
            return (base + z).min(n);
        }
        e = base + 64;
    }
    n
}

// ── Milestone 2b: bulk start-bitmap for cl100k (no apostrophes) ───────────────

/// A whole-text bitmap: bit `i` = property of byte `i`, packed 64 bits/word.
struct Bitmap {
    w: Vec<u64>,
}

thread_local! {
    /// Recycled bitmap storage. A bulk scan builds ~10 bitmaps; on mixed text the
    /// bulk scanner runs on many short ASCII stretches between non-ASCII islands,
    /// where fresh allocations would cost as much as the scan itself.
    static BITMAP_POOL: std::cell::RefCell<Vec<Vec<u64>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Bitmaps kept in the pool, and the largest (in words) worth keeping — bigger
/// ones belong to huge single documents, where one allocation is amortized.
const BITMAP_POOL_MAX: usize = 32;
const BITMAP_POOL_MAX_WORDS: usize = 1 << 16;

impl Bitmap {
    fn zeros(n: usize) -> Self {
        Self::zeroed_words(n.div_ceil(64).max(1))
    }

    /// An empty bitmap with room for `n` bits, to be filled word by word with
    /// [`Self::push`] — for bitmaps a pass writes in full, sparing the zero fill.
    fn with_capacity(n: usize) -> Self {
        let mut w = BITMAP_POOL
            .try_with(|p| p.borrow_mut().pop())
            .ok()
            .flatten()
            .unwrap_or_default();
        w.clear();
        w.reserve(n.div_ceil(64).max(1));
        Self { w }
    }

    #[inline(always)]
    fn push(&mut self, word: u64) {
        self.w.push(word);
    }

    fn zeroed_words(nw: usize) -> Self {
        let mut w = BITMAP_POOL
            .try_with(|p| p.borrow_mut().pop())
            .ok()
            .flatten()
            .unwrap_or_default();
        w.clear();
        w.resize(nw, 0);
        Self { w }
    }
    #[inline]
    fn get(&self, i: usize) -> bool {
        (self.w[i >> 6] >> (i & 63)) & 1 != 0
    }
    #[inline]
    #[allow(dead_code)]
    fn set(&mut self, i: usize) {
        self.w[i >> 6] |= 1u64 << (i & 63);
    }
}

impl Drop for Bitmap {
    fn drop(&mut self) {
        let w = std::mem::take(&mut self.w);
        if w.capacity() == 0 || w.capacity() > BITMAP_POOL_MAX_WORDS {
            return;
        }
        let _ = BITMAP_POOL.try_with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < BITMAP_POOL_MAX {
                p.push(w);
            }
        });
    }
}

/// Forward fill: every position of `c` at or after a marker of `m` inside the same
/// run of `c` (Parabix `MatchStar`: adding `c` carries each marker to its run's
/// end). `carry_in` is a virtual marker at bit 0 — the run crossing in from the
/// previous word was already filled. Returns the fill and whether the run touching
/// bit 63 is filled (the next word's carry).
#[inline(always)]
fn fill_forward(m: u64, c: u64, carry_in: bool) -> (u64, bool) {
    let m = (m | carry_in as u64) & c;
    let fill = (m.wrapping_add(c) ^ c | m) & c;
    (fill, fill >> 63 != 0)
}

/// Punct trailing run (`[\r\n]*`, or `[\r\n/]*` for o200k): a `class` run that
/// begins with a newline immediately preceded by an `other` char belongs to that
/// punct token. Word-parallel (was a per-newline scalar walk, whose cost scaled
/// with the newline count — 6× the bitstream cost on line-heavy documents).
fn punct_trailing_runs(nl: &Bitmap, o: &Bitmap, class: &Bitmap) -> Bitmap {
    let nwords = nl.w.len();
    let mut out = Bitmap::zeroed_words(nwords);
    let (mut carry, mut nl_hi, mut o_hi) = (false, 0u64, 0u64);
    for w in 0..nwords {
        let (nlw, ow) = (nl.w[w], o.w[w]);
        // A newline that starts a newline run and follows an `other` char.
        let start = nlw & !((nlw << 1) | nl_hi) & ((ow << 1) | o_hi);
        let (fill, co) = fill_forward(start, class.w[w], carry);
        out.w[w] = fill;
        carry = co;
        nl_hi = nlw >> 63;
        o_hi = ow >> 63;
    }
    out
}

/// `\s*[\r\n]+` newline split: a token opens right after a whitespace run's last
/// newline. Position `q` opens one iff byte `q-1` is a newline, `q` is whitespace
/// content, and no newline lies at or after `q` within `q`'s `(ws & !trailing)`
/// run. That last test is a *backward* fill, so it runs on bit-reversed words from
/// the last word down, carrying "the run entering from above has a later newline".
fn newline_split(
    nl: &Bitmap,
    ws: &Bitmap,
    trailing: &Bitmap,
    ws_content: &Bitmap,
    starts: &mut Bitmap,
) {
    let nwords = nl.w.len();
    let mut carry = false;
    for w in (0..nwords).rev() {
        let after_nl = (nl.w[w] << 1) | if w > 0 { nl.w[w - 1] >> 63 } else { 0 };
        // No newline here and none later in a run entering from above: nothing
        // to fill back (and no carry out) — skip the bit reversals.
        if nl.w[w] == 0 && !carry {
            starts.w[w] |= after_nl & ws_content.w[w];
            continue;
        }
        let run = ws.w[w] & !trailing.w[w];
        let (rev_fill, co) =
            fill_forward((nl.w[w] & run).reverse_bits(), run.reverse_bits(), carry);
        carry = co;
        let later_nl = rev_fill.reverse_bits();
        starts.w[w] |= after_nl & ws_content.w[w] & !later_nl;
    }
}

/// ASCII stretches shorter than this between two non-ASCII islands are scanned
/// with the island (the bulk scanner's per-call setup would dominate).
const MIN_BULK_STRETCH: usize = 256;

/// Index of the first byte `>= 0x80` in `b`, 8 bytes at a time.
#[inline]
fn first_non_ascii(b: &[u8]) -> Option<usize> {
    const HI: u64 = 0x8080_8080_8080_8080;
    let mut i = 0;
    while i + 8 <= b.len() {
        let w = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        if w & HI != 0 {
            return Some(i + ((w & HI).trailing_zeros() / 8) as usize);
        }
        i += 8;
    }
    b[i..].iter().position(|&x| x >= 0x80).map(|r| i + r)
}

/// Is `q` a guaranteed cl100k/o200k pretoken boundary independent of any text
/// outside `b[q-1..=q+1]`? True when `b[q]` is a space, `b[q+1]` an ASCII letter
/// and `b[q-1]` ASCII non-whitespace. The pretoken holding `b[q-1]` must end there
/// — letter/number runs, contractions and punct runs cannot absorb a space (punct
/// trailing is only `[\r\n]` / `[\r\n/]`) and `b[q-1]` is not whitespace — and
/// ` word` opens the next one. Neither grammar has look-behind, and a piece ending
/// at `q` ends on non-whitespace, so `\s+(?!\S)` inside it sees the same next
/// char as in the whole text: pieces cut at such positions scan independently.
#[inline]
fn is_word_cut(b: &[u8], q: usize) -> bool {
    q > 0
        && q + 1 < b.len()
        && b[q] == b' '
        && b[q + 1].is_ascii_alphabetic()
        && b[q - 1] < 0x80
        && !b[q - 1].is_ascii_whitespace()
}

/// The last cut point in `(lo, p]`, else `lo` (itself a boundary).
#[inline]
fn cut_before(b: &[u8], lo: usize, p: usize) -> usize {
    let mut q = p;
    while q > lo {
        if is_word_cut(b, q) {
            return q;
        }
        q -= 1;
    }
    lo
}

/// The first cut point after `p` (a non-ASCII byte), else `b.len()`.
#[inline]
fn cut_after(b: &[u8], p: usize) -> usize {
    let mut q = p + 1;
    while q < b.len() {
        if is_word_cut(b, q) {
            return q;
        }
        q += 1;
    }
    b.len()
}

/// A whole-text start bitmap from a bulk scanner, for callers that walk the spans
/// themselves ([`BulkStarts::spans`]) instead of taking a per-span callback — so
/// the per-span work inlines into one loop. Over transcoded text the bitmap is in
/// char offsets and `mb` maps them to bytes (see [`emit_spans_mapped`]); the
/// buffers go back to the thread-local pool on drop.
pub(crate) struct BulkStarts {
    starts: Bitmap,
    mb: Vec<MbRun>,
    buf: Vec<u8>,
    pooled: bool,
}

impl Drop for BulkStarts {
    fn drop(&mut self) {
        if self.pooled {
            TRANSCODE.set((std::mem::take(&mut self.buf), std::mem::take(&mut self.mb)));
        }
    }
}

impl BulkStarts {
    /// The `(start, end)` byte spans, in order, for a text of `n_bytes`.
    #[inline(always)]
    /// The start bitmap's words and the multibyte runs its bit offsets map
    /// through ([`ByteMap`]; empty when the text was scanned as is, so bit
    /// offsets are byte offsets). Bit `i` set ⟺ a span starts at char offset `i`;
    /// the spans run from each start to the next, the last to the end.
    pub(crate) fn starts_and_runs(&self) -> (&[u64], &[MbRun]) {
        (&self.starts.w, &self.mb)
    }

    pub(crate) fn spans(&self, n_bytes: usize) -> SpanIter<'_> {
        let w = &self.starts.w[..];
        SpanIter {
            w,
            wk: 0,
            bits: w.first().copied().unwrap_or(0),
            prev: usize::MAX,
            n: n_bytes,
            map: ByteMap::new(&self.mb),
        }
    }
}

/// Iterator behind [`BulkStarts::spans`]: consecutive start bits, mapped to byte
/// offsets through the (possibly empty) multibyte runs.
pub(crate) struct SpanIter<'a> {
    w: &'a [u64],
    wk: usize,
    bits: u64,
    prev: usize,
    n: usize,
    map: ByteMap<'a>,
}

impl Iterator for SpanIter<'_> {
    type Item = (usize, usize);

    #[inline(always)]
    fn next(&mut self) -> Option<(usize, usize)> {
        loop {
            if self.bits != 0 {
                let c = self.wk * 64 + self.bits.trailing_zeros() as usize;
                self.bits &= self.bits - 1;
                let b = self.map.byte(c);
                let p = std::mem::replace(&mut self.prev, b);
                if p != usize::MAX {
                    return Some((p, b));
                }
                continue;
            }
            self.wk += 1;
            if self.wk >= self.w.len() {
                // The last span runs to the end of the text (once).
                let p = std::mem::replace(&mut self.prev, usize::MAX);
                return (p != usize::MAX).then_some((p, self.n));
            }
            self.bits = self.w[self.wk];
        }
    }
}

/// The start bitmap of `text` under `kind` when one bulk pass covers all of it —
/// pure ASCII for every bulk grammar, and any text for cl100k / Qwen / DeepSeek
/// (transcoded) — else `None` (o200k with non-ASCII, other grammars: use
/// [`scan_fast`]). Same spans as [`scan_fast`].
pub(crate) fn bulk_starts(kind: super::scan::ScanKind, text: &str) -> Option<BulkStarts> {
    use super::scan::ScanKind;
    let b = text.as_bytes();
    let plain = |starts: Bitmap| BulkStarts {
        starts,
        mb: Vec::new(),
        buf: Vec::new(),
        pooled: false,
    };
    let Some(first_na) = first_non_ascii(b) else {
        return match kind {
            ScanKind::Cl100k => cl100k_bulk_starts::<3>(text).map(plain),
            ScanKind::Qwen | ScanKind::Qwen35 => cl100k_bulk_starts::<1>(text).map(plain),
            ScanKind::O200k => o200k_bulk_starts::<true, true, 3, false>(b).map(plain),
            ScanKind::Tekken => o200k_bulk_starts::<true, false, 1, false>(b).map(plain),
            // (Pure ASCII holds no Han char: Kimi's grammar is o200k's here.)
            ScanKind::Kimi => o200k_bulk_starts::<false, true, 3, false>(b).map(plain),
            ScanKind::DeepSeek => deepseek_bulk_starts(b, false).map(plain),
            ScanKind::Gpt2 => gpt2_bulk_starts(b).map(plain),
        };
    };
    let t = super::unicode_class::tables();
    let (mut buf, mut mb) = TRANSCODE.take();
    let starts = match kind {
        ScanKind::Cl100k | ScanKind::Qwen | ScanKind::Qwen35 => {
            if kind == ScanKind::Qwen35 {
                transcode(b, first_na, &mut buf, &mut mb, |cp| qwen35_rep(t, cp));
            } else {
                transcode(b, first_na, &mut buf, &mut mb, cl100k_reps(t));
            }
            // SAFETY: every byte pushed is ASCII.
            let ascii = unsafe { std::str::from_utf8_unchecked(&buf) };
            if kind == ScanKind::Cl100k {
                cl100k_bulk_starts::<3>(ascii)
            } else {
                cl100k_bulk_starts::<1>(ascii)
            }
        }
        ScanKind::DeepSeek => {
            transcode(b, first_na, &mut buf, &mut mb, deepseek_reps(t));
            deepseek_bulk_starts(&buf, true)
        }
        ScanKind::Gpt2 => {
            // GPT-2 sees non-ASCII chars only as `\p{L}` / `\p{N}` / `\s` / other,
            // and its contractions and space prefix are ASCII — cl100k's stand-ins.
            transcode(b, first_na, &mut buf, &mut mb, cl100k_reps(t));
            gpt2_bulk_starts(&buf)
        }
        // The o200k family has a whole-text pass only when no char lacks a stand-in
        // (else scalar islands, which feed `scan_fast`'s callback faster than a
        // bitmap — measured). Transcoding stops at the first such char.
        ScanKind::O200k | ScanKind::Tekken | ScanKind::Kimi => {
            let kimi = kind == ScanKind::Kimi;
            let mut side = O200K_SIDE.take();
            let full = transcode_opt::<true>(
                b,
                first_na,
                &mut buf,
                &mut mb,
                &mut side,
                o200k_family_reps(t, kimi),
            );
            let starts = if !full {
                None
            } else {
                match kind {
                    ScanKind::O200k => o200k_bulk_starts_side::<true, true, 3, false>(&buf, &side),
                    ScanKind::Tekken => {
                        o200k_bulk_starts_side::<true, false, 1, false>(&buf, &side)
                    }
                    // Without a Han stand-in the text holds no Han char: o200k's
                    // grammar again (the common case of a few quotes or accents).
                    _ if memchr::memchr(0x80, &buf).is_none() => {
                        o200k_bulk_starts_side::<false, true, 3, false>(&buf, &side)
                    }
                    _ => o200k_bulk_starts_side::<false, true, 3, true>(&buf, &side),
                }
            };
            O200K_SIDE.set(side);
            starts
        }
    };
    match starts {
        Some(starts) => Some(BulkStarts {
            starts,
            mb,
            buf,
            pooled: true,
        }),
        None => {
            TRANSCODE.set((buf, mb));
            None
        }
    }
}

/// Qwen3.5's stand-in: [`cl100k_rep`] with marks as letters (`a`).
#[inline(always)]
fn qwen35_rep(t: &super::unicode_class::Tables, cp: u32) -> u8 {
    if !t.is_letter(cp) && t.is_ugroup(cp) && t.is_lgroup(cp) {
        b'a'
    } else {
        cl100k_rep(t, cp)
    }
}

/// cl100k's one-byte stand-in for a non-ASCII char (see [`scan_cl100k_transcoded`]).
#[inline(always)]
fn cl100k_rep(t: &super::unicode_class::Tables, cp: u32) -> u8 {
    if t.is_letter(cp) {
        b'a'
    } else if t.is_number(cp) {
        b'0'
    } else if t.is_ws(cp) {
        b'\t'
    } else {
        b'!'
    }
}

/// A per-codepoint stand-in for the BMP as a two-level table: 256-codepoint
/// blocks, identical blocks shared (whole scripts — the CJK ideographs, Hangul —
/// are one block each), so the hot part of it stays in L1 where the class
/// bitsets it replaces are 139 KiB apiece. Astral chars use the function itself.
struct RepTable {
    index: [u16; 256],
    blocks: Box<[[u8; 256]]>,
}

impl RepTable {
    fn build(rep: impl Fn(u32) -> u8) -> Self {
        let mut index = [0u16; 256];
        let mut blocks: Vec<[u8; 256]> = Vec::new();
        let mut seen: std::collections::HashMap<[u8; 256], u16> = Default::default();
        for (hi, slot) in index.iter_mut().enumerate() {
            let mut blk = [0u8; 256];
            for (lo, v) in blk.iter_mut().enumerate() {
                let cp = (hi << 8 | lo) as u32;
                // Surrogates are no chars; ASCII never reaches a stand-in.
                if cp >= 0x80 && !(0xD800..0xE000).contains(&cp) {
                    *v = rep(cp);
                }
            }
            *slot = *seen.entry(blk).or_insert_with(|| {
                blocks.push(blk);
                (blocks.len() - 1) as u16
            });
        }
        Self {
            index,
            blocks: blocks.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn get(&self, cp: u32) -> u8 {
        debug_assert!(cp < 0x10000);
        let b = self.index[(cp >> 8) as usize & 0xFF] as usize;
        // SAFETY: `index` holds indices into `blocks`.
        unsafe { self.blocks.get_unchecked(b)[cp as usize & 0xFF] }
    }
}

/// The o200k family's stand-in for `cp` (Kimi's when `kimi`): [`REP_BOTH`] or
/// [`REP_MARK`] for a char in both case classes (see [`transcode_opt`]'s side
/// bits), `None` for one nothing can represent (Kimi's Han non-letters, see
/// [`o200k_hard`]) — through a [`RepTable`] for the BMP (`0xFF` marks those).
#[cfg(test)]
fn o200k_family_rep(t: &super::unicode_class::Tables, cp: u32, kimi: bool) -> Option<u8> {
    o200k_family_reps(t, kimi)(cp)
}

/// [`o200k_family_rep`] with its table fetched once, for a per-char loop.
#[inline(always)]
fn o200k_family_reps(
    t: &super::unicode_class::Tables,
    kimi: bool,
) -> impl Fn(u32) -> Option<u8> + '_ {
    fn direct(t: &super::unicode_class::Tables, cp: u32, kimi: bool) -> Option<u8> {
        if kimi && t.is_han(cp) {
            t.is_letter(cp).then_some(0x80)
        } else if t.is_ugroup(cp) && t.is_lgroup(cp) {
            Some(if t.is_letter(cp) { REP_BOTH } else { REP_MARK })
        } else {
            Some(o200k_easy_rep(t, cp))
        }
    }
    const HARD: u8 = 0xFF;
    static TABLES: [std::sync::OnceLock<RepTable>; 2] = [const { std::sync::OnceLock::new() }; 2];
    let table = TABLES[kimi as usize].get_or_init(|| {
        RepTable::build(|cp| direct(super::unicode_class::tables(), cp, kimi).unwrap_or(HARD))
    });
    move |cp| {
        if cp >= 0x10000 {
            return direct(t, cp, kimi);
        }
        match table.get(cp) {
            HARD => None,
            r => Some(r),
        }
    }
}

/// [`cl100k_rep`] through a [`RepTable`].
#[inline(always)]
fn cl100k_rep_fast(t: &super::unicode_class::Tables, cp: u32) -> u8 {
    cl100k_reps(t)(cp)
}

/// [`cl100k_rep_fast`] with its table fetched once, for a per-char loop.
#[inline(always)]
fn cl100k_reps(t: &super::unicode_class::Tables) -> impl Fn(u32) -> u8 + '_ {
    static TABLE: std::sync::OnceLock<RepTable> = std::sync::OnceLock::new();
    let table =
        TABLE.get_or_init(|| RepTable::build(|cp| cl100k_rep(super::unicode_class::tables(), cp)));
    move |cp| {
        if cp < 0x10000 {
            table.get(cp)
        } else {
            cl100k_rep(t, cp)
        }
    }
}

/// [`deepseek_rep`] through a [`RepTable`].
#[cfg(test)]
fn deepseek_rep_fast(t: &super::unicode_class::Tables, cp: u32) -> u8 {
    deepseek_reps(t)(cp)
}

/// [`deepseek_rep_fast`] with its table fetched once, for a per-char loop.
#[inline(always)]
fn deepseek_reps(t: &super::unicode_class::Tables) -> impl Fn(u32) -> u8 + '_ {
    static TABLE: std::sync::OnceLock<RepTable> = std::sync::OnceLock::new();
    let table = TABLE
        .get_or_init(|| RepTable::build(|cp| deepseek_rep(super::unicode_class::tables(), cp)));
    move |cp| {
        if cp < 0x10000 {
            table.get(cp)
        } else {
            deepseek_rep(t, cp)
        }
    }
}

/// DeepSeek's one-byte stand-in for a non-ASCII char (see [`scan_deepseek_transcoded`]).
#[inline(always)]
fn deepseek_rep(t: &super::unicode_class::Tables, cp: u32) -> u8 {
    let run = t.is_ugroup(cp) || t.is_lgroup(cp);
    if t.is_number(cp) {
        b'0'
    } else if super::scan::is_ds_cjk(char::from_u32(cp).unwrap_or('\0')) {
        if run {
            DS_CL
        } else if t.is_psym(cp) {
            DS_CP
        } else {
            DS_CG
        }
    } else if run {
        DS_VL
    } else if t.is_psym(cp) {
        DS_VP
    } else if t.is_ws(cp) {
        b'\t'
    } else {
        0x01
    }
}

/// Scan `text` into pretoken spans with the fastest byte-exact scanner for its
/// grammar and content, all bit-parallel except where noted:
/// - pure ASCII: the grammar's bulk scanner over the text itself;
/// - cl100k / Qwen / DeepSeek with non-ASCII: the bulk scanner over a one-byte-
///   per-char stand-in ([`scan_cl100k_transcoded`], [`scan_deepseek_transcoded`]);
/// - o200k with non-ASCII: scalar [`scan::scan_core`] islands around each
///   non-ASCII run, bulk-scanned ASCII stretches between them;
/// - other grammars (Kimi): [`scan::scan_core`].
///
/// `emit` receives offsets relative to `text`.
pub(crate) fn scan_fast<F>(
    kind: super::scan::ScanKind,
    text: &str,
    mut emit: F,
) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    use super::scan::{ScanKind, scan_core};
    if kind == ScanKind::DeepSeek {
        return match first_non_ascii(text.as_bytes()) {
            None => {
                let ok = try_scan_deepseek_bulk(text.as_bytes(), false, emit)?;
                debug_assert!(ok);
                Ok(())
            }
            Some(first_na) => scan_deepseek_transcoded(text, first_na, emit),
        };
    }
    if matches!(kind, ScanKind::Gpt2 | ScanKind::Qwen35) {
        let bs = bulk_starts(kind, text).expect("a whole-text bulk pass");
        for (s, e) in bs.spans(text.len()) {
            emit(s, e)?;
        }
        return Ok(());
    }
    if !matches!(
        kind,
        ScanKind::Cl100k | ScanKind::Qwen | ScanKind::O200k | ScanKind::Tekken | ScanKind::Kimi
    ) {
        return scan_core(kind, text, emit);
    }
    let n = text.len();
    let try_bulk = |piece: &str, base: usize, emit: &mut F| -> Result<bool, String> {
        let e = |a: usize, b: usize| emit(a + base, b + base);
        let bytes = piece.as_bytes();
        let st = match kind {
            ScanKind::Cl100k => return try_scan_cl100k_bulk_cap::<3, _>(piece, e),
            ScanKind::Qwen => return try_scan_cl100k_bulk_cap::<1, _>(piece, e),
            ScanKind::Tekken => o200k_bulk_starts::<true, false, 1, false>(bytes),
            ScanKind::Kimi => o200k_bulk_starts::<false, true, 3, true>(bytes),
            _ => o200k_bulk_starts::<true, true, 3, false>(bytes),
        };
        match st {
            None => Ok(false),
            Some(st) => emit_spans(&st, bytes.len(), e).map(|()| true),
        }
    };
    let b = text.as_bytes();
    let Some(first_na) = first_non_ascii(b) else {
        let ok = try_bulk(text, 0, &mut emit)?;
        debug_assert!(ok);
        return Ok(());
    };
    match kind {
        ScanKind::Cl100k => return scan_cl100k_transcoded::<3, F>(text, first_na, emit),
        ScanKind::Qwen => return scan_cl100k_transcoded::<1, F>(text, first_na, emit),
        _ => {}
    }
    // o200k with non-ASCII. Most non-ASCII chars have an exact one-byte ASCII
    // stand-in under o200k's classes (see [`o200k_easy_rep`]), so ASCII-plus-those
    // stretches still take the bulk scanner via transcoding. Caseless letters and
    // marks (`\p{Lm}\p{Lo}\p{M}`: CJK, most scripts' vowel signs) sit in *both* of
    // the pattern's case classes — no ASCII byte does — so a small scalar *island*
    // is carved around each, cut at guaranteed pretoken boundaries (see
    // [`is_word_cut`]).
    let t = super::unicode_class::tables();
    let kimi = kind == ScanKind::Kimi;
    let mut pos = 0usize;
    let mut q = first_na;
    loop {
        let Some(p) = next_o200k_hard(t, b, q, kimi) else {
            scan_o200k_easy(kind, t, text, pos, n, &mut emit)?;
            break;
        };
        let mut c0 = cut_before(b, pos, p);
        let c1 = cut_after(b, p);
        // A stretch too short to amortize the bulk scanner's setup joins the island.
        if c0 - pos < MIN_BULK_STRETCH {
            c0 = pos;
        }
        if c0 > pos {
            scan_o200k_easy(kind, t, text, pos, c0, &mut emit)?;
        }
        scan_core(kind, &text[c0..c1], |a, e| emit(a + c0, e + c0))?;
        pos = c1;
        q = c1;
        if pos >= n {
            break;
        }
    }
    Ok(())
}

/// First char at or after byte `q` of `b` with no one-byte stand-in ([`o200k_hard`]).
#[inline]
fn next_o200k_hard(
    t: &super::unicode_class::Tables,
    b: &[u8],
    mut q: usize,
    kimi: bool,
) -> Option<usize> {
    while q < b.len() {
        q += first_non_ascii(&b[q..])?;
        let (cp, len) = decode_multibyte(b, q);
        if o200k_hard(t, cp, kimi) {
            return Some(q);
        }
        q += len;
    }
    None
}

/// Whether a (non-ASCII) char has no one-byte stand-in under the o200k family: it is
/// in *both* case classes (`\p{Lm}\p{Lo}\p{M}`), which no ASCII byte is. For Kimi,
/// a Han *letter* has one ([`kimi_rep`]), but any other Han-script char (radicals
/// `\p{So}`, numerals like 〇 `\p{Nl}`) does not: the `[\p{Han}]+` arm takes it at a
/// token start, yet a punct run (`[^\s\p{L}\p{N}]+`) or digit run absorbs it.
#[inline(always)]
fn o200k_hard(t: &super::unicode_class::Tables, cp: u32, kimi: bool) -> bool {
    if kimi && t.is_han(cp) {
        return !t.is_letter(cp);
    }
    t.is_ugroup(cp) && t.is_lgroup(cp)
}

/// Kimi's stand-in: `0x80` for a Han letter, else [`o200k_easy_rep`]. A Han letter
/// is matched only by the leading `[\p{Han}]+` arm — Han is subtracted from both
/// letter classes, and a letter is neither prefix, punct nor number — so all the
/// grammar sees is "Han"; [`o200k_bulk_starts`] with `HAN` gives `0x80` exactly
/// that class. (Not for other Han-script chars: see [`o200k_hard`].)
#[inline(always)]
fn kimi_rep(t: &super::unicode_class::Tables, cp: u32) -> u8 {
    if t.is_han(cp) && t.is_letter(cp) { 0x80 } else { o200k_easy_rep(t, cp) }
}

/// The ASCII stand-in of a non-ASCII char with exactly one's o200k classes: `A` for
/// `\p{Lu}\p{Lt}` (upper class only), `a` for `\p{Ll}` (lower class only), `0` for
/// `\p{N}`, `\t` for other whitespace, `!` for the rest (`[^\s\p{L}\p{N}]`, not
/// in either case class). The contraction and `[\r\n/]` tail are ASCII-only, and
/// neither `A` nor `a` is a contraction letter. Not for chars in both case classes.
#[inline(always)]
fn o200k_easy_rep(t: &super::unicode_class::Tables, cp: u32) -> u8 {
    if t.is_ugroup(cp) {
        b'A'
    } else if t.is_lgroup(cp) {
        b'a'
    } else if t.is_number(cp) {
        b'0'
    } else if t.is_ws(cp) {
        b'\t'
    } else {
        b'!'
    }
}

/// Bulk-scan `text[lo..hi]` (a piece between pretoken boundaries holding no hard
/// char, see [`next_o200k_hard`]) under the o200k-family `kind`, transcoding its
/// non-ASCII chars via [`o200k_easy_rep`].
fn scan_o200k_easy<F>(
    kind: super::scan::ScanKind,
    t: &super::unicode_class::Tables,
    text: &str,
    lo: usize,
    hi: usize,
    emit: &mut F,
) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    use super::scan::ScanKind;
    let piece = &text.as_bytes()[lo..hi];
    let e = |a: usize, b: usize| emit(a + lo, b + lo);
    let starts_of = |bytes: &[u8]| match kind {
        ScanKind::Tekken => o200k_bulk_starts::<true, false, 1, false>(bytes),
        ScanKind::Kimi => o200k_bulk_starts::<false, true, 3, true>(bytes),
        _ => o200k_bulk_starts::<true, true, 3, false>(bytes),
    };
    let Some(first_na) = first_non_ascii(piece) else {
        let starts = starts_of(piece).expect("ASCII piece");
        return emit_spans(&starts, piece.len(), e);
    };
    let (mut buf, mut mb) = TRANSCODE.take();
    let kimi = kind == ScanKind::Kimi;
    transcode(piece, first_na, &mut buf, &mut mb, |cp| {
        if kimi { kimi_rep(t, cp) } else { o200k_easy_rep(t, cp) }
    });
    let starts = starts_of(&buf).expect("transcoded text is ASCII");
    let r = emit_spans_mapped(&starts, &mb, piece.len(), e);
    TRANSCODE.set((buf, mb));
    r
}

thread_local! {
    /// Reused buffers for the transcoded scans (see [`transcode`]): the stand-in
    /// text and its multibyte runs.
    static TRANSCODE: std::cell::Cell<TranscodeBufs> =
        const { std::cell::Cell::new((Vec::new(), Vec::new())) };
}

/// The stand-in text of a transcoded scan and its `(char index, extra bytes)`
/// multibyte table (see [`transcode`]).
type TranscodeBufs = (Vec<u8>, Vec<MbRun>);

/// Bits over a transcoded text's stand-in bytes (see [`transcode_opt`]): chars in
/// both o200k case classes (`\p{Lm}\p{Lo}\p{M}`, written as `a`), and which of
/// them are marks (`\p{M}`, no letter).
type SideBits = (Vec<u64>, Vec<u64>);

/// [`o200k_family_rep`]'s codes (never stand-in bytes) for a char in both case
/// classes: a letter (`\p{Lm}\p{Lo}`), or a mark (`\p{M}`).
const REP_BOTH: u8 = 0xFE;
const REP_MARK: u8 = 0xFD;

thread_local! {
    /// [`SideBits`] storage of the o200k family's transcoded scans, reused.
    static O200K_SIDE: std::cell::Cell<SideBits> = const { std::cell::Cell::new((Vec::new(), Vec::new())) };
}

/// A run of consecutive multibyte chars of one width in a transcoded text: `n`
/// chars from char index `c`, each `w1 + 1` bytes; `acc` = extra bytes (beyond one
/// per char) of all multibyte chars before the run. CJK text is long runs of
/// 3-byte chars, so this stays small where a per-char table would not.
#[derive(Clone, Copy, Default)]
pub(crate) struct MbRun {
    c: u32,
    n: u32,
    w1: u32,
    acc: u32,
}

/// Maps ascending char offsets of a transcoded text to byte offsets of the
/// original, walking its [`MbRun`]s once.
pub(crate) struct ByteMap<'a> {
    runs: &'a [MbRun],
    k: usize,
    /// Extra bytes of all runs already passed.
    done: u32,
}

impl<'a> ByteMap<'a> {
    #[inline(always)]
    pub(crate) fn new(runs: &'a [MbRun]) -> Self {
        Self {
            runs,
            k: 0,
            done: 0,
        }
    }

    /// Byte offset of char offset `c` (`c` never below the previous call's).
    #[inline(always)]
    pub(crate) fn byte(&mut self, c: usize) -> usize {
        let c = c as u32;
        while let Some(r) = self.runs.get(self.k) {
            if r.c + r.n <= c {
                self.done = r.acc + r.n * r.w1;
                self.k += 1;
                continue;
            }
            if r.c <= c {
                return (c + r.acc + (c - r.c) * r.w1) as usize;
            }
            break;
        }
        (c + self.done) as usize
    }
}

/// Decode the (valid, multibyte) UTF-8 char whose lead byte is `b[i]`.
#[inline(always)]
fn decode_multibyte(b: &[u8], i: usize) -> (u32, usize) {
    let c0 = b[i] as u32;
    if c0 < 0xE0 {
        (((c0 & 0x1F) << 6) | (b[i + 1] as u32 & 0x3F), 2)
    } else if c0 < 0xF0 {
        (
            ((c0 & 0x0F) << 12) | ((b[i + 1] as u32 & 0x3F) << 6) | (b[i + 2] as u32 & 0x3F),
            3,
        )
    } else {
        (
            ((c0 & 0x07) << 18)
                | ((b[i + 1] as u32 & 0x3F) << 12)
                | ((b[i + 2] as u32 & 0x3F) << 6)
                | (b[i + 3] as u32 & 0x3F),
            4,
        )
    }
}

/// Fill `buf` with the one-byte-per-char stand-in of `b` (whose first non-ASCII
/// byte is at `first_na`): ASCII is copied, each non-ASCII char becomes
/// `rep(codepoint)`. `runs` records where the multibyte chars were ([`MbRun`]), so
/// a [`ByteMap`] can take char offsets back to bytes.
#[inline(always)]
fn transcode(
    b: &[u8],
    first_na: usize,
    buf: &mut Vec<u8>,
    runs: &mut Vec<MbRun>,
    mut rep: impl FnMut(u32) -> u8,
) {
    let full = transcode_opt::<false>(b, first_na, buf, runs, &mut SideBits::default(), |cp| {
        Some(rep(cp))
    });
    debug_assert!(full);
}

/// [`transcode`] with a stand-in that may refuse a char (`None`): stops there and
/// returns `false` (the buffers are then to be discarded).
///
/// Non-Latin text comes in stretches of one char width (2 bytes: Cyrillic,
/// Greek, Hebrew, Arabic; 3: CJK, Devanagari, Thai, Ethiopic), each decoded with
/// that width's fixed shifts and recorded as a single [`MbRun`] — the runs a
/// char-at-a-time walk would merge them into.
#[inline(always)]
fn transcode_opt<const SIDE: bool>(
    b: &[u8],
    first_na: usize,
    buf: &mut Vec<u8>,
    runs: &mut Vec<MbRun>,
    side: &mut SideBits,
    mut rep: impl FnMut(u32) -> Option<u8>,
) -> bool {
    buf.clear();
    runs.clear();
    if SIDE {
        // One bit per stand-in byte (at most one per input byte).
        let nw = b.len().div_ceil(64);
        for v in [&mut side.0, &mut side.1] {
            v.clear();
            v.resize(nw, 0);
        }
    }
    // One stand-in byte per input char at most: written through `out` into this
    // room, the length set once at the end (per-char pushes cost a check each).
    buf.reserve(b.len());
    let out = buf.as_mut_ptr();
    let mut o = first_na;
    // SAFETY: `first_na <= b.len()` bytes fit the reserved room.
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), out, first_na) };
    let mut acc = 0u32;
    let mut i = first_na;
    // The chars of one width from `i` on: stand-ins written, `i` past them, count.
    // SAFETY (of the unchecked reads): `b` is UTF-8 and `i` a char boundary, so a
    // lead byte of width `w` at `i` has its `w - 1` continuation bytes in bounds;
    // (of the writes) `o <= i` throughout, as every char is at least a byte.
    macro_rules! stretch {
        ($w:literal, $mask:literal, $lead:literal, |$at:ident| $cp:expr) => {{
            let mut n = 0u32;
            while i < b.len() && unsafe { *b.get_unchecked(i) } & $mask == $lead {
                let $at = |k: usize| unsafe { *b.get_unchecked(i + k) } as u32;
                let Some(mut r) = rep($cp) else { return false };
                if SIDE && r >= REP_MARK {
                    // A char in both o200k case classes: `a`, marked in `side`.
                    let (w, bit) = (o >> 6, 1u64 << (o & 63));
                    // SAFETY: `o < b.len()`, and the bitmaps hold `b.len()` bits.
                    unsafe {
                        *side.0.get_unchecked_mut(w) |= bit;
                        if r == REP_MARK {
                            *side.1.get_unchecked_mut(w) |= bit;
                        }
                    }
                    r = b'a';
                }
                unsafe { *out.add(o) = r };
                o += 1;
                i += $w;
                n += 1;
            }
            n
        }};
    }
    loop {
        while i < b.len() && b[i] >= 0x80 {
            let c = o as u32;
            let (w1, n) = if b[i] < 0xE0 {
                let n = stretch!(2, 0xE0, 0xC0, |at| ((at(0) & 0x1F) << 6) | (at(1) & 0x3F));
                (1, n)
            } else if b[i] < 0xF0 {
                let n = stretch!(3, 0xF0, 0xE0, |at| ((at(0) & 0x0F) << 12)
                    | ((at(1) & 0x3F) << 6)
                    | (at(2) & 0x3F));
                (2, n)
            } else {
                let n = stretch!(4, 0xF8, 0xF0, |at| ((at(0) & 0x07) << 18)
                    | ((at(1) & 0x3F) << 12)
                    | ((at(2) & 0x3F) << 6)
                    | (at(3) & 0x3F));
                (3, n)
            };
            runs.push(MbRun { c, n, w1, acc });
            acc += n * w1;
        }
        // The ASCII between: byte by byte while short (between the words of a
        // script, mostly a space or two), then a word-parallel search.
        let gap = i;
        while i < b.len() && i - gap < 8 && b[i] < 0x80 {
            // SAFETY: `o <= i < b.len()`.
            unsafe { *out.add(o) = b[i] };
            o += 1;
            i += 1;
        }
        if i < b.len() && b[i] < 0x80 {
            let r = first_non_ascii(&b[i..]).unwrap_or(b.len() - i);
            // SAFETY: `o <= i` and `i + r <= b.len()`, so the copy fits the room.
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr().add(i), out.add(o), r) };
            o += r;
            i += r;
        }
        if i == b.len() {
            // SAFETY: the first `o` bytes are written.
            unsafe { buf.set_len(o) };
            return true;
        }
    }
}

/// cl100k over text with non-ASCII chars (the first at byte `first_na`), still on
/// the bit-parallel scanner: every char becomes one ASCII byte of the same class
/// under the grammar — `\p{L}` → `a`, `\p{N}` → `0`, other `\s` → `\t`, anything
/// else → `!` — the stand-in text is bulk-scanned, and its char offsets are mapped
/// back to bytes.
///
/// Exact because cl100k only ever inspects a non-ASCII char through those four
/// classes: `\r`, `\n`, space and `'` are ASCII, and the contraction letters
/// match ASCII only (see `scan::contraction_len`) — `a` is none of them, so no
/// contraction forms or breaks. Every quantity the grammar counts (the one-char
/// prefix, `\p{N}{1,3}`, the whitespace back-off of `\s+(?!\S)`) counts chars,
/// which the stand-in makes one byte each.
fn scan_cl100k_transcoded<const CAP: u32, F>(
    text: &str,
    first_na: usize,
    mut emit: F,
) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let t = super::unicode_class::tables();
    let b = text.as_bytes();
    let (mut buf, mut mb) = TRANSCODE.take();
    transcode(b, first_na, &mut buf, &mut mb, cl100k_reps(t));
    // SAFETY: every byte pushed is ASCII.
    let ascii = unsafe { std::str::from_utf8_unchecked(&buf) };
    let starts = cl100k_bulk_starts::<CAP>(ascii).expect("transcoded text is ASCII");
    let r = emit_spans_mapped(&starts, &mb, b.len(), &mut emit);
    TRANSCODE.set((buf, mb));
    r
}

// ── GPT-2 ByteLevel regex ─────────────────────────────────────────────────────

/// Bit-parallel GPT-2 pretokenizer (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
/// ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`) over ASCII (or transcoded) text, byte-exact
/// with `scan::scan_core_gpt2`; `None` on a non-ASCII byte. A token starts at:
/// - each letter / digit / punct run start, unless a space precedes it — the
///   space (always the last char of its whitespace run, so never claimed by the
///   whitespace token) then starts the token instead;
/// - each whitespace run start, and the run's last char when non-whitespace
///   follows (`\s+(?!\S)` gives it back; a space takes it as prefix, any other
///   whitespace char is a token of its own);
/// - a contraction (case-sensitive) at an apostrophe that starts a token — one
///   after a punct char or a space is punct — which also cuts the text after it.
fn gpt2_bulk_starts(bytes: &[u8]) -> Option<Bitmap> {
    let n = bytes.len();
    if n == 0 {
        return Some(Bitmap::zeros(0));
    }
    let nwords = n.div_ceil(64);
    let last = nwords - 1;
    let tail_bits = n - last * 64;
    let tail_mask = if tail_bits == 64 {
        !0u64
    } else {
        (1u64 << tail_bits) - 1
    };
    let mut starts = Bitmap::zeros(n);
    let mut apb = Bitmap::zeros(n);
    let mut any_ap = 0u64;
    let classify = |w: usize| unsafe_bulk_classify::<false, false>(bytes, w * 64);
    let mut cur = classify(0);
    if cur.bad {
        return None;
    }
    let (mut h_l, mut h_d, mut h_p, mut h_w, mut h_s) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for w in 0..nwords {
        let nx = if w < last {
            let c = classify(w + 1);
            if c.bad {
                return None;
            }
            c
        } else {
            BulkClasses::default()
        };
        let valid = if w == last { tail_mask } else { !0 };
        let valid_n = if w + 1 == last {
            tail_mask
        } else if w < last {
            !0
        } else {
            0
        };
        let (lw, dw, ww, sw) = (cur.alpha, cur.digit, cur.ws, cur.sp);
        let pw = valid & !(lw | dw | ww);
        let wn = nx.ws;
        let nonws_n = valid_n & !wn;
        let prev = |x: u64, h: u64| (x << 1) | h;
        let next = |x: u64, nxw: u64| (x >> 1) | (nxw << 63);
        let (lp, dp, pp, wp, sp_) = (
            prev(lw, h_l),
            prev(dw, h_d),
            prev(pw, h_p),
            prev(ww, h_w),
            prev(sw, h_s),
        );
        let nonws_next = next(valid & !ww, nonws_n);
        let mut s = ((lw & !lp) | (dw & !dp) | (pw & !pp)) & !sp_;
        s |= sw & nonws_next;
        s |= ww & !wp;
        s |= ww & nonws_next;
        if w == 0 {
            s |= 1;
        }
        starts.w[w] = s & valid;
        apb.w[w] = cur.ap;
        any_ap |= cur.ap;
        h_l = lw >> 63;
        h_d = dw >> 63;
        h_p = pw >> 63;
        h_w = ww >> 63;
        h_s = sw >> 63;
        cur = nx;
    }
    if any_ap != 0 {
        for w in 0..nwords {
            let mut bits = apb.w[w] & starts.w[w];
            while bits != 0 {
                let p = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let k = super::scan::gpt2_contraction_len(&bytes[p..]);
                if k == 0 {
                    continue;
                }
                for q in (p + 1)..(p + k) {
                    starts.w[q >> 6] &= !(1u64 << (q & 63));
                }
                if p + k < n {
                    starts.w[(p + k) >> 6] |= 1u64 << ((p + k) & 63);
                }
            }
        }
    }
    Some(starts)
}

// ── DeepSeek: bit-parallel scan of its three sequenced splits ────────────────

/// Transcoded-text stand-ins for DeepSeek's non-ASCII classes. They sit above the
/// ASCII range, so they never collide with real text bytes (the buffer holding
/// them is only ever read by [`try_scan_deepseek_bulk`], never as `str`).
const DS_VL: u8 = 0x80; // `[\p{L}\p{M}]` outside the CJK range
const DS_VP: u8 = 0x81; // `[\p{P}\p{S}]` outside the CJK range
const DS_CL: u8 = 0x82; // CJK-range `[\p{L}\p{M}]`
const DS_CP: u8 = 0x83; // CJK-range `[\p{P}\p{S}]`
const DS_CG: u8 = 0x84; // CJK-range anything else (unassigned)

/// One 64-byte block's DeepSeek class words. `ctrl` = ASCII controls that are not
/// whitespace (DeepSeek's "gap" class in ASCII); `hi` = bytes `>= 0x80`, which
/// only [`DS_VL`]..=[`DS_CG`] may be, and are decoded into the last five words
/// when `VIRT` (transcoded input).
#[derive(Default, Debug, PartialEq, Eq)]
struct DsClasses {
    alpha: u64,
    digit: u64,
    ws: u64,
    nl: u64,
    sp: u64,
    ctrl: u64,
    hi: u64,
    vl: u64,
    vp: u64,
    cl: u64,
    cp: u64,
    cg: u64,
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn classify_block_ds<const VIRT: bool>(bytes: &[u8], off: usize) -> DsClasses {
    use std::arch::aarch64::*;
    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn movemask(v: uint8x16_t) -> u64 {
        const POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bits = vandq_u8(v, vld1q_u8(POWERS.as_ptr()));
        (vaddv_u8(vget_low_u8(bits)) as u64) | ((vaddv_u8(vget_high_u8(bits)) as u64) << 8)
    }
    let mut c = DsClasses::default();
    let end = (off + 64).min(bytes.len());
    let mut p = off;
    let mut j = 0usize;
    while p < end {
        let take = (end - p).min(16);
        let v = if take == 16 {
            vld1q_u8(bytes.as_ptr().add(p))
        } else {
            let mut buf = [0u8; 16];
            buf[..take].copy_from_slice(&bytes[p..p + take]);
            vld1q_u8(buf.as_ptr())
        };
        let lower = vorrq_u8(v, vdupq_n_u8(0x20));
        let alpha = vandq_u8(
            vcgeq_u8(lower, vdupq_n_u8(b'a')),
            vcleq_u8(lower, vdupq_n_u8(b'z')),
        );
        let digit = vandq_u8(vcgeq_u8(v, vdupq_n_u8(b'0')), vcleq_u8(v, vdupq_n_u8(b'9')));
        let sp = vceqq_u8(v, vdupq_n_u8(0x20));
        let ws = vorrq_u8(
            vandq_u8(vcgeq_u8(v, vdupq_n_u8(0x09)), vcleq_u8(v, vdupq_n_u8(0x0D))),
            sp,
        );
        let nl = vorrq_u8(vceqq_u8(v, vdupq_n_u8(0x0A)), vceqq_u8(v, vdupq_n_u8(0x0D)));
        let ctrl = vorrq_u8(
            vbicq_u8(vcltq_u8(v, vdupq_n_u8(0x20)), ws),
            vceqq_u8(v, vdupq_n_u8(0x7F)),
        );
        let hi = vcgeq_u8(v, vdupq_n_u8(0x80));
        let valid: u64 = if take == 16 {
            0xFFFF
        } else {
            (1u64 << take) - 1
        };
        c.alpha |= (movemask(alpha) & valid) << j;
        c.digit |= (movemask(digit) & valid) << j;
        c.ws |= (movemask(ws) & valid) << j;
        c.nl |= (movemask(nl) & valid) << j;
        c.sp |= (movemask(sp) & valid) << j;
        if vmaxvq_u8(ctrl) != 0 {
            c.ctrl |= (movemask(ctrl) & valid) << j;
        }
        if vmaxvq_u8(hi) != 0 {
            c.hi |= (movemask(hi) & valid) << j;
            if VIRT {
                c.vl |= (movemask(vceqq_u8(v, vdupq_n_u8(DS_VL))) & valid) << j;
                c.vp |= (movemask(vceqq_u8(v, vdupq_n_u8(DS_VP))) & valid) << j;
                c.cl |= (movemask(vceqq_u8(v, vdupq_n_u8(DS_CL))) & valid) << j;
                c.cp |= (movemask(vceqq_u8(v, vdupq_n_u8(DS_CP))) & valid) << j;
                c.cg |= (movemask(vceqq_u8(v, vdupq_n_u8(DS_CG))) & valid) << j;
            }
        }
        j += take;
        p += take;
    }
    c
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn classify_block_ds<const VIRT: bool>(bytes: &[u8], off: usize) -> DsClasses {
    #[cfg(target_arch = "x86_64")]
    if avx2::available() {
        // SAFETY: AVX2 support was just detected.
        return unsafe { avx2::classify_block_ds::<VIRT>(bytes, off) };
    }
    classify_block_ds_scalar::<VIRT>(bytes, off)
}

/// Portable per-byte DeepSeek classify (the reference the SIMD kernels match).
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn classify_block_ds_scalar<const VIRT: bool>(bytes: &[u8], off: usize) -> DsClasses {
    let mut c = DsClasses::default();
    let end = (off + 64).min(bytes.len());
    for (bit, &b) in bytes[off..end].iter().enumerate() {
        let m = 1u64 << bit;
        let ws = matches!(b, 0x09..=0x0D | 0x20);
        if (b | 0x20).wrapping_sub(b'a') < 26 && b < 0x80 {
            c.alpha |= m;
        }
        if b.is_ascii_digit() {
            c.digit |= m;
        }
        if ws {
            c.ws |= m;
        }
        if b == b'\n' || b == b'\r' {
            c.nl |= m;
        }
        if b == b' ' {
            c.sp |= m;
        }
        if (b < 0x20 && !ws) || b == 0x7F {
            c.ctrl |= m;
        }
        if b >= 0x80 {
            c.hi |= m;
            if VIRT {
                match b {
                    DS_VL => c.vl |= m,
                    DS_VP => c.vp |= m,
                    DS_CL => c.cl |= m,
                    DS_CP => c.cp |= m,
                    DS_CG => c.cg |= m,
                    _ => {}
                }
            }
        }
    }
    c
}

/// Mark every third digit of each digit run in `d` as a start (`\p{N}{1,3}` cuts
/// past the run start, which the caller sets). Word-parallel with a carry for runs
/// crossing a word edge (as in [`try_scan_cl100k_bulk_cap`]).
fn digit_groups_of_3(d: &Bitmap, n: usize, starts: &mut Bitmap) {
    const CAP: u32 = 3;
    let nwords = d.w.len();
    let last = nwords - 1;
    let tail_bits = n - last * 64;
    let mut dopen = false;
    let mut dsince = 0u32;
    let mut prev_dw = 0u64;
    for w in 0..nwords {
        let dw = d.w[w];
        let vld = if w == last && tail_bits < 64 {
            (1u64 << tail_bits) - 1
        } else {
            !0
        };
        if dw == 0 {
            dopen = false;
            dsince = 0;
            prev_dw = 0;
            continue;
        }
        let prev_is_digit = (prev_dw >> 63) & 1 != 0;
        let mut m = dw & vld & !((dw << 1) | u64::from(prev_is_digit));
        if dopen && prev_is_digit {
            let mut s = 1u64 & dw;
            for _ in 0..((CAP - dsince % CAP) % CAP) {
                s = (s << 1) & dw & vld;
            }
            m |= s;
        }
        let mut groups = m;
        let nk = dw & (dw << 1) & (dw << 2);
        let mut mm = m;
        while mm != 0 {
            mm = (mm << CAP) & nk;
            groups |= mm;
        }
        starts.w[w] |= groups;
        let len = if w == last { tail_bits } else { 64 };
        let next_is_digit = w + 1 < nwords && d.w[w + 1] & 1 != 0;
        let tn = if dw & (1u64 << (len - 1)) == 0 {
            0
        } else {
            let z = !dw & vld;
            if z == 0 {
                vld
            } else {
                vld & !((1u64 << (64 - z.leading_zeros())) - 1)
            }
        };
        dopen = tn != 0 && next_is_digit;
        dsince = if dopen {
            let g = groups & tn;
            let counted = if g == 0 {
                dsince + (dw & vld & tn).count_ones()
            } else {
                (dw & vld & tn & !((1u64 << (63 - g.leading_zeros())) - 1)).count_ones()
            };
            counted % CAP
        } else {
            0
        };
        prev_dw = dw;
    }
}

/// Bit-parallel DeepSeek pretokenizer — `Sequence([Split(\p{N}{1,3}),
/// Split([一-龥぀-ゟ゠-ヿ]+), Split(gpt-like)])`, all `Isolated` — byte-exact with
/// `scan::scan_core_deepseek`. Input is ASCII, or (`virt`) a transcoded stand-in
/// whose non-ASCII chars are one [`DS_VL`]..=[`DS_CG`] / `0` / `\t` / `\x01` byte
/// each (see [`scan_deepseek_transcoded`]). Returns `Ok(false)` on a byte `>= 0x80`
/// when not `virt`.
///
/// A token starts at (with `X'` = "the char before is X", `X^` = "the char after"):
/// - digits: each run start, every 3rd digit (`\p{N}{1,3}`), and the char after;
/// - every CJK-range boundary (split 2), and within the CJK side each run start of
///   letters (`CL`), punctuation (`CP`) and gaps (`CG`), where a `CG` directly
///   before a `CL` run is that run's prefix;
/// - letters `L` (ASCII `a` or `VL`): a run start, unless `L'` is a prefix —
///   whitespace but `\r\n`, or a gap char `G` — which then starts it instead, or
///   the run follows a `[<ascii punct>][A-Za-z]+` opener;
/// - that opener `X`: an ASCII punct with `a^` at a token start, i.e. neither punct
///   (`P`) nor a space before it (both absorb it); its `[A-Za-z]+` ends at the
///   first non-ASCII-letter, which starts a token even when it is a `VL`;
/// - punct `P` (ASCII or `VP`): a run start, or the space before it (` ?P+`);
///   `[\r\n]` right after a `P` run belong to it;
/// - gaps: a run start, and any `G` with `L^` (it prefixes that run);
/// - whitespace (minus those trailing newlines): a run start, the char after a
///   run's last newline, and a run's last char when the run is 2+ long and the
///   next char is non-whitespace in the same piece — `\s+(?!\S)` backs off one
///   char, but never before a digit or CJK char, which start new pieces.
fn try_scan_deepseek_bulk<F>(bytes: &[u8], virt: bool, emit: F) -> Result<bool, String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    match deepseek_bulk_starts(bytes, virt) {
        None => Ok(false),
        Some(starts) => {
            emit_spans(&starts, bytes.len(), emit)?;
            Ok(true)
        }
    }
}

/// The start bitmap of [`try_scan_deepseek_bulk`] (`None`: a byte `>= 0x80` and
/// not `virt`).
fn deepseek_bulk_starts(bytes: &[u8], virt: bool) -> Option<Bitmap> {
    let n = bytes.len();
    if n == 0 {
        return Some(Bitmap::zeros(0));
    }
    let nwords = n.div_ceil(64);
    let last = nwords - 1;
    let tail_bits = n - last * 64;
    let tail_mask = if tail_bits == 64 {
        !0u64
    } else {
        (1u64 << tail_bits) - 1
    };
    let classify = |w: usize| -> DsClasses {
        // SAFETY: NEON is baseline on aarch64 (scalar elsewhere).
        unsafe {
            if virt {
                classify_block_ds::<true>(bytes, w * 64)
            } else {
                classify_block_ds::<false>(bytes, w * 64)
            }
        }
    };
    let mut starts = Bitmap::zeros(n);
    let mut nlb = Bitmap::zeros(n);
    let mut wseb = Bitmap::zeros(n);
    let mut db = Bitmap::zeros(n);
    let (mut any_nl, mut any_d) = (0u64, 0u64);
    // One word of lookahead (`nx`); the previous word only through its top bits.
    let mut cur = classify(0);
    if !virt && cur.hi != 0 {
        return None;
    }
    // Top bit (as 0/1) of the previous word, per class.
    let (mut h_l, mut h_a, mut h_d, mut h_cjk, mut h_cl, mut h_cp, mut h_cg) =
        (0u64, 0, 0, 0, 0, 0, 0);
    let (mut h_p, mut h_sp, mut h_g, mut h_nl, mut h_pre, mut h_wse, mut h_x3a, mut h_run) =
        (0u64, 0, 0, 0, 0, 0, 0, 0);
    let (mut pt_carry, mut run_carry) = (false, false);
    for w in 0..nwords {
        let nx = if w < last {
            let c = classify(w + 1);
            if !virt && c.hi != 0 {
                return None;
            }
            c
        } else {
            DsClasses::default()
        };
        let valid = if w == last { tail_mask } else { !0 };
        let valid_n = if w + 1 == last {
            tail_mask
        } else if w < last {
            !0
        } else {
            0
        };
        let apw = valid & !(cur.alpha | cur.digit | cur.ws | cur.ctrl | cur.hi);
        let apn = valid_n & !(nx.alpha | nx.digit | nx.ws | nx.ctrl | nx.hi);
        let (aw, dw, wsw, nlw, spw, gw) = (cur.alpha, cur.digit, cur.ws, cur.nl, cur.sp, cur.ctrl);
        let (vlw, clw, cpw, cgw) = (cur.vl, cur.cl, cur.cp, cur.cg);
        let pw = apw | cur.vp;
        let pnx = apn | nx.vp;
        let lw = aw | vlw;
        let lnx = nx.alpha | nx.vl;
        let cjkw = clw | cpw | cgw;
        let cjknx = nx.cl | nx.cp | nx.cg;
        let prev = |x: u64, h: u64| (x << 1) | h;
        let next = |x: u64, nxw: u64| (x >> 1) | (nxw << 63);

        let (lp, dp, cjkp) = (prev(lw, h_l), prev(dw, h_d), prev(cjkw, h_cjk));
        let (clp, cpp, cgp) = (prev(clw, h_cl), prev(cpw, h_cp), prev(cgw, h_cg));
        let (pp, spp, gp, nlp) = (
            prev(pw, h_p),
            prev(spw, h_sp),
            prev(gw, h_g),
            prev(nlw, h_nl),
        );
        let (an, pn, cln, ln) = (
            next(aw, nx.alpha),
            next(pw, pnx),
            next(clw, nx.cl),
            next(lw, lnx),
        );
        let wsn = next(wsw, nx.ws);
        let pre = (wsw & !nlw) | gw;
        let prep = prev(pre, h_pre);
        let x3a = apw & an & !pp & !spp;
        let x3ap = prev(x3a, h_x3a);
        // `[\r\n]*` trailing a (non-CJK) punct run belongs to it.
        let (pt, co) = fill_forward(nlw & !nlp & pp, nlw, pt_carry);
        pt_carry = co;
        let wsew = wsw & !pt;
        let wsep = prev(wsew, h_wse);
        let piece_n = next(valid & !dw & !cjkw, valid_n & !nx.digit & !cjknx);

        let mut s = (dw ^ dp) & valid; // digit run starts, and the char after a run
        s |= (cjkw ^ cjkp) & valid;
        s |= clw & !clp & !cgp;
        s |= cpw & !cpp;
        s |= cgw & (!cgp | cln);
        s |= lw & !lp & !prep & !(aw & x3ap);
        s |= pre & ln;
        s |= pw & !pp & !spp;
        s |= spw & pn;
        s |= gw & !gp;
        s |= wsew & !wsep;
        s |= wsew & wsep & !wsn & !nlw & !nlp & piece_n;
        // The opener's `[A-Za-z]+` ends at the first non-ASCII-letter: a `VL` there
        // starts a token (a plain letter run would have absorbed it).
        let (run, rc) = fill_forward(x3ap & aw, aw, run_carry);
        run_carry = rc;
        s |= prev(run, h_run) & !aw & vlw;
        if w == 0 {
            s |= 1;
        }
        starts.w[w] = s & valid;
        nlb.w[w] = nlw;
        wseb.w[w] = wsew;
        db.w[w] = dw;
        any_nl |= nlw;
        any_d |= dw;

        h_l = lw >> 63;
        h_a = aw >> 63;
        h_d = dw >> 63;
        h_cjk = cjkw >> 63;
        h_cl = clw >> 63;
        h_cp = cpw >> 63;
        h_cg = cgw >> 63;
        h_p = pw >> 63;
        h_sp = spw >> 63;
        h_g = gw >> 63;
        h_nl = nlw >> 63;
        h_pre = pre >> 63;
        h_wse = wsew >> 63;
        h_x3a = x3a >> 63;
        h_run = run >> 63;
        cur = nx;
    }
    let _ = h_a;
    if any_nl != 0 {
        // `\s*[\r\n]+`: a token opens after a whitespace run's last newline (see
        // [`newline_split`]; the run is `wse`, which already excludes punct trailing).
        let mut carry = false;
        for w in (0..nwords).rev() {
            let run = wseb.w[w];
            let (rev_fill, co) =
                fill_forward((nlb.w[w] & run).reverse_bits(), run.reverse_bits(), carry);
            carry = co;
            let later_nl = rev_fill.reverse_bits();
            let after_nl = (nlb.w[w] << 1) | if w > 0 { nlb.w[w - 1] >> 63 } else { 0 };
            starts.w[w] |= after_nl & run & !later_nl;
        }
    }
    if any_d != 0 {
        digit_groups_of_3(&db, n, &mut starts);
    }
    Some(starts)
}

/// DeepSeek over text with non-ASCII chars (the first at `first_na`): each char
/// becomes one byte of its class — `\p{N}` → `0`, other whitespace → `\t`,
/// letters/marks → [`DS_VL`] / [`DS_CL`], punctuation/symbols → [`DS_VP`] /
/// [`DS_CP`], the rest → `\x01` / [`DS_CG`] (CJK-range variants for
/// `scan::is_ds_cjk` chars) — then [`try_scan_deepseek_bulk`] runs on the stand-in
/// and char offsets are mapped back to bytes. Exact: the grammar sees non-ASCII
/// chars only through those classes (its `[<ascii punct>][A-Za-z]+` arm is ASCII
/// by definition, which is why letters and punctuation need their own codes).
fn scan_deepseek_transcoded<F>(text: &str, first_na: usize, mut emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let t = super::unicode_class::tables();
    let b = text.as_bytes();
    let (mut buf, mut mb) = TRANSCODE.take();
    transcode(b, first_na, &mut buf, &mut mb, deepseek_reps(t));
    let starts = deepseek_bulk_starts(&buf, true).expect("virtual input is accepted");
    let r = emit_spans_mapped(&starts, &mb, b.len(), &mut emit);
    TRANSCODE.set((buf, mb));
    r
}

/// Benchmark-only: seconds spent in (transcode, start bitmap, mapped emit) of the
/// transcoded cl100k path over `texts` (pure-ASCII texts are skipped).
#[doc(hidden)]
pub fn __bench_transcode_phases(texts: &[String]) -> (f64, f64, f64, usize) {
    let t = super::unicode_class::tables();
    let (mut tt, mut ts, mut te, mut spans) = (0.0, 0.0, 0.0, 0usize);
    let (mut buf, mut runs) = (Vec::new(), Vec::new());
    for text in texts {
        let b = text.as_bytes();
        let Some(first_na) = first_non_ascii(b) else {
            continue;
        };
        let t0 = std::time::Instant::now();
        transcode(b, first_na, &mut buf, &mut runs, |cp| {
            cl100k_rep_fast(t, cp)
        });
        let t1 = std::time::Instant::now();
        // SAFETY: every byte pushed is ASCII.
        let ascii = unsafe { std::str::from_utf8_unchecked(&buf) };
        let starts = cl100k_bulk_starts::<3>(ascii).unwrap();
        let t2 = std::time::Instant::now();
        let _ = emit_spans_mapped(&starts, &runs, b.len(), |_, _| {
            spans += 1;
            Ok(())
        });
        let t3 = std::time::Instant::now();
        tt += (t1 - t0).as_secs_f64();
        ts += (t2 - t1).as_secs_f64();
        te += (t3 - t2).as_secs_f64();
    }
    (tt, ts, te, spans)
}

/// Per-phase profiling for the bulk cl100k scanner. Off by default; the bench
/// enables it, runs a batch, then reads the accumulated per-phase seconds.
#[doc(hidden)]
pub mod prof {
    use std::cell::Cell;
    use std::sync::atomic::AtomicBool;
    pub static ENABLED: AtomicBool = AtomicBool::new(false);
    pub const PHASES: [&str; 7] = [
        "classify",
        "other",
        "punct_trail",
        "compute",
        "nl_split",
        "digit_cap",
        "emit",
    ];
    thread_local! {
        pub static SUMS: Cell<[f64; 7]> = const { Cell::new([0.0; 7]) };
        pub static CALLS: Cell<u64> = const { Cell::new(0) };
    }
}

/// Bulk cl100k pretoken scan over pure-ASCII `text` (contractions included), driven
/// by a per-block classify + whole-text start-bitmap. Emits one span per token
/// (splitting digit runs into `{1,3}` at emit, and cutting contractions in a scalar
/// post-pass over the sparse apostrophe start-bits). Returns `Err` if any non-ASCII
/// byte is present (caller falls back to the scalar path — milestone 3's fused
/// interface). Verified byte-exact against `scan_cl100k_ascii`
/// (`cl100k_bulk_matches_oracle`).
pub(crate) fn try_scan_cl100k_bulk<F>(text: &str, emit: F) -> Result<bool, String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    try_scan_cl100k_bulk_cap::<3, F>(text, emit)
}

/// [`try_scan_cl100k_bulk`] with the `\p{N}{1,CAP}` digit cap as a parameter:
/// `3` is cl100k, `1` is Qwen's single-digit `\p{N}` (same grammar otherwise).
pub(crate) fn try_scan_cl100k_bulk_cap<const CAP: u32, F>(
    text: &str,
    emit: F,
) -> Result<bool, String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    match cl100k_bulk_starts::<CAP>(text) {
        None => Ok(false),
        Some(starts) => {
            emit_spans(&starts, text.len(), emit)?;
            Ok(true)
        }
    }
}

/// The start bitmap of [`try_scan_cl100k_bulk_cap`] (`None`: not pure ASCII):
/// the fused one-pass scan ([`family_starts_fused`]), or this grammar's
/// multi-pass scan for the one shape the fused scan hands back.
fn cl100k_bulk_starts<const CAP: u32>(text: &str) -> Option<Bitmap> {
    const { assert!(CAP == 1 || CAP == 3) };
    match family_starts_fused::<true, false, false, CAP, false, false>(
        text.as_bytes(),
        &SideBits::default(),
    ) {
        Fused::Done(starts) => Some(starts),
        Fused::Bad => None,
        Fused::LongRun => cl100k_bulk_starts_multipass::<CAP>(text),
    }
}

/// [`cl100k_bulk_starts`] as separate whole-text passes (classes, punct
/// trailing runs, word rules, newline split, digit cap, contractions): the fused
/// scan's fallback and its differential-test oracle.
fn cl100k_bulk_starts_multipass<const CAP: u32>(text: &str) -> Option<Bitmap> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    if n == 0 {
        return Some(Bitmap::zeros(0));
    }
    let prof = prof::ENABLED.load(std::sync::atomic::Ordering::Relaxed);
    let tnow = || std::time::Instant::now();
    let t_start = prof.then(tnow);
    // Materialize class bitmaps via the cheaper bulk classify (5 movemasks); the
    // pure-ASCII bulk path derives `other` at the bitmap level below. `ap`
    // (apostrophes) is kept only for the contraction post-pass.
    let (mut a, mut d, mut o, mut ws, mut nl, mut sp, mut ap) = (
        Bitmap::with_capacity(n),
        Bitmap::with_capacity(n),
        Bitmap::with_capacity(n),
        Bitmap::with_capacity(n),
        Bitmap::with_capacity(n),
        Bitmap::with_capacity(n),
        Bitmap::with_capacity(n),
    );
    let mut off = 0;
    let mut any_ap = 0u64; // OR of all apostrophe words — skip the post-pass if none
    while off < n {
        // SAFETY: NEON is baseline on aarch64; on other targets this is the SWAR path.
        let c = unsafe_bulk_classify::<false, false>(bytes, off);
        if c.bad {
            return None;
        }
        a.push(c.alpha);
        d.push(c.digit);
        ws.push(c.ws);
        nl.push(c.nl);
        sp.push(c.sp);
        ap.push(c.ap);
        any_ap |= c.ap;
        // `other = !(alpha|digit|ws)` (pure ASCII, so exactly `[^\s\p{L}\p{N}]`);
        // apostrophes fall in here naturally, matching the scalar rule. Derived here
        // to avoid a second whole-bitmap pass; the last word's bits past `n` are
        // masked to 0 below.
        o.push(!(c.alpha | c.digit | c.ws));
        off += 64;
    }

    let t_classify = prof.then(tnow);
    let nwords = a.w.len();
    // Bits past `n` in the last `other` word must stay 0 (they read as set from the
    // negation above).
    let last = nwords - 1;
    let tail_bits = n - last * 64; // 1..=64 valid bits in the last word
    if tail_bits < 64 {
        o.w[last] &= (1u64 << tail_bits) - 1;
    }
    let t_other = prof.then(tnow);
    // Run-reduction #2 — punct trailing `[\r\n]*`: newlines in a newline-run whose
    // first char is preceded by `O` belong to that punct token. Iterate only the
    // (sparse) newline bits; fill each qualifying run.
    let punct_trailing = punct_trailing_runs(&nl, &o, &nl);

    let t_pt = prof.then(tnow);
    // Single fused word-loop: word / number / punct starts + whitespace run-start
    // and tail-split, straight into the start bitmap. `ws_content` is kept so the
    // newline-split can be added next (as a sparse write, not a whole pass).
    let mut starts = Bitmap::zeros(n);
    let mut ws_content = Bitmap::zeros(n);
    let mut prev_wp = 0u64; // word_prefix of word w-1 (for the LRS prefix shift)
    let mut prev_wsc = 0u64; // ws_content of word w-1
    // Rolling window: each class vec is read once per word (the lookahead word),
    // the current word is carried forward, and only the previous word's top bit is
    // kept — no per-word re-indexing / bounds checks / edge branches.
    let (av, dv, ov, wsv, nlv, spv, ptvv) =
        (&a.w, &d.w, &o.w, &ws.w, &nl.w, &sp.w, &punct_trailing.w);
    let (mut ca, mut cd, mut co, mut cws, mut cnl, mut csp, mut cpt) =
        (av[0], dv[0], ov[0], wsv[0], nlv[0], spv[0], ptvv[0]);
    let (mut ha, mut ho, mut hsp, mut hd, mut hnl) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for w in 0..nwords {
        let nx = w + 1;
        let na = av.get(nx).copied().unwrap_or(0);
        let nd = dv.get(nx).copied().unwrap_or(0);
        let no = ov.get(nx).copied().unwrap_or(0);
        let nws = wsv.get(nx).copied().unwrap_or(0);
        let nnl = nlv.get(nx).copied().unwrap_or(0);
        let nsp = spv.get(nx).copied().unwrap_or(0);
        let npt = punct_trailing.w.get(nx).copied().unwrap_or(0);

        let aw = ca;
        let dw = cd;
        let ow = co;
        let wsw = cws;
        let nlw = cnl;
        let spw = csp;
        let ptw = cpt;
        let ap_ = (aw << 1) | ha;
        let an = (aw >> 1) | (na << 63);
        let op = (ow << 1) | ho;
        let on = (ow >> 1) | (no << 63);
        let spp = (spw << 1) | hsp;
        let dp = (dw << 1) | hd;
        let nlp = (nlw << 1) | hnl;

        let wp = (ow & an & !op & !spp) | ((wsw & !nlw) & an); // word_prefix
        let wp_prev = (wp << 1) | (prev_wp >> 63);
        let lrs = aw & !ap_;
        let psp = spw & on; // punct space-prefix
        let nonws = wp | (lrs & !wp_prev) | (dw & !dp) | (ow & !op & !spp & !an) | psp;

        let wcw = (wsw & !ptw) & !wp & !psp; // ws_content
        let wsc_prev = (wcw << 1) | (prev_wsc >> 63);
        let wsc_run_start = wcw & !wsc_prev;
        let ws_eff_next = ((wsw & !ptw) >> 1) | ((nws & !npt) << 63);
        let non_nl = !nlw & !nlp;
        let mut ws_tail = wcw & wsc_prev & !ws_eff_next & non_nl;
        if w == nwords - 1 {
            ws_tail &= !(1u64 << ((n - 1) & 63)); // no real non-ws follows the last byte
        }
        starts.w[w] = nonws | wsc_run_start | ws_tail;
        ws_content.w[w] = wcw;
        prev_wp = wp;
        prev_wsc = wcw;
        // advance the window
        ha = aw >> 63;
        ho = ow >> 63;
        hsp = spw >> 63;
        hd = dw >> 63;
        hnl = nlw >> 63;
        ca = na;
        cd = nd;
        co = no;
        cws = nws;
        cnl = nnl;
        csp = nsp;
        cpt = npt;
    }

    let t_compute = prof.then(tnow);
    // Run-reduction #1 — newline split at (last-nl-of-run)+1, written straight into
    // `starts`. Iterates only the sparse newline bits; a newline is its
    // effective-ws run's last iff no newline follows before the run ends. Keeping
    // it in the bitmap (vs re-scanning per token at emit) keeps emit branch-light.
    newline_split(&nl, &ws, &punct_trailing, &ws_content, &mut starts);

    let t_nl = prof.then(tnow);
    // Digit `\p{N}{1,3}` cap, computed word-parallel (bitcannon's fast ASCII path).
    // Instead of walking each digit run scalar-wise, mark every 3rd digit of every
    // run in a whole 64-bit word at once: `nk` holds each position whose two
    // predecessors are also digits (`n & n<<1 & n<<2`), so group boundaries fall out
    // of `m = (m << 3) & nk` iterated only a handful of times per word. Runs that
    // cross a word edge carry `open`/`since` (digits consumed toward the current
    // group at the edge) exactly as bitcannon's `Digits` does. Every bit produced is
    // a legitimate token start, so OR-ing into `starts` is safe (the run-start bits
    // are already set by the compute loop; this adds the interior `{1,3}` cuts). Kept
    // as its own streaming pass — folding it into the compute loop measurably slowed
    // that loop's otherwise branch-free pipeline.
    const { assert!(CAP == 1 || CAP == 3) };
    let mut dopen = false; // a digit run was still open at the previous word's top edge
    let mut dsince = 0u32; // digits of the current group consumed by that open run
    let mut prev_dw = 0u64;
    for w in 0..nwords {
        if CAP == 1 {
            // `\p{N}`: every digit is its own token.
            starts.w[w] |= d.w[w];
            continue;
        }
        let dw = d.w[w];
        let vld = if w == last {
            if tail_bits == 64 {
                !0
            } else {
                (1u64 << tail_bits) - 1
            }
        } else {
            !0
        };
        if dw == 0 {
            dopen = false;
            dsince = 0;
            prev_dw = 0;
            continue;
        }
        let prev_is_digit = (prev_dw >> 63) & 1 != 0;
        // fresh run starts: a digit with no digit immediately before it
        let mut m = dw & vld & !((dw << 1) | u64::from(prev_is_digit));
        // resume a run crossing the edge: place its first in-word group boundary
        // `(CAP - since % CAP) % CAP` digits in from bit 0.
        if dopen && prev_is_digit {
            let mut s = 1u64 & dw; // bit 0 if it is a digit (the run continues)
            for _ in 0..((CAP - dsince % CAP) % CAP) {
                s = (s << 1) & dw & vld;
            }
            m |= s;
        }
        let mut groups = m;
        let nk = dw & (dw << 1) & (dw << 2); // p, p-1, p-2 all digits (cap = 3)
        let mut mm = m;
        while mm != 0 {
            mm = (mm << CAP) & nk;
            groups |= mm;
        }
        starts.w[w] |= groups;
        // carry the run into the next word
        let len = if w == last { tail_bits } else { 64 };
        let next_is_digit = w + 1 < nwords && d.w[w + 1] & 1 != 0;
        // trailing digit run of this word (bitcannon's `trail_run`)
        let tn = if dw & (1u64 << (len - 1)) == 0 {
            0
        } else {
            let z = !dw & vld;
            if z == 0 {
                vld
            } else {
                vld & !((1u64 << (64 - z.leading_zeros())) - 1)
            }
        };
        dopen = tn != 0 && next_is_digit;
        dsince = if dopen {
            // bitcannon's `digits_since`: digits into the current group at the edge
            let g = groups & tn;
            let counted = if g == 0 {
                dsince + (dw & vld & tn).count_ones()
            } else {
                (dw & vld & tn & !((1u64 << (63 - g.leading_zeros())) - 1)).count_ones()
            };
            counted % CAP
        } else {
            0
        };
        prev_dw = dw;
    }

    // Contraction alternative `(?i:'s|'t|'re|'ve|'m|'ll|'d)` — highest priority, so a
    // contraction cuts the letter run behind it (`'store` → `'s`, `tore`). The
    // apostrophe already opens a token (word-prefix rule), so two things remain: the
    // literal is one token (clear any interior start), and the char after it opens the
    // next token. Apostrophes are ~0.1–0.7 % of bytes, so read the literal scalar-side
    // at the sparse apostrophe start-bits rather than as a per-suffix-letter stream.
    // No chaining loop is needed: the char after a contraction, if itself an
    // apostrophe, has a letter predecessor and so is already a natural start-bit.
    if any_ap != 0 {
        for w in 0..nwords {
            let mut bits = ap.w[w] & starts.w[w];
            while bits != 0 {
                let p = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let k = contraction_len(bytes, p);
                if k == 0 {
                    continue;
                }
                // the literal `(p, p + k)` is interior — it opens nothing (letters
                // never carry an interior start, but clear defensively for parity)
                for q in (p + 1)..(p + k) {
                    starts.w[q >> 6] &= !(1u64 << (q & 63));
                }
                // ...and the char after the literal opens the next token
                if p + k < n {
                    starts.w[(p + k) >> 6] |= 1u64 << ((p + k) & 63);
                }
            }
        }
    }

    let t_digit = prof.then(tnow);
    if prof {
        let t_emit = tnow();
        let secs = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64();
        let ts = [
            secs(t_start.unwrap(), t_classify.unwrap()),
            secs(t_classify.unwrap(), t_other.unwrap()),
            secs(t_other.unwrap(), t_pt.unwrap()),
            secs(t_pt.unwrap(), t_compute.unwrap()),
            secs(t_compute.unwrap(), t_nl.unwrap()),
            secs(t_nl.unwrap(), t_digit.unwrap()),
            secs(t_digit.unwrap(), t_emit),
        ];
        prof::SUMS.with(|s| {
            let mut acc = s.get();
            for i in 0..7 {
                acc[i] += ts[i];
            }
            s.set(acc);
        });
        prof::CALLS.with(|c| c.set(c.get() + 1));
    }
    Some(starts)
}

/// Emit one span per consecutive pair of start bits — branch-free. No start bit is
/// ever set at a position `>= n`, so the walk needs no per-span range check.
#[inline(always)]
fn emit_spans<F>(starts: &Bitmap, n: usize, mut emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let mut prev_start = usize::MAX;
    for wk in 0..starts.w.len() {
        let base = wk * 64;
        let mut bits = starts.w[wk];
        while bits != 0 {
            let s = base + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            if prev_start != usize::MAX {
                emit(prev_start, s)?;
            }
            prev_start = s;
        }
    }
    if prev_start != usize::MAX {
        emit(prev_start, n)?;
    }
    Ok(())
}

/// [`emit_spans`] for a start bitmap over transcoded text (one byte per char):
/// char offsets become byte offsets on the fly through a [`ByteMap`] cursor over
/// the multibyte runs.
#[inline(always)]
fn emit_spans_mapped<F>(
    starts: &Bitmap,
    runs: &[MbRun],
    n_bytes: usize,
    mut emit: F,
) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let mut map = ByteMap::new(runs);
    let mut prev = usize::MAX;
    for wk in 0..starts.w.len() {
        let base = wk * 64;
        let mut bits = starts.w[wk];
        while bits != 0 {
            let c = base + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let b = map.byte(c);
            if prev != usize::MAX {
                emit(prev, b)?;
            }
            prev = b;
        }
    }
    if prev != usize::MAX {
        emit(prev, n_bytes)?;
    }
    Ok(())
}

/// [`try_scan_cl100k_bulk`], with "not pure ASCII" reported as an `Err` (nothing emitted).
pub(crate) fn scan_cl100k_bulk<F>(text: &str, emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    if try_scan_cl100k_bulk(text, emit)? {
        Ok(())
    } else {
        Err("non-ascii".into())
    }
}

/// Bulk o200k pretoken scan over pure-ASCII `text`. Same bit-parallel engine as
/// [`scan_cl100k_bulk`], with o200k's three grammar differences:
///  1. **word case-split** — a letter run splits at every lower→upper transition
///     (`HelloWorld` → `Hello`, `World`; `HTTPRequest` stays whole), which for ASCII
///     is exactly the `[A-Z]*[a-z]+` / `[A-Z]+[a-z]*` alternation;
///  2. **contraction is a suffix** glued onto the word (`don't` → one token), the
///     inverse of cl100k — a scalar post-pass *merges* it rather than splitting;
///  3. **punct trailing is `[\r\n/]*`** (cl100k's is `[\r\n]*`) — a punct run's tail
///     absorbs `/` as well as newlines.
///
/// Returns `Err` on any non-ASCII byte (caller falls back to the scalar path).
/// Verified byte-exact against `scan::scan_core(O200k, …)` (`o200k_bulk_matches_scalar`).
pub(crate) fn try_scan_o200k_bulk<F>(text: &str, emit: F) -> Result<bool, String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    match o200k_bulk_starts::<true, true, 3, false>(text.as_bytes()) {
        None => Ok(false),
        Some(starts) => {
            emit_spans(&starts, text.len(), emit)?;
            Ok(true)
        }
    }
}

/// The start bitmap of [`try_scan_o200k_bulk`] (`None`: not pure ASCII), for the
/// o200k family: `SLASH` = the punct tail is `[\r\n/]*` (o200k, tekken; Kimi's is
/// `[\r\n]*`), `CONTR` = letter tokens take the contraction suffix (not tekken),
/// `CAP` = the `\p{N}{1,CAP}` digit cap (tekken: 1). Kimi differs from o200k in
/// ASCII only by its tail (its `[\p{Han}]+` arm and Han-less classes are non-ASCII).
///
/// One forward pass: each 64-byte block is classified one block ahead and its
/// start bits derived in place — the boundary algebra of
/// [`o200k_bulk_starts_multipass`], whose passes each need at most the next
/// word, fused into one loop with only the start bitmap materialized. The one
/// rule that can look further, the newline split's "a newline later in this
/// whitespace run", reads the next word; a run spanning a whole word without a
/// newline (64 blanks — rare) takes the multi-pass scan instead.
fn o200k_bulk_starts<const SLASH: bool, const CONTR: bool, const CAP: u32, const HAN: bool>(
    bytes: &[u8],
) -> Option<Bitmap> {
    const { assert!(CAP == 1 || CAP == 3) };
    match family_starts_fused::<false, SLASH, CONTR, CAP, HAN, false>(bytes, &SideBits::default()) {
        Fused::Done(starts) => Some(starts),
        Fused::Bad => None,
        Fused::LongRun => o200k_bulk_starts_multipass::<SLASH, CONTR, CAP, HAN>(bytes),
    }
}

/// [`o200k_bulk_starts`] over a transcoded text whose `a` stand-ins include the
/// letters (and marks) of both case classes flagged in `side`.
fn o200k_bulk_starts_side<const SLASH: bool, const CONTR: bool, const CAP: u32, const HAN: bool>(
    bytes: &[u8],
    side: &SideBits,
) -> Option<Bitmap> {
    const { assert!(CAP == 1 || CAP == 3) };
    match family_starts_fused::<false, SLASH, CONTR, CAP, HAN, true>(bytes, side) {
        Fused::Done(starts) => Some(starts),
        Fused::Bad => None,
        // The multi-pass scan has no both-class letters.
        Fused::LongRun if side.0.iter().any(|&w| w != 0) => None,
        Fused::LongRun => o200k_bulk_starts_multipass::<SLASH, CONTR, CAP, HAN>(bytes),
    }
}

/// Outcome of [`family_starts_fused`].
enum Fused {
    Done(Bitmap),
    /// A char no class covers (the caller's fallback).
    Bad,
    /// A whitespace run spans a whole word with no newline: use the multi-pass scan.
    LongRun,
}

/// One word's classes for [`family_starts_fused`] (`o` masked to the text).
#[derive(Clone, Copy, Default)]
struct OWord {
    a: u64,
    d: u64,
    o: u64,
    ws: u64,
    nl: u64,
    sp: u64,
    ap: u64,
    up: u64,
    slash: u64,
    han: u64,
    /// With `BOTH`: letters in both case classes (`a` stand-ins), and marks.
    both: u64,
    mark: u64,
}

/// See [`o200k_bulk_starts`]; with `CL` the cl100k grammar instead (see
/// [`cl100k_bulk_starts`]): its letter runs are not case-split, and its
/// contraction `(?i:'s|'t|'re|'ve|'m|'ll|'d)` is a token of its own rather than a
/// suffix (`SLASH`, `CONTR` and `HAN` are then false).
#[inline(always)]
fn family_starts_fused<
    const CL: bool,
    const SLASH: bool,
    const CONTR: bool,
    const CAP: u32,
    const HAN: bool,
    const BOTH: bool,
>(
    bytes: &[u8],
    side: &SideBits,
) -> Fused {
    const { assert!(!CL || !(SLASH || CONTR || HAN || BOTH)) };
    let n = bytes.len();
    if n == 0 {
        return Fused::Done(Bitmap::zeros(0));
    }
    let nwords = n.div_ceil(64);
    let last = nwords - 1;
    let tail_bits = n - last * 64;
    let classify = |k: usize| -> Option<OWord> {
        // cl100k needs no case classes.
        let c = if CL {
            unsafe_bulk_classify::<false, false>(bytes, k * 64)
        } else {
            unsafe_bulk_classify::<true, HAN>(bytes, k * 64)
        };
        if c.bad {
            return None;
        }
        // A Han stand-in is none of letter / digit / space / punct (Kimi).
        let mut o = !(c.alpha | c.digit | c.ws | c.han);
        if k == last && tail_bits < 64 {
            o &= (1u64 << tail_bits) - 1;
        }
        Some(OWord {
            a: c.alpha,
            d: c.digit,
            o,
            ws: c.ws,
            nl: c.nl,
            sp: c.sp,
            ap: c.ap,
            up: c.up,
            slash: c.slash,
            han: if HAN { c.han } else { 0 },
            both: if BOTH {
                side.0.get(k).copied().unwrap_or(0)
            } else {
                0
            },
            mark: if BOTH {
                side.1.get(k).copied().unwrap_or(0)
            } else {
                0
            },
        })
    };
    // Punct trailing `[\r\n/]*` of a word (`fill_forward` of the newline-run starts
    // right after an `other` char), given the previous word and the carry.
    let trailing = |c: &OWord, prev_nl: u64, prev_o: u64, carry: bool| -> (u64, bool) {
        let start = c.nl & !((c.nl << 1) | (prev_nl >> 63)) & ((c.o << 1) | (prev_o >> 63));
        let class = if SLASH { c.nl | c.slash } else { c.nl };
        fill_forward(start, class, carry)
    };

    let mut starts = Bitmap::zeros(n);
    let Some(mut cur) = classify(0) else {
        return Fused::Bad;
    };
    let (mut pt, mut pt_carry) = trailing(&cur, 0, 0, false);
    // Carries from the previous word.
    let (mut prev, mut prev_wp, mut prev_wsc) = (OWord::default(), 0u64, 0u64);
    let mut prev_ow_open = 0u64;
    // Digit cap state (see `o200k_bulk_starts_multipass`).
    let (mut dopen, mut dsince) = (false, 0u32);
    // Contraction merges reaching into this word from the previous one.
    let (mut pend_clear, mut pend_set, mut after_contraction) = (0u64, 0u64, usize::MAX);
    // With `BOTH`: whether the letter before this word is in a lower run that holds
    // a lower-only letter (see the case split below).
    let mut lower_seen = false;
    for w in 0..nwords {
        // The next word, one block ahead, and its punct trailing.
        let (next, npt, npt_carry) = if w < last {
            let Some(nx) = classify(w + 1) else {
                return Fused::Bad;
            };
            let (t, c) = trailing(&nx, cur.nl, cur.o, pt_carry);
            (nx, t, c)
        } else {
            (OWord::default(), 0, false)
        };

        // Word rules — o200k's case split, trailing chars kept out of punct runs.
        let ow_open = cur.o & !pt;
        let no_open = next.o & !npt;
        let ap_ = (cur.a << 1) | (prev.a >> 63);
        let an = (cur.a >> 1) | (next.a << 63);
        let op = (ow_open << 1) | (prev_ow_open >> 63);
        let on = (ow_open >> 1) | ((no_open & 1) << 63);
        let spp = (cur.sp << 1) | (prev.sp >> 63);
        let dp = (cur.d << 1) | (prev.d >> 63);
        // An uppercase letter opens a token after a lowercase one (o200k splits
        // `[U]*[L]+` runs at each lower→upper step). With letters in both classes
        // (`\p{Lm}\p{Lo}\p{M}`: `U` and `L` share them), after the run of lower-or-
        // both letters before it holds a lower-only one. When that run holds none,
        // the uppercase run continues the token — unless it is the last of its
        // `[U]` run: then `[U]*` backtracks to the both-class letter before it (the
        // `[L]+`) and the run opens a token (`both_split`).
        let (lop, both_split) = if BOTH {
            let lower = cur.a & !cur.up;
            let (q, carry) = fill_forward(lower & !cur.both, lower, lower_seen);
            let lop = (q << 1) | u64::from(lower_seen);
            lower_seen = carry;
            // A mark right after punct may belong to the punct run instead (the
            // grammar's `[^\s\p{L}\p{N}]` holds marks): the scalar scan's call.
            if cur.mark & ((cur.o << 1) | (prev.o >> 63)) != 0 {
                return Fused::Bad;
            }
            let bp = (cur.both << 1) | (prev.both >> 63);
            let (mut cand, mut split) = (cur.up & bp & !lop, 0u64);
            while cand != 0 {
                let c = cand.trailing_zeros();
                cand &= cand - 1;
                let after = c + (cur.up >> c).trailing_ones();
                if after >= 64 {
                    return Fused::Bad; // (the run crosses into the next word: rare)
                }
                if cur.a >> after & 1 == 0 {
                    split |= 1 << c;
                }
            }
            (lop, split)
        } else {
            let low = cur.a & !cur.up;
            ((low << 1) | ((prev.a & !prev.up) >> 63), 0)
        };
        let wp = (ow_open & an & !op & !spp) | ((cur.ws & !cur.nl) & an);
        let wp_prev = (wp << 1) | (prev_wp >> 63);
        let lrs = cur.a & !ap_;
        let case_split = if CL { 0 } else { cur.up & lop | both_split };
        let psp = cur.sp & on;
        let nonws =
            wp | (lrs & !wp_prev) | case_split | (cur.d & !dp) | (ow_open & !op & !spp & !an) | psp;
        let wcw = (cur.ws & !pt) & !wp & !psp;
        let wsc_prev = (wcw << 1) | (prev_wsc >> 63);
        let wsc_run_start = wcw & !wsc_prev;
        let ws_eff_next = ((cur.ws & !pt) >> 1) | ((next.ws & !npt) << 63);
        let non_nl = !cur.nl & !((cur.nl << 1) | (prev.nl >> 63));
        let mut ws_tail = wcw & wsc_prev & !ws_eff_next & non_nl;
        if w == last {
            ws_tail &= !(1u64 << ((n - 1) & 63));
        }
        let mut st = nonws | wsc_run_start | ws_tail;

        // Newline split: a token opens right after a whitespace run's last newline —
        // where no newline follows in the run. That run may continue into the next
        // word: whether a newline lies in its part there is the backward carry.
        let run = cur.ws & !pt;
        let after_nl = (cur.nl << 1) | (prev.nl >> 63);
        let carry_in = {
            let r = next.ws & !npt;
            if r & 1 == 0 {
                false
            } else {
                let len = r.trailing_ones();
                if len == 64 && next.nl & r == 0 {
                    // The run spans the next word and holds no newline there.
                    return Fused::LongRun;
                }
                let head = if len == 64 { !0 } else { (1u64 << len) - 1 };
                next.nl & r & head != 0
            }
        };
        let later_nl = if cur.nl & run == 0 && !carry_in {
            0
        } else {
            fill_forward((cur.nl & run).reverse_bits(), run.reverse_bits(), carry_in)
                .0
                .reverse_bits()
        };
        st |= after_nl & wcw & !later_nl;

        // Digit cap `\p{N}{1,CAP}`.
        let dw = cur.d;
        if CAP == 1 {
            st |= dw;
        } else if dw == 0 {
            dopen = false;
            dsince = 0;
        } else {
            let vld = if w == last && tail_bits < 64 {
                (1u64 << tail_bits) - 1
            } else {
                !0
            };
            let prev_is_digit = (prev.d >> 63) & 1 != 0;
            let mut m = dw & vld & !((dw << 1) | u64::from(prev_is_digit));
            if dopen && prev_is_digit {
                let mut s = 1u64 & dw;
                for _ in 0..((CAP - dsince % CAP) % CAP) {
                    s = (s << 1) & dw & vld;
                }
                m |= s;
            }
            let mut groups = m;
            let nk = dw & (dw << 1) & (dw << 2);
            let mut mm = m;
            while mm != 0 {
                mm = (mm << CAP) & nk;
                groups |= mm;
            }
            st |= groups;
            let len = if w == last { tail_bits } else { 64 };
            let next_is_digit = w < last && next.d & 1 != 0;
            let tn = if dw & (1u64 << (len - 1)) == 0 {
                0
            } else {
                let z = !dw & vld;
                if z == 0 {
                    vld
                } else {
                    vld & !((1u64 << (64 - z.leading_zeros())) - 1)
                }
            };
            dopen = tn != 0 && next_is_digit;
            dsince = if dopen {
                let g = groups & tn;
                let counted = if g == 0 {
                    dsince + (dw & vld & tn).count_ones()
                } else {
                    (dw & vld & tn & !((1u64 << (63 - g.leading_zeros())) - 1)).count_ones()
                };
                counted % CAP
            } else {
                0
            };
        }

        // Contraction suffix, glued onto the preceding word (see the multi-pass
        // scan): first what a contraction from the previous word did here.
        if CONTR {
            st = (st & !pend_clear) | pend_set;
            (pend_clear, pend_set) = (0, 0);
            let mut bits = cur.ap & st;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let p = w * 64 + b;
                if p == after_contraction {
                    continue; // opens a new word, not a suffix of the previous one
                }
                let letter_before = if b == 0 {
                    prev.a >> 63 != 0
                } else {
                    cur.a >> (b - 1) & 1 != 0
                };
                if p == 0 || !letter_before {
                    continue;
                }
                let k = contraction_len(bytes, p);
                if k == 0 {
                    continue;
                }
                // Clear `[p, p + k)` and open a token at `p + k` (possibly in the
                // next word).
                let (e, set) = (p + k, p + k < n);
                // The token after the suffix starts a new case run, which the case
                // split above (run through the suffix's letter) did not see: when it
                // begins with a both-class letter, the scalar scan decides.
                if BOTH && set && side.0.get(e >> 6).is_some_and(|&x| x >> (e & 63) & 1 != 0) {
                    return Fused::Bad;
                }
                if e <= (w + 1) * 64 {
                    let hi = e - w * 64; // <= 64
                    let m = if hi == 64 { !0 } else { (1u64 << hi) - 1 };
                    st &= !(m & !((1u64 << b) - 1));
                    if set {
                        if hi < 64 {
                            st |= 1u64 << hi;
                        } else {
                            pend_set |= 1;
                        }
                    }
                } else {
                    st &= !(!0u64 << b);
                    let hi = e - (w + 1) * 64; // 1..=2
                    pend_clear |= (1u64 << hi) - 1;
                    if set {
                        pend_set |= 1u64 << hi;
                    }
                }
                after_contraction = e;
            }
        }

        // cl100k's contraction alternative — highest priority, so it cuts the letter
        // run behind it (`'store` → `'s`, `tore`). The apostrophe already opens a
        // token (word-prefix rule); the literal's interior opens nothing, and the
        // char after it opens the next token — possibly in the next word.
        if CL {
            st = (st & !pend_clear) | pend_set;
            (pend_clear, pend_set) = (0, 0);
            let mut bits = cur.ap & st;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let p = w * 64 + b;
                let k = contraction_len(bytes, p);
                if k == 0 {
                    continue;
                }
                let (e, set) = (p + k, p + k < n);
                let keep = u64::MAX >> (63 - b); // bits 0..=b
                let hi = e - w * 64; // b + 2 ..= b + 3
                if hi <= 64 {
                    let upto = if hi == 64 { !0 } else { (1u64 << hi) - 1 };
                    st &= !(upto & !keep);
                    if set {
                        if hi < 64 {
                            st |= 1u64 << hi;
                        } else {
                            pend_set |= 1;
                        }
                    }
                } else {
                    st &= keep;
                    let h = hi - 64; // 1..=2
                    pend_clear |= (1u64 << h) - 1;
                    if set {
                        pend_set |= 1u64 << h;
                    }
                }
            }
        }

        // Kimi's `[\p{Han}]+`: a run of Han letters opens a token.
        if HAN {
            st |= cur.han & !((cur.han << 1) | (prev.han >> 63));
        }
        starts.w[w] = st;

        prev = cur;
        prev_wp = wp;
        prev_wsc = wcw;
        prev_ow_open = ow_open;
        cur = next;
        pt = npt;
        pt_carry = npt_carry;
    }
    Fused::Done(starts)
}

/// [`o200k_bulk_starts`] as separate passes over whole-text class bitmaps: the
/// reference the fused scan is checked against, and its fallback for long runs.
fn o200k_bulk_starts_multipass<
    const SLASH: bool,
    const CONTR: bool,
    const CAP: u32,
    const HAN: bool,
>(
    bytes: &[u8],
) -> Option<Bitmap> {
    const { assert!(CAP == 1 || CAP == 3) };
    let n = bytes.len();
    if n == 0 {
        return Some(Bitmap::zeros(0));
    }
    // Materialize class bitmaps (o200k needs case + `/`): `up` (uppercase) and
    // `slash` are the `CASED` extras; `lo` (lowercase letters) is derived below.
    let (mut a, mut d, mut o, mut ws, mut nl, mut sp, mut ap, mut up, mut slash) = (
        Bitmap::zeros(n),
        Bitmap::zeros(n),
        Bitmap::zeros(n),
        Bitmap::zeros(n),
        Bitmap::zeros(n),
        Bitmap::zeros(n),
        Bitmap::zeros(n),
        Bitmap::zeros(n),
        Bitmap::zeros(n),
    );
    let mut hb = if HAN { Bitmap::zeros(n) } else { Bitmap::zeros(0) };
    let mut any_han = 0u64;
    let mut off = 0;
    let mut wk = 0;
    let mut any_ap = 0u64;
    while off < n {
        // SAFETY: NEON is baseline on aarch64; on other targets this is the SWAR path.
        let c = unsafe_bulk_classify::<true, HAN>(bytes, off);
        if c.bad {
            return None;
        }
        a.w[wk] = c.alpha;
        d.w[wk] = c.digit;
        ws.w[wk] = c.ws;
        nl.w[wk] = c.nl;
        sp.w[wk] = c.sp;
        ap.w[wk] = c.ap;
        up.w[wk] = c.up;
        slash.w[wk] = c.slash;
        any_ap |= c.ap;
        // A Han stand-in is none of letter / digit / space / punct (Kimi).
        o.w[wk] = !(c.alpha | c.digit | c.ws | c.han);
        if HAN {
            hb.w[wk] = c.han;
            any_han |= c.han;
        }
        off += 64;
        wk += 1;
    }

    let nwords = a.w.len();
    let last = nwords - 1;
    let tail_bits = n - last * 64;
    if tail_bits < 64 {
        o.w[last] &= (1u64 << tail_bits) - 1;
    }

    // Run-reduction #2 — punct trailing `[\r\n/]*`: a `[\r\n/]` run whose first char
    // is a newline immediately after an `O` char (the other-run stopped at the
    // newline; `/` inside the run continues it). Iterate the sparse newline bits.
    let mut trail_class = Bitmap::zeroed_words(nl.w.len());
    for ((t, a), b) in trail_class.w.iter_mut().zip(&nl.w).zip(&slash.w) {
        *t = if SLASH { a | b } else { *a };
    }
    let punct_trailing = punct_trailing_runs(&nl, &o, &trail_class);

    // Fused word-loop → start bitmap. Adds o200k's case-split term `up & lo_prev`
    // and excludes trailing-run chars (`ptw`) from the token-opening `other` uses.
    let mut starts = Bitmap::zeros(n);
    let mut ws_content = Bitmap::zeros(n);
    let mut prev_wp = 0u64;
    let mut prev_wsc = 0u64;
    let (av, dv, ov, wsv, nlv, spv, upv, ptvv) = (
        &a.w,
        &d.w,
        &o.w,
        &ws.w,
        &nl.w,
        &sp.w,
        &up.w,
        &punct_trailing.w,
    );
    let (mut ca, mut cd, mut co, mut cws, mut cnl, mut csp, mut cup, mut cpt) =
        (av[0], dv[0], ov[0], wsv[0], nlv[0], spv[0], upv[0], ptvv[0]);
    let (mut ha, mut ho, mut hsp, mut hlo) = (0u64, 0u64, 0u64, 0u64);
    for w in 0..nwords {
        let nx = w + 1;
        let na = av.get(nx).copied().unwrap_or(0);
        let nd = dv.get(nx).copied().unwrap_or(0);
        let no = ov.get(nx).copied().unwrap_or(0);
        let nws = wsv.get(nx).copied().unwrap_or(0);
        let nnl = nlv.get(nx).copied().unwrap_or(0);
        let nsp = spv.get(nx).copied().unwrap_or(0);
        let nup = upv.get(nx).copied().unwrap_or(0);
        let npt = ptvv.get(nx).copied().unwrap_or(0);

        let aw = ca;
        let dw = cd;
        let ow = co;
        let wsw = cws;
        let nlw = cnl;
        let spw = csp;
        let upw = cup;
        let ptw = cpt;
        let low = aw & !upw; // lowercase letters
        // An `other` that can open/body a punct token — i.e. not a `[\r\n/]*` trailing
        // char. `op`/`on` (prev/next is a body-other) are computed from this so a
        // trailing char never chains onto the punct token before or after it.
        let ow_open = ow & !ptw;
        let no_open = no & !npt;
        let ap_ = (aw << 1) | ha;
        let an = (aw >> 1) | (na << 63);
        let op = (ow_open << 1) | ho;
        let on = (ow_open >> 1) | ((no_open & 1) << 63);
        let spp = (spw << 1) | hsp;
        let dp = (dw << 1) | hd_zero(w, &d.w);
        let lop = (low << 1) | hlo; // lowercase shifted: previous byte was lowercase

        let wp = (ow_open & an & !op & !spp) | ((wsw & !nlw) & an);
        let wp_prev = (wp << 1) | (prev_wp >> 63);
        let lrs = aw & !ap_;
        let case_split = upw & lop; // uppercase preceded by lowercase
        let psp = spw & on;
        let nonws =
            wp | (lrs & !wp_prev) | case_split | (dw & !dp) | (ow_open & !op & !spp & !an) | psp;

        let wcw = (wsw & !ptw) & !wp & !psp;
        let wsc_prev = (wcw << 1) | (prev_wsc >> 63);
        let wsc_run_start = wcw & !wsc_prev;
        let ws_eff_next = ((wsw & !ptw) >> 1) | ((nws & !npt) << 63);
        let non_nl = !nlw & !((nlw << 1) | hnl_zero(w, &nl.w));
        let mut ws_tail = wcw & wsc_prev & !ws_eff_next & non_nl;
        if w == last {
            ws_tail &= !(1u64 << ((n - 1) & 63));
        }
        starts.w[w] = nonws | wsc_run_start | ws_tail;
        ws_content.w[w] = wcw;
        prev_wp = wp;
        prev_wsc = wcw;

        ha = aw >> 63;
        ho = ow_open >> 63;
        hsp = spw >> 63;
        hlo = low >> 63;
        ca = na;
        cd = nd;
        co = no;
        cws = nws;
        cnl = nnl;
        csp = nsp;
        cup = nup;
        cpt = npt;
    }

    // Run-reduction #1 — newline split (identical to cl100k: `\s*[\r\n]+` runs
    // through its run's last newline; a token opens right after it).
    newline_split(&nl, &ws, &punct_trailing, &ws_content, &mut starts);

    // Digit `\p{N}{1,3}` cap — identical to cl100k.
    if CAP == 1 {
        // `\p{N}`: every digit is its own token.
        for (st, &dw) in starts.w.iter_mut().zip(&d.w) {
            *st |= dw;
        }
    } else {
        let mut dopen = false;
        let mut dsince = 0u32;
        let mut prev_dw = 0u64;
        for w in 0..nwords {
            let dw = d.w[w];
            let vld = if w == last {
                if tail_bits == 64 {
                    !0
                } else {
                    (1u64 << tail_bits) - 1
                }
            } else {
                !0
            };
            if dw == 0 {
                dopen = false;
                dsince = 0;
                prev_dw = 0;
                continue;
            }
            let prev_is_digit = (prev_dw >> 63) & 1 != 0;
            let mut m = dw & vld & !((dw << 1) | u64::from(prev_is_digit));
            if dopen && prev_is_digit {
                let mut s = 1u64 & dw;
                for _ in 0..((CAP - dsince % CAP) % CAP) {
                    s = (s << 1) & dw & vld;
                }
                m |= s;
            }
            let mut groups = m;
            let nk = dw & (dw << 1) & (dw << 2);
            let mut mm = m;
            while mm != 0 {
                mm = (mm << CAP) & nk;
                groups |= mm;
            }
            starts.w[w] |= groups;
            let len = if w == last { tail_bits } else { 64 };
            let next_is_digit = w + 1 < nwords && d.w[w + 1] & 1 != 0;
            let tn = if dw & (1u64 << (len - 1)) == 0 {
                0
            } else {
                let z = !dw & vld;
                if z == 0 {
                    vld
                } else {
                    vld & !((1u64 << (64 - z.leading_zeros())) - 1)
                }
            };
            dopen = tn != 0 && next_is_digit;
            dsince = if dopen {
                let g = groups & tn;
                let counted = if g == 0 {
                    dsince + (dw & vld & tn).count_ones()
                } else {
                    (dw & vld & tn & !((1u64 << (63 - g.leading_zeros())) - 1)).count_ones()
                };
                counted % CAP
            } else {
                0
            };
            prev_dw = dw;
        }
    }

    // Contraction suffix `(?i:'s|'t|'re|'ve|'m|'ll|'d)` — o200k glues it onto the
    // preceding word, so *merge* rather than split: at each apostrophe start-bit that
    // follows a letter and opens a valid contraction, clear the token starts across
    // `[p, p + k)` (the apostrophe and its letters) and open the next token after it.
    // A word takes at most ONE contraction (`(contraction)?`), so this is a
    // left-to-right pass: an apostrophe sitting exactly where a just-merged
    // contraction ended is a NEW word's prefix (its preceding letter is a contraction
    // letter, not a base-word letter), so it does not merge — that is how
    // `'all'd've` becomes `'all'd`, `'ve` rather than one token.
    if CONTR && any_ap != 0 {
        let mut after_contraction = usize::MAX;
        for w in 0..nwords {
            let mut bits = ap.w[w] & starts.w[w];
            while bits != 0 {
                let p = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if p == after_contraction {
                    continue; // opens a new word, not a suffix of the previous one
                }
                if p == 0 || !a.get(p - 1) {
                    continue; // not a word suffix — leave to the normal rules
                }
                let k = contraction_len(bytes, p);
                if k == 0 {
                    continue;
                }
                for q in p..(p + k) {
                    starts.w[q >> 6] &= !(1u64 << (q & 63));
                }
                if p + k < n {
                    starts.w[(p + k) >> 6] |= 1u64 << ((p + k) & 63);
                }
                after_contraction = p + k;
            }
        }
    }

    // Kimi's `[\p{Han}]+` over Han letters (the stand-in `0x80`, see `kimi_rep`): a
    // run of them is always its own token — no prefix joins it, and no other class
    // contains a Han letter — so its first char opens one; what follows it opens one
    // by its own rules.
    if HAN && any_han != 0 {
        let mut prev = 0u64;
        for (st, &h) in starts.w.iter_mut().zip(&hb.w) {
            *st |= h & !((h << 1) | prev);
            prev = h >> 63;
        }
    }
    Some(starts)
}

/// [`try_scan_o200k_bulk`], with "not pure ASCII" reported as an `Err` (nothing emitted).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn scan_o200k_bulk<F>(text: &str, emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    if try_scan_o200k_bulk(text, emit)? {
        Ok(())
    } else {
        Err("non-ascii".into())
    }
}

/// Previous-word carry bit for `digit` at word `w` (bit 63 of word `w-1`, else 0).
#[inline]
fn hd_zero(w: usize, dw: &[u64]) -> u64 {
    if w > 0 { dw[w - 1] >> 63 } else { 0 }
}

/// Previous-word carry bit for `nl` at word `w` (bit 63 of word `w-1`, else 0).
#[inline]
fn hnl_zero(w: usize, nlw: &[u64]) -> u64 {
    if w > 0 { nlw[w - 1] >> 63 } else { 0 }
}

// ── x86-64 AVX2 classify kernels ─────────────────────────────────────────────

/// AVX2 versions of the classify kernels (x86-64), selected at runtime. Each
/// classifies a 64-byte block as two 32-byte vectors and gathers every class with
/// one `movemask` per half — the portable SWAR kernel spends ~25x the instructions
/// on the same block. Outputs are bit-identical to the portable kernels
/// (`avx2_matches_portable`), including the zeroed bits past a short final block.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)] // AVX2 intrinsics; every fn is an unsafe contract
mod avx2 {
    use std::arch::x86_64::*;

    use super::{BulkClasses, Classes, DS_CG, DS_CL, DS_CP, DS_VL, DS_VP, DsClasses};

    /// Whether this CPU runs the AVX2 kernels (cached by `std` after the first call).
    #[inline(always)]
    pub(super) fn available() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    /// The 64-byte block at `off` as two vectors, zero-filled past the end of
    /// `bytes`, plus the mask of its valid bits.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load(bytes: &[u8], off: usize) -> (__m256i, __m256i, u64) {
        if off + 64 <= bytes.len() {
            let p = bytes.as_ptr().add(off);
            return (
                _mm256_loadu_si256(p as *const __m256i),
                _mm256_loadu_si256(p.add(32) as *const __m256i),
                !0,
            );
        }
        let tail = &bytes[off.min(bytes.len())..];
        let mut buf = [0u8; 64];
        buf[..tail.len()].copy_from_slice(tail);
        let p = buf.as_ptr();
        (
            _mm256_loadu_si256(p as *const __m256i),
            _mm256_loadu_si256(p.add(32) as *const __m256i),
            (1u64 << tail.len()) - 1, // tail.len() < 64 here
        )
    }

    /// Lane masks of both halves → one 64-bit class word (lane `j` → bit `j`).
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mm(lo: __m256i, hi: __m256i) -> u64 {
        (_mm256_movemask_epi8(lo) as u32 as u64) | ((_mm256_movemask_epi8(hi) as u32 as u64) << 32)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn eq(v: __m256i, c: u8) -> __m256i {
        _mm256_cmpeq_epi8(v, _mm256_set1_epi8(c as i8))
    }

    /// Lanes in `[lo, hi]` for ASCII bounds `1 <= lo <= hi < 0x7F`. The compare is
    /// signed, so bytes `>= 0x80` (negative) never match.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn range(v: __m256i, lo: u8, hi: u8) -> __m256i {
        _mm256_and_si256(
            _mm256_cmpgt_epi8(v, _mm256_set1_epi8(lo as i8 - 1)),
            _mm256_cmpgt_epi8(_mm256_set1_epi8(hi as i8 + 1), v),
        )
    }

    /// `[A-Za-z]`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn alpha(v: __m256i) -> __m256i {
        range(_mm256_or_si256(v, _mm256_set1_epi8(0x20)), b'a', b'z')
    }

    /// `\s` over ASCII: `\t`..=`\r` and space.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn ws(v: __m256i) -> __m256i {
        _mm256_or_si256(range(v, 0x09, 0x0D), eq(v, b' '))
    }

    /// `[\r\n]`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn nl(v: __m256i) -> __m256i {
        _mm256_or_si256(eq(v, b'\n'), eq(v, b'\r'))
    }

    /// Apply a per-vector class test to both halves and gather the class word.
    macro_rules! word {
        ($lo:expr, $hi:expr, $f:expr) => {
            mm($f($lo), $f($hi))
        };
    }

    /// AVX2 [`super::classify_block_swar`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn classify_block(bytes: &[u8], off: usize) -> Classes {
        let (lo, hi, valid) = load(bytes, off);
        let alpha = word!(lo, hi, alpha) & valid;
        let digit = word!(lo, hi, |v| range(v, b'0', b'9')) & valid;
        let ws = word!(lo, hi, ws) & valid;
        let nonascii = mm(lo, hi) & valid;
        Classes {
            alpha,
            digit,
            ws,
            nl: word!(lo, hi, nl) & valid,
            other: valid & !nonascii & !(alpha | digit | ws),
            nonascii,
            sp: word!(lo, hi, |v| eq(v, b' ')) & valid,
            ap: word!(lo, hi, |v| eq(v, b'\'')) & valid,
        }
    }

    /// AVX2 [`super::classify_block_bulk_portable`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn classify_block_bulk<const CASED: bool, const HAN: bool>(
        bytes: &[u8],
        off: usize,
    ) -> BulkClasses {
        let (lo, hi, valid) = load(bytes, off);
        let nonascii = mm(lo, hi) & valid;
        let han = if HAN {
            word!(lo, hi, |v| eq(v, 0x80)) & valid
        } else {
            0
        };
        let (up, slash) = if CASED {
            (
                word!(lo, hi, |v| range(v, b'A', b'Z')) & valid,
                word!(lo, hi, |v| eq(v, b'/')) & valid,
            )
        } else {
            (0, 0)
        };
        BulkClasses {
            alpha: word!(lo, hi, alpha) & valid,
            digit: word!(lo, hi, |v| range(v, b'0', b'9')) & valid,
            ws: word!(lo, hi, ws) & valid,
            nl: word!(lo, hi, nl) & valid,
            sp: word!(lo, hi, |v| eq(v, b' ')) & valid,
            ap: word!(lo, hi, |v| eq(v, b'\'')) & valid,
            up,
            slash,
            han,
            // Only non-ASCII is bad — except Kimi's Han stand-in `0x80` when `HAN`.
            bad: nonascii & !han != 0,
        }
    }

    /// AVX2 [`super::classify_block_ds_scalar`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn classify_block_ds<const VIRT: bool>(
        bytes: &[u8],
        off: usize,
    ) -> DsClasses {
        let (lo, hi, valid) = load(bytes, off);
        let ws = word!(lo, hi, ws) & valid;
        // ASCII controls that are not whitespace: `0x00..=0x1F` (a signed `> -1`
        // keeps bytes `>= 0x80` out of `< 0x20`) outside `\s`, and DEL.
        let low = |v: __m256i| {
            _mm256_and_si256(
                _mm256_cmpgt_epi8(_mm256_set1_epi8(0x20), v),
                _mm256_cmpgt_epi8(v, _mm256_set1_epi8(-1)),
            )
        };
        let ctrl = ((word!(lo, hi, low) & !ws) | word!(lo, hi, |v| eq(v, 0x7F))) & valid;
        let hib = mm(lo, hi) & valid;
        let mut c = DsClasses {
            alpha: word!(lo, hi, alpha) & valid,
            digit: word!(lo, hi, |v| range(v, b'0', b'9')) & valid,
            ws,
            nl: word!(lo, hi, nl) & valid,
            sp: word!(lo, hi, |v| eq(v, b' ')) & valid,
            ctrl,
            hi: hib,
            ..DsClasses::default()
        };
        if VIRT && hib != 0 {
            c.vl = word!(lo, hi, |v| eq(v, DS_VL)) & valid;
            c.vp = word!(lo, hi, |v| eq(v, DS_VP)) & valid;
            c.cl = word!(lo, hi, |v| eq(v, DS_CL)) & valid;
            c.cp = word!(lo, hi, |v| eq(v, DS_CP)) & valid;
            c.cg = word!(lo, hi, |v| eq(v, DS_CG)) & valid;
        }
        c
    }
}

/// AVX-512BW class kernels: the 64-byte block is one vector and every class
/// test lands in a mask register — a third of the AVX2 kernels' instructions
/// (two halves, a `movemask` each), and the tail is a masked load, not a copy.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod avx512 {
    use std::arch::x86_64::*;

    use super::BulkClasses;

    /// Whether this CPU runs the AVX-512BW kernels (cached by `std`).
    #[inline(always)]
    pub(super) fn available() -> bool {
        std::is_x86_feature_detected!("avx512bw")
    }

    /// Lanes equal to `c`.
    #[target_feature(enable = "avx512bw")]
    #[inline]
    unsafe fn eq(v: __m512i, c: u8) -> u64 {
        _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(c as i8))
    }

    /// Lanes in `[lo, hi]`, ASCII bounds: `v - lo < hi - lo + 1` unsigned (a byte
    /// `>= 0x80` wraps to at least `0x80 - lo`, past the width).
    #[target_feature(enable = "avx512bw")]
    #[inline]
    unsafe fn range(v: __m512i, lo: u8, hi: u8) -> u64 {
        _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(lo as i8)),
            _mm512_set1_epi8((hi - lo + 1) as i8),
        )
    }

    /// AVX-512BW [`super::classify_block_bulk_portable`].
    #[target_feature(enable = "avx512bw")]
    pub(super) unsafe fn classify_block_bulk<const CASED: bool, const HAN: bool>(
        bytes: &[u8],
        off: usize,
    ) -> BulkClasses {
        let n = bytes.len().saturating_sub(off).min(64);
        let valid = if n == 64 { !0u64 } else { (1u64 << n) - 1 };
        // Lanes past the end load as zero (a NUL: no class below).
        let v =
            _mm512_maskz_loadu_epi8(valid, bytes.as_ptr().add(off.min(bytes.len())) as *const i8);
        let nonascii = _mm512_movepi8_mask(v);
        let han = if HAN { eq(v, 0x80) } else { 0 };
        let (up, slash) = if CASED {
            (range(v, b'A', b'Z'), eq(v, b'/'))
        } else {
            (0, 0)
        };
        let sp = eq(v, b' ');
        BulkClasses {
            alpha: range(_mm512_or_si512(v, _mm512_set1_epi8(0x20)), b'a', b'z'),
            digit: range(v, b'0', b'9'),
            ws: range(v, 0x09, 0x0D) | sp,
            nl: eq(v, b'\n') | eq(v, b'\r'),
            sp,
            ap: eq(v, b'\''),
            up,
            slash,
            han,
            // Only non-ASCII is bad — except Kimi's Han stand-in `0x80` when `HAN`.
            bad: nonascii & !han != 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn o200k_fused_matches_multipass() {
        // Dense in everything the boundary rules key on, across block edges.
        let pool: &[&[u8]] = &[
            b"a",
            b"b",
            b"Z",
            b"Q",
            b" ",
            b"  ",
            b"\n",
            b"\r\n",
            b"\t",
            b"'",
            b"'s",
            b"'ll",
            b"'VE",
            b"'d",
            b"'",
            b"1",
            b"12",
            b"123",
            b"4567",
            b"/",
            b"//",
            b".",
            b"!?",
            b"-",
            b"\x80",
            b"\x80\x80",
            b"x",
            b"Hello",
            b"camelCase",
            b"HTTPRequest",
            b"don't",
            b"\n\n",
            b" \n ",
            b"\x0b",
            b"#",
            b"_",
            b"'lL",
            b"'rE",
            b"'vE",
            b"'Ll",
            b"'D",
            b"'T",
        ];
        let mut state = 0x1357_9bdf_2468_ace0u64;
        let check = |t: &[u8], han: bool| {
            let fused = |r: Option<Bitmap>| r.map(|b| b.w.clone());
            assert_eq!(
                fused(o200k_bulk_starts::<true, true, 3, false>(t)),
                fused(o200k_bulk_starts_multipass::<true, true, 3, false>(t)),
                "o200k {t:?}"
            );
            assert_eq!(
                fused(o200k_bulk_starts::<true, false, 1, false>(t)),
                fused(o200k_bulk_starts_multipass::<true, false, 1, false>(t)),
                "tekken {t:?}"
            );
            assert_eq!(
                fused(o200k_bulk_starts::<false, true, 3, false>(t)),
                fused(o200k_bulk_starts_multipass::<false, true, 3, false>(t)),
                "kimi {t:?}"
            );
            if han {
                assert_eq!(
                    fused(o200k_bulk_starts::<false, true, 3, true>(t)),
                    fused(o200k_bulk_starts_multipass::<false, true, 3, true>(t)),
                    "kimi han {t:?}"
                );
            }
        };
        for round in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let target = (state % 400) as usize;
            let mut t = Vec::new();
            while t.len() < target {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                t.extend_from_slice(pool[(state % pool.len() as u64) as usize]);
            }
            // Long blank runs now and then (the fallback), and plain ASCII.
            if round % 50 == 0 {
                let at = (state >> 20) as usize % (t.len() + 1);
                let blanks = 60 + (state >> 40) as usize % 80;
                t.splice(at..at, std::iter::repeat_n(b' ', blanks));
            }
            check(&t, true);
            let ascii: Vec<u8> = t
                .iter()
                .map(|&b| if b >= 0x80 { b'x' } else { b })
                .collect();
            check(&ascii, false);
        }
    }

    /// [`transcode`] with a stand-in that may refuse a char (`None`): stops there and
    /// returns `false` (the buffers then hold a prefix, to be discarded).
    fn transcode_opt_ref(
        b: &[u8],
        first_na: usize,
        buf: &mut Vec<u8>,
        runs: &mut Vec<MbRun>,
        mut rep: impl FnMut(u32) -> Option<u8>,
    ) -> bool {
        buf.clear();
        runs.clear();
        buf.reserve(b.len());
        let mut acc = 0u32;
        let mut i = first_na;
        buf.extend_from_slice(&b[..i]);
        loop {
            while i < b.len() && b[i] >= 0x80 {
                let (cp, len) = decode_multibyte(b, i);
                let Some(r) = rep(cp) else { return false };
                let (c, w1) = (buf.len() as u32, len as u32 - 1);
                match runs.last_mut() {
                    Some(run) if run.w1 == w1 && run.c + run.n == c => run.n += 1,
                    _ => runs.push(MbRun { c, n: 1, w1, acc }),
                }
                acc += w1;
                buf.push(r);
                i += len;
            }
            match first_non_ascii(&b[i..]) {
                Some(r) => {
                    buf.extend_from_slice(&b[i..i + r]);
                    i += r;
                }
                None => {
                    buf.extend_from_slice(&b[i..]);
                    return true;
                }
            }
        }
    }

    /// The stretch-wise transcoder records exactly the stand-ins and runs of the
    /// char-at-a-time walk (and refuses at the same char), on mixed-width text.
    #[test]
    fn transcode_stretches_match_char_walk() {
        let pool = [
            "a",
            " ",
            "é",
            "ж",
            "中",
            "文",
            "ሀ",
            "😀",
            "\u{10348}",
            "1",
            "\n",
            "ק",
            "न",
            "ก",
            "߷",
            "!",
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for round in 0..20_000 {
            let mut t = String::new();
            let len = (round % 40) + 1;
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                t.push_str(pool[(state % pool.len() as u64) as usize]);
            }
            let b = t.as_bytes();
            let Some(first_na) = b.iter().position(|&x| x >= 0x80) else {
                continue;
            };
            // A stand-in that refuses one char now and then.
            let refuse = if round % 7 == 0 { 0x1F600 } else { u32::MAX };
            let st = |cp: u32| (cp != refuse).then_some((cp % 91) as u8 + 32);
            let (mut b1, mut r1, mut b2, mut r2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let full1 =
                transcode_opt::<false>(b, first_na, &mut b1, &mut r1, &mut SideBits::default(), st);
            let full2 = transcode_opt_ref(b, first_na, &mut b2, &mut r2, st);
            assert_eq!(full1, full2, "{t:?}");
            if full1 {
                assert_eq!(b1, b2, "{t:?}");
                let key = |r: &[MbRun]| {
                    r.iter()
                        .map(|m| (m.c, m.n, m.w1, m.acc))
                        .collect::<Vec<_>>()
                };
                assert_eq!(key(&r1), key(&r2), "{t:?}");
            }
        }
    }

    /// The fused cl100k scan is the multi-pass one, bit for bit (both digit caps),
    /// on text dense in its rules — contractions at and across block edges
    /// (`'s'll`, `'store`, `don't`), punct newline tails, whitespace runs, digit
    /// runs — including the long blank runs it hands back.
    #[test]
    fn cl100k_fused_matches_multipass() {
        let pool: &[&[u8]] = &[
            b"a",
            b"b",
            b"Z",
            b" ",
            b"  ",
            b"\n",
            b"\r\n",
            b"\t",
            b"'",
            b"'s",
            b"'S",
            b"'ll",
            b"'LL",
            b"'lL",
            b"'ve",
            b"'Re",
            b"'re",
            b"'d",
            b"'m",
            b"'t",
            b"'store",
            b"'s'll",
            b"''",
            b"1",
            b"12",
            b"123",
            b"4567",
            b".",
            b"!?",
            b"-",
            b"/",
            b"#",
            b"_",
            b"x",
            b"Hello",
            b"camelCase",
            b"don't",
            b"I'M",
            b"\n\n",
            b" \n ",
            b"\x0b",
            b"'\n",
            b"\x80",
        ];
        let mut state = 0x0f1e_2d3c_4b5a_6978u64;
        let check = |t: &[u8]| {
            let Ok(text) = std::str::from_utf8(t) else {
                return;
            };
            let bits = |r: Option<Bitmap>| r.map(|b| b.w.clone());
            assert_eq!(
                bits(cl100k_bulk_starts::<3>(text)),
                bits(cl100k_bulk_starts_multipass::<3>(text)),
                "cl100k {t:?}"
            );
            assert_eq!(
                bits(cl100k_bulk_starts::<1>(text)),
                bits(cl100k_bulk_starts_multipass::<1>(text)),
                "cl100k cap 1 {t:?}"
            );
        };
        for round in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let target = (state % 400) as usize;
            let mut t = Vec::new();
            while t.len() < target {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                t.extend_from_slice(pool[(state % pool.len() as u64) as usize]);
            }
            if round % 50 == 0 {
                let at = (state >> 20) as usize % (t.len() + 1);
                let blanks = 60 + (state >> 40) as usize % 80;
                t.splice(at..at, std::iter::repeat_n(b' ', blanks));
            }
            let ascii: Vec<u8> = t
                .iter()
                .map(|&b| if b >= 0x80 { b'x' } else { b })
                .collect();
            check(&ascii);
            check(&t);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_matches_portable() {
        if !avx512::available() {
            return;
        }
        let mut state = 0x7654_3210_fedc_ba98u64;
        for len in 0..=200usize {
            let mut buf = Vec::with_capacity(len);
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let b = match state & 3 {
                    0 => (state >> 8) as u8,
                    1 => [
                        0x00, 0x08, 0x09, 0x0D, 0x0E, 0x1F, 0x20, 0x7F, 0x80, 0x81, 0x82, 0x83,
                        0x84, 0x85, 0xFF,
                    ][(state >> 8) as usize % 15],
                    _ => b" \t\n\rAZaz@[`{09/:'!_.,;#"[(state >> 8) as usize % 23],
                };
                buf.push(b);
            }
            for off in [0usize, 1, 7, 8, 31, 32, 63, 64, 65, 128] {
                if off > buf.len() {
                    continue;
                }
                let ctx = format!("len {len} off {off}");
                unsafe {
                    assert_eq!(
                        avx512::classify_block_bulk::<false, false>(&buf, off),
                        classify_block_bulk_portable::<false, false>(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx512::classify_block_bulk::<true, false>(&buf, off),
                        classify_block_bulk_portable::<true, false>(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx512::classify_block_bulk::<true, true>(&buf, off),
                        classify_block_bulk_portable::<true, true>(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx512::classify_block_bulk::<false, true>(&buf, off),
                        classify_block_bulk_portable::<false, true>(&buf, off),
                        "{ctx}"
                    );
                }
            }
        }
    }

    /// Scalar reference: classify one byte into the same class booleans.
    fn scalar_class(b: u8) -> (bool, bool, bool, bool, bool, bool) {
        let nonascii = b >= 0x80;
        if nonascii {
            return (false, false, false, false, false, true);
        }
        let alpha = (b | 0x20).wrapping_sub(b'a') < 26;
        let digit = b.wrapping_sub(b'0') < 10;
        let ws = matches!(b, 9..=13 | 32);
        let nl = b == b'\n' || b == b'\r';
        let other = !(alpha || digit || ws);
        (alpha, digit, ws, nl, other, false)
    }

    fn bit(map: u64, j: usize) -> bool {
        (map >> j) & 1 != 0
    }

    #[test]
    fn classify_matches_scalar() {
        // Deterministic pseudo-random bytes across the full 0..=255 range plus
        // structured ASCII, at every block length 0..=64 and sub-8 tail.
        let mut state = 0x1234_5678_9abc_def0u64;
        for len in 0..=64usize {
            let mut buf = Vec::with_capacity(len);
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Bias toward ASCII so all classes are well exercised.
                let b = if state & 3 == 0 {
                    (state >> 8) as u8
                } else {
                    b" \t\n\rAZaz09!'/_.,:;@#"[(state >> 8) as usize % 20]
                };
                buf.push(b);
            }
            let c = classify_block(&buf, 0);
            for (j, &b) in buf.iter().enumerate() {
                let (a, d, w, nl, o, na) = scalar_class(b);
                assert_eq!(bit(c.alpha, j), a, "alpha byte {b:#x} at {j} len {len}");
                assert_eq!(bit(c.digit, j), d, "digit byte {b:#x} at {j}");
                assert_eq!(bit(c.ws, j), w, "ws byte {b:#x} at {j}");
                assert_eq!(bit(c.nl, j), nl, "nl byte {b:#x} at {j}");
                assert_eq!(bit(c.other, j), o, "other byte {b:#x} at {j}");
                assert_eq!(bit(c.nonascii, j), na, "nonascii byte {b:#x} at {j}");
                assert_eq!(bit(c.sp, j), b == b' ', "sp byte {b:#x} at {j}");
                assert_eq!(bit(c.ap, j), b == b'\'', "ap byte {b:#x} at {j}");
            }
            // Bytes past the block length must be unset everywhere.
            for j in buf.len()..64 {
                let all = c.alpha | c.digit | c.ws | c.nl | c.other | c.nonascii | c.sp | c.ap;
                assert!(!bit(all, j));
            }
        }
    }

    fn scalar_cl100k_spans(text: &str) -> Vec<(usize, usize)> {
        use crate::pre_tokenizers::scan::{ScanKind, scan_core};
        let mut out = Vec::new();
        scan_core(ScanKind::Cl100k, text, |s, e| {
            out.push((s, e));
            Ok(())
        })
        .unwrap();
        out
    }

    fn simd_cl100k_spans(text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        scan_cl100k_ascii(text, |s, e| {
            out.push((s, e));
            Ok(())
        })
        .unwrap();
        out
    }

    #[test]
    fn cl100k_simd_matches_scalar() {
        let corpus: Vec<String> = vec![
            "".into(),
            "hello world".into(),
            "The quick brown fox. Don't you think it's O'Brien's?".into(),
            "HTTPRequest HelloWorld camelCase ALLCAPS iOS getHTTPResponseCode()".into(),
            "a.b a..b a. b _foo .NET /usr/bin/env x!b (hello) [tag] {x} a!!!b".into(),
            "1234567 3.14 1,000 42 007 v1.2.3 100000000 a1b2c3".into(),
            "  leading   trailing  a  b\ta\t\tb end.\n\nfoo\r\nbar\n\n  x  \n  y".into(),
            "don't I'll they're y'all can't won't 'tis 'twas o'clock rock'n'roll".into(),
            "'s 's'x '' ''' a'b a''b 3's it's...here".into(),
            "def f(x: int) -> int:\n    return x*2 + 1  # doubles\n".into(),
            "{\"key\": \"value\", \"n\": 42, \"arr\": [1, 2, 3]}".into(),
            "\t\ttabs   spaces\n\n\n\nnewlines\r\r\rreturns \n \t\n mixed".into(),
            "SELECT id FROM t WHERE a>=18 AND s!='x'; -- comment".into(),
            "a\x00b\x07c\x1bd control chars e.g. i.e. U.S.A. snake_case-kebab".into(),
            "The rain in Spain. ".repeat(40),
        ];
        for s in &corpus {
            assert_eq!(
                simd_cl100k_spans(s),
                scalar_cl100k_spans(s),
                "corpus input {s:?}"
            );
        }

        // ASCII fuzz across lengths, biased to the tricky alphabet.
        let alpha = b" \t\n\r'.!/_,-abcABC01()\"";
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for len in 0..=160usize {
            for _ in 0..200 {
                let mut buf = Vec::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    buf.push(alpha[(state >> 11) as usize % alpha.len()]);
                }
                let s = std::str::from_utf8(&buf).unwrap();
                assert_eq!(
                    simd_cl100k_spans(s),
                    scalar_cl100k_spans(s),
                    "fuzz input {s:?}"
                );
            }
        }
    }

    fn bulk_cl100k_spans(text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        scan_cl100k_bulk(text, |s, e| {
            out.push((s, e));
            Ok(())
        })
        .unwrap();
        out
    }

    fn scalar_o200k_spans(text: &str) -> Vec<(usize, usize)> {
        use crate::pre_tokenizers::scan::{ScanKind, scan_core};
        let mut out = Vec::new();
        scan_core(ScanKind::O200k, text, |s, e| {
            out.push((s, e));
            Ok(())
        })
        .unwrap();
        out
    }

    fn bulk_o200k_spans(text: &str) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        scan_o200k_bulk(text, |s, e| {
            out.push((s, e));
            Ok(())
        })
        .unwrap();
        out
    }

    /// `scan_fast` on mixed text (ASCII prose with sprinkled non-ASCII, so both the
    /// island cuts and the bulk stretches between them are exercised) must equal the
    /// scalar scanner exactly, for cl100k and o200k.
    #[test]
    fn scan_fast_islands_match_scalar() {
        use crate::pre_tokenizers::scan::{ScanKind, scan_core};
        let collect = |kind, t: &str, fast: bool| {
            let mut v = Vec::new();
            let f = |a: usize, b: usize| {
                v.push((a, b));
                Ok(())
            };
            if fast {
                scan_fast(kind, t, f).unwrap()
            } else {
                scan_core(kind, t, f).unwrap()
            }
            v
        };
        let ascii: &[&str] = &[
            " the",
            " quick",
            " Brown",
            " fox",
            ".",
            ",",
            " 12345",
            "\n",
            "\n\n",
            "  ",
            "\t",
            " don't",
            " I'll",
            "'s",
            " x!",
            " (a)",
            " /usr/bin",
            "!\n/",
            " a",
            "HTTPRequest",
            " camelCase",
            "   ",
            "?",
            " -",
            "\r\n",
        ];
        let non: &[&str] = &[
            "é",
            "’",
            "“",
            "”",
            "—",
            "中文",
            "😀",
            "\u{00a0}",
            "\u{3000}",
            "e\u{0301}",
            "ß",
            " café",
            " naïve",
            "…",
            "\u{00a0}word",
            "£5",
        ];
        let mut state = 0x5eed_1234_abcd_9876u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..1500 {
            let target = 200 + (next() % 3000) as usize;
            let every = 1 + (next() % 250) as usize; // non-ASCII density
            let mut t = String::new();
            let mut k = 0;
            while t.len() < target {
                if k % every == every - 1 {
                    t.push_str(non[(next() % non.len() as u64) as usize]);
                } else {
                    t.push_str(ascii[(next() % ascii.len() as u64) as usize]);
                }
                k += 1;
            }
            for kind in [
                ScanKind::Cl100k,
                ScanKind::Qwen,
                ScanKind::Qwen35,
                ScanKind::O200k,
                ScanKind::Tekken,
                ScanKind::Kimi,
                ScanKind::DeepSeek,
                ScanKind::Gpt2,
            ] {
                assert_eq!(
                    collect(kind, &t, true),
                    collect(kind, &t, false),
                    "round {round} {kind:?} {t:?}"
                );
            }
        }
    }

    /// Dense random char soup over every class the grammars distinguish, with
    /// multibyte members of each (digits, whitespace, marks, caseless/titlecase
    /// letters, symbols), against the scalar scanner.
    #[test]
    fn scan_fast_unicode_soup() {
        use crate::pre_tokenizers::scan::{ScanKind, scan_core};
        let collect = |kind, t: &str, fast: bool| {
            let mut v = Vec::new();
            let f = |a: usize, b: usize| {
                v.push((a, b));
                Ok(())
            };
            if fast {
                scan_fast(kind, t, f).unwrap()
            } else {
                scan_core(kind, t, f).unwrap()
            }
            v
        };
        let pool: &[&str] = &[
            "a",
            "Z",
            "s",
            "t",
            "e",
            "l",
            "'",
            "'",
            "0",
            "7",
            " ",
            " ",
            "\t",
            "\n",
            "\r",
            "\x0b",
            "\x0c",
            "!",
            ".",
            "/",
            "(",
            "-",
            "_",
            "é",
            "É",
            "ß",
            "ſ",
            "ǅ",
            "中",
            "文",
            "ー",
            "ʰ",
            "\u{0301}",
            "\u{094d}",
            "हि",
            "م",
            "١",
            "٣",
            "１",
            "２",
            "²",
            "½",
            "Ⅻ",
            "\u{0085}",
            "\u{00a0}",
            "\u{2000}",
            "\u{2028}",
            "\u{202f}",
            "\u{3000}",
            "’",
            "“",
            "—",
            "…",
            "，",
            "。",
            "😀",
            "👍",
            "\u{1f3fd}",
            "\u{200d}",
            "€",
            "£",
            "\u{0000}",
            "\u{0001}",
            "\u{007f}",
            "\u{feff}",
            "𝐀",
            "𝟎",
            "\u{10ffff}",
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..20000 {
            let len = 1 + (next() % 300) as usize;
            let ascii_bias = next() % 4; // 0: uniform .. 3: mostly ASCII
            let mut t = String::new();
            for _ in 0..len {
                let r = next();
                let idx = if ascii_bias > 0 && r % 4 < ascii_bias {
                    (r >> 8) % 23
                } else {
                    (r >> 8) % pool.len() as u64
                };
                t.push_str(pool[idx as usize]);
            }
            for kind in [
                ScanKind::Cl100k,
                ScanKind::Qwen,
                ScanKind::Qwen35,
                ScanKind::O200k,
                ScanKind::Tekken,
                ScanKind::Kimi,
                ScanKind::DeepSeek,
                ScanKind::Gpt2,
            ] {
                assert_eq!(
                    collect(kind, &t, true),
                    collect(kind, &t, false),
                    "round {round} {kind:?} {t:?}"
                );
            }
        }
    }

    /// o200k over text whose non-ASCII chars all have ASCII stand-ins (no char in
    /// both case classes), so every piece takes the transcoded bulk path: case
    /// splits across upper/lower/titlecase non-ASCII letters, contractions next to
    /// them, multibyte digits / whitespace / punctuation, `/` trailing.
    #[test]
    fn o200k_transcoded_matches_scalar() {
        use crate::pre_tokenizers::scan::{ScanKind, scan_core};
        let collect = |t: &str, fast: bool| {
            let mut v = Vec::new();
            let f = |a: usize, b: usize| {
                v.push((a, b));
                Ok(())
            };
            if fast {
                scan_fast(ScanKind::O200k, t, f).unwrap()
            } else {
                scan_core(ScanKind::O200k, t, f).unwrap()
            }
            v
        };
        let pool: &[&str] = &[
            "a", "b", "Z", "Q", "s", "t", "'", "'", "0", "9", " ", " ", "\t", "\n", "\r", "/", "!",
            ".", "é", "É", "ß", "ǅ", "Ω", "ω", "Ж", "ж", "١", "１", "²", "½", "\u{00a0}",
            "\u{3000}", "\u{2028}", "’", "“", "—", "…", "€", "😀", "\u{200d}", "\u{feff}", "𝐀",
            "𝐚", "𝟎",
        ];
        let mut state = 0xbb67_ae85_84ca_a73bu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..6000 {
            let len = 1 + (next() % 400) as usize;
            let mut t = String::new();
            for _ in 0..len {
                t.push_str(pool[(next() % pool.len() as u64) as usize]);
            }
            assert_eq!(collect(&t, true), collect(&t, false), "round {round} {t:?}");
        }
    }

    /// The whole-text start bitmap (`bulk_starts`, which the fused encode loop
    /// walks) yields exactly `scan_fast`'s spans for every bulk grammar, over long
    /// texts with sparse-to-dense non-ASCII (transcoded for cl100k / Qwen /
    /// DeepSeek; mixed o200k has no whole-text bitmap).
    #[test]
    fn o200k_family_rep_table_matches_classes() {
        let t = super::super::unicode_class::tables();
        for kimi in [false, true] {
            for cp in (0x80..0x11_0000u32).filter(|c| !(0xD800..0xE000).contains(c)) {
                let direct = if kimi && t.is_han(cp) {
                    t.is_letter(cp).then_some(0x80)
                } else if o200k_hard(t, cp, kimi) {
                    Some(if t.is_letter(cp) { REP_BOTH } else { REP_MARK })
                } else if kimi {
                    Some(kimi_rep(t, cp))
                } else {
                    Some(o200k_easy_rep(t, cp))
                };
                assert_eq!(
                    o200k_family_rep(t, cp, kimi),
                    direct,
                    "U+{cp:04X} kimi={kimi}"
                );
                assert_ne!(direct, Some(0xFF), "0xFF is the hard marker");
            }
        }
    }

    #[test]
    fn cl100k_rep_table_matches_classes() {
        let t = super::super::unicode_class::tables();
        for cp in (0x80..0x11_0000u32).filter(|c| !(0xD800..0xE000).contains(c)) {
            assert_eq!(cl100k_rep_fast(t, cp), cl100k_rep(t, cp), "U+{cp:04X}");
        }
    }

    #[test]
    fn deepseek_rep_table_matches_classes() {
        let t = super::super::unicode_class::tables();
        for cp in (0x80..0x11_0000u32).filter(|c| !(0xD800..0xE000).contains(c)) {
            assert_eq!(deepseek_rep_fast(t, cp), deepseek_rep(t, cp), "U+{cp:04X}");
        }
    }

    #[test]
    fn bulk_starts_matches_scan_fast() {
        use crate::pre_tokenizers::scan::ScanKind;
        let pool: &[&str] = &[
            " the",
            " quick",
            " Brown",
            "fox",
            ".",
            ",",
            " 12345",
            "\n",
            "\n\n",
            "  ",
            "\t",
            " don't",
            "'s",
            " /usr/bin",
            "!\n/",
            "HTTPRequest",
            " camelCase",
            "\r\n",
            "é",
            "É",
            "’",
            "“",
            "—",
            "中文",
            "日本",
            "ー",
            "e\u{0301}",
            "ʰ",
            "😀",
            "\u{00a0}",
            "\u{3000}",
            "１２",
            "ß",
            "ǅ",
            " naïve",
            "…",
            "\u{200d}",
            "゠",
            "\x01",
        ];
        let mut state = 0x3c6e_f372_fe94_f82bu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..1500 {
            let target = 1 + (next() % 3000) as usize;
            let every = 1 + (next() % 200) as usize;
            let mut t = String::new();
            let mut k = 0;
            while t.len() < target {
                let i = if k % every == every - 1 {
                    next() % pool.len() as u64
                } else {
                    next() % 18
                };
                t.push_str(pool[i as usize]);
                k += 1;
            }
            for kind in [
                ScanKind::Cl100k,
                ScanKind::Qwen,
                ScanKind::Qwen35,
                ScanKind::O200k,
                ScanKind::Tekken,
                ScanKind::Kimi,
                ScanKind::DeepSeek,
                ScanKind::Gpt2,
            ] {
                let mut want = Vec::new();
                scan_fast(kind, &t, |a, b| {
                    want.push((a, b));
                    Ok(())
                })
                .unwrap();
                let Some(bs) = bulk_starts(kind, &t) else {
                    assert!(
                        matches!(kind, ScanKind::O200k | ScanKind::Tekken | ScanKind::Kimi)
                            && !t.is_ascii(),
                        "{kind:?} has a bulk pass"
                    );
                    continue;
                };
                let got: Vec<(usize, usize)> = bs.spans(t.len()).collect();
                assert_eq!(got, want, "round {round} {kind:?} {t:?}");
            }
        }
    }

    /// The o200k family's bulk pass over letters in both case classes (`\p{Lo}`
    /// scripts, `\p{Lm}`, marks) is `scan_fast`'s spans — on script-only text,
    /// where it must apply, and mixed with Latin case runs, contractions and stray
    /// marks, where it may hand the text back.
    #[test]
    fn o200k_both_case_bulk_matches_scan_fast() {
        use crate::pre_tokenizers::scan::ScanKind;
        let script: &[&str] = &[
            " שלום",
            "עולם",
            " מה",
            " मौसम",
            " हिंदी",
            "क्षत्र",
            " مرحبا",
            " بِسْمِ",
            "ـ",
            " 한국어",
            "를",
            " カタカナ",
            "ひらがな",
            "ー",
            " 中文",
            "日本",
            " ",
            "  ",
            ".",
            "!",
            ",",
            "\n",
            "\n\n",
            " 12",
            "3456",
            " —",
            "…",
            "\t",
            "ʰ",
        ];
        let latin: &[&str] = &[
            "A",
            "B",
            "a",
            "b",
            "HTTP",
            "Request",
            " camel",
            "Case",
            "'s",
            "'S",
            "'t",
            "'ll",
            "é",
            "É",
            "ǅ",
            " Ωμέγα",
            " Привет",
            "\u{05B0}",
            "\u{0301}",
            " \u{0301}",
            "\u{200d}",
            "/",
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut bulk = 0usize;
        let rounds = 3000;
        for round in 0..rounds {
            let mixed = round % 2 == 1;
            let target = 1 + (next() % 1500) as usize;
            let mut t = String::new();
            while t.len() < target {
                let from_latin = mixed && next() % 3 == 0;
                let pool = if from_latin { latin } else { script };
                t.push_str(pool[(next() % pool.len() as u64) as usize]);
            }
            for kind in [ScanKind::O200k, ScanKind::Tekken, ScanKind::Kimi] {
                let mut want = Vec::new();
                scan_fast(kind, &t, |a, b| {
                    want.push((a, b));
                    Ok(())
                })
                .unwrap();
                let Some(bs) = bulk_starts(kind, &t) else {
                    continue;
                };
                if !mixed {
                    bulk += 1;
                }
                let got: Vec<(usize, usize)> = bs.spans(t.len()).collect();
                assert_eq!(got, want, "round {round} {kind:?} {t:?}");
            }
        }
        // Script-only text takes the bulk pass.
        assert!(
            bulk * 4 > rounds / 2 * 3 * 3,
            "bulk pass on {bulk} script-only scans"
        );
    }

    /// Newline, whitespace and punct-trailing runs slid across 64-bit word edges —
    /// including runs longer than a word — exercise the carries in the word-parallel
    /// `punct_trailing_runs` / `newline_split` passes for both grammars.
    #[test]
    fn newline_runs_across_word_edges() {
        let mut units: Vec<String> = vec![
            "x!\n\n\n  \ny".into(),
            "a \n \n \t\n b".into(),
            "x.\n/\n/y".into(),
            "end.\n/usr/bin\nx".into(),
            "q)\r\n\r\n\r\n   z".into(),
            "w;  \n\n\t\t\n\n  v".into(),
        ];
        units.push(format!("p!{}z", "\n".repeat(150)));
        units.push(format!("p {}z", " \n".repeat(90)));
        units.push(format!("p.{}z", "\n/".repeat(80)));
        for pad in 40..140usize {
            for u in &units {
                let s = format!("{}{}{}", "ab ".repeat(pad / 3), &"xyz"[..pad % 3], u);
                assert_eq!(bulk_cl100k_spans(&s), simd_cl100k_spans(&s), "cl100k {s:?}");
                assert_eq!(bulk_o200k_spans(&s), scalar_o200k_spans(&s), "o200k {s:?}");
            }
        }
    }

    #[test]
    fn o200k_bulk_matches_scalar() {
        let corpus: Vec<String> = vec![
            "".into(),
            "hello world".into(),
            // case-splitting: the o200k word rule `[A-Z]*[a-z]+ | [A-Z]+[a-z]*`.
            "HTTPRequest HelloWorld camelCase ALLCAPS iOS aBc getHTTPResponseCode".into(),
            "XMLHttpRequest parseJSON toURL IPv4Address a1B2c3 ABCdef ABcDe".into(),
            // contraction-as-suffix (glued): don't stays whole; O'Brien splits.
            "don't I'll they're we've I'm he'd can't O'Brien 'Store rock'n'roll".into(),
            "DON'T Isn'T it's Y'all'd've a'b a''b 'tis 's 't 're".into(),
            "a.b a..b a. b _foo .NET /usr/bin/env x!b (hello) [tag] {x} a!!!b".into(),
            // o200k punct trailing `[\r\n/]*`: `/` continues a run past a newline.
            "end.\n/usr/bin\nx!\n/\n/y path//to\n//c".into(),
            "1234567 3.14 1,000 42 007 v1.2.3 100000000 a1b2c3".into(),
            "  leading   trailing  a  b\ta\t\tb end.\n\nfoo\r\nbar\n\n  x  \n  y".into(),
            "def fooBar(x: int) -> int:\n    return x*2 + 1  # doubles\n".into(),
            "{\"key\": \"valueHere\", \"n\": 42, \"arr\": [1, 2, 3]}".into(),
            "\t\ttabs   spaces\n\n\n\nnewLines\r\r\rReturns \n \t\n Mixed".into(),
            "SELECT idHere FROM tName WHERE a>=18 AND s!='x'; -- Comment".into(),
            "The rain in Spain. ".repeat(40),
        ];
        for s in &corpus {
            assert_eq!(bulk_o200k_spans(s), scalar_o200k_spans(s), "corpus {s:?}");
        }

        // Case-split and contraction slid across the 64-byte word boundary.
        for pad in 40..96usize {
            for lit in [
                "HelloWorld",
                "aB",
                "ABCdef",
                "don't",
                "I'll",
                "'s",
                "x'Ll",
                "O'Brien",
            ] {
                let mut s = "x".repeat(pad);
                s.push(' ');
                s.push_str(lit);
                s.push_str("Yz");
                assert_eq!(
                    bulk_o200k_spans(&s),
                    scalar_o200k_spans(&s),
                    "boundary {s:?}"
                );
            }
        }

        // ASCII fuzz, mixed case + apostrophes + slashes (all o200k classes).
        let alpha = b" \t\n\r'/.!_,-abcABCXYZ01()\"; \n's't're've'll dEeRrVvLl";
        let mut state = 0x1234_5678_9abc_def0u64;
        for len in 0..=200usize {
            for _ in 0..150 {
                let mut buf = Vec::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    buf.push(alpha[(state >> 11) as usize % alpha.len()]);
                }
                let s = std::str::from_utf8(&buf).unwrap();
                assert_eq!(bulk_o200k_spans(s), scalar_o200k_spans(s), "fuzz {s:?}");
            }
        }
    }

    #[test]
    fn cl100k_bulk_matches_oracle() {
        let corpus: Vec<String> = vec![
            "".into(),
            "hello world".into(),
            "The quick brown fox. You think it is O'Brien's?".into(),
            "HTTPRequest HelloWorld camelCase ALLCAPS iOS getHTTPResponseCode()".into(),
            "a.b a..b a. b _foo .NET /usr/bin/env x!b (hello) [tag] {x} a!!!b".into(),
            "1234567 3.14 1,000 42 007 v1.2.3 100000000 a1b2c3".into(),
            "  leading   trailing  a  b\ta\t\tb end.\n\nfoo\r\nbar\n\n  x  \n  y".into(),
            "def f(x: int) -> int:\n    return x*2 + 1  # doubles\n".into(),
            "{\"key\": \"value\", \"n\": 42, \"arr\": [1, 2, 3]}".into(),
            "\t\ttabs   spaces\n\n\n\nnewlines\r\r\rreturns \n \t\n mixed".into(),
            "SELECT id FROM t WHERE a>=18; -- comment".into(),
            "a\x00b\x07c\x1bd control e.g. i.e. U.S.A. snake_case-kebab 3   4  5".into(),
            "trailing spaces then eof   ".into(),
            "The rain in Spain. ".repeat(40),
            // Contractions `(?i:'s|'t|'re|'ve|'m|'ll|'d)`: as their own token, cutting
            // the letter run behind them, chained, uppercased, and the non-contraction
            // apostrophe cases (word-prefix `'Brien`, punct `''`, `'5`).
            "don't I'll we've they're I'm he'd can't've y'all'd've".into(),
            "DON'T Isn'T O'Brien 'Store 'store'd it's'nt".into(),
            "'s 't 're 've 'm 'll 'd '' ''' 'x '5 a'b a''b ' '".into(),
            "'store 'reve 's's 'ss x's'store rock'n'roll".into(),
            "'".into(),
            "'s".into(),
            "trailing apostrophe'".into(),
        ];
        for s in &corpus {
            assert_eq!(bulk_cl100k_spans(s), simd_cl100k_spans(s), "corpus {s:?}");
        }

        // Contractions slid across the 64-byte word boundary: the apostrophe, and each
        // interior/boundary byte of the literal, lands at every offset around bit 63.
        for pad in 40..96usize {
            for lit in ["'s", "'re", "'ll", "'ve", "'d", "'store", "'s'd"] {
                let mut s = "x".repeat(pad);
                s.push_str("ab"); // a letter run so the apostrophe is a word-prefix start
                s.push_str(lit);
                s.push_str("yz");
                assert_eq!(bulk_cl100k_spans(&s), simd_cl100k_spans(&s), "contr {s:?}");
            }
        }

        // Digit `\p{N}{1,3}` cap across the 64-byte word boundary: a run of every
        // length, started at every offset around bit 63, so the `open`/`since` carry
        // is exercised at all three group phases. Padded with a letter so the run is
        // its own pre-token, and prefixed with fillers to slide it past the edge.
        for pad in 0..80usize {
            for runlen in 1..=15usize {
                let mut s = String::new();
                s.push('x');
                for _ in 0..pad {
                    s.push('x');
                }
                s.push(' ');
                for k in 0..runlen {
                    s.push((b'0' + (k % 10) as u8) as char);
                }
                s.push('y');
                assert_eq!(bulk_cl100k_spans(&s), simd_cl100k_spans(&s), "digit {s:?}");
            }
        }

        // ASCII fuzz, apostrophes included (contraction letters s/t/r/v/l/m/d/e are in
        // the alphabet, so contractions form and chain at random).
        let alpha = b" \t\n\r.!/_,-abcABC01()\"; \n's't're've'll'md e r v l";
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for len in 0..=200usize {
            for _ in 0..200 {
                let mut buf = Vec::with_capacity(len);
                for _ in 0..len {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    buf.push(alpha[(state >> 11) as usize % alpha.len()]);
                }
                let s = std::str::from_utf8(&buf).unwrap();
                assert_eq!(bulk_cl100k_spans(s), simd_cl100k_spans(s), "fuzz {s:?}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_swar() {
        let mut state = 0xdead_beef_cafe_babeu64;
        for len in 0..=130usize {
            let mut buf = Vec::with_capacity(len);
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let b = if state & 3 == 0 {
                    (state >> 8) as u8
                } else {
                    b" \t\n\rAZaz09!'/_.,:;@#"[(state >> 8) as usize % 20]
                };
                buf.push(b);
            }
            for off in [0usize, 1, 7, 8, 63] {
                if off > buf.len() {
                    continue;
                }
                let neon = unsafe { classify_block_neon(&buf, off) };
                let swar = classify_block_swar(&buf, off);
                assert_eq!(neon, swar, "len {len} off {off}");
            }
        }
    }

    /// Every AVX2 kernel is bit-identical to its portable counterpart, over all
    /// byte values (weighted toward the class boundaries and DeepSeek's stand-in
    /// bytes), every short-block length and several offsets.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_portable() {
        if !avx2::available() {
            return;
        }
        let mut state = 0x0123_4567_89ab_cdefu64;
        for len in 0..=200usize {
            let mut buf = Vec::with_capacity(len);
            for _ in 0..len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let b = match state & 3 {
                    0 => (state >> 8) as u8,
                    1 => [
                        0x00, 0x08, 0x09, 0x0D, 0x0E, 0x1F, 0x20, 0x7F, 0x80, 0x81, 0x82, 0x83,
                        0x84, 0x85, 0xFF,
                    ][(state >> 8) as usize % 15],
                    _ => b" \t\n\rAZaz@[`{09/:'!_.,;#"[(state >> 8) as usize % 23],
                };
                buf.push(b);
            }
            for off in [0usize, 1, 7, 8, 31, 32, 63, 64, 65, 128] {
                if off > buf.len() {
                    continue;
                }
                let ctx = format!("len {len} off {off}");
                unsafe {
                    assert_eq!(
                        avx2::classify_block(&buf, off),
                        classify_block_swar(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx2::classify_block_bulk::<false, false>(&buf, off),
                        classify_block_bulk_portable::<false, false>(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx2::classify_block_bulk::<true, false>(&buf, off),
                        classify_block_bulk_portable::<true, false>(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx2::classify_block_bulk::<true, true>(&buf, off),
                        classify_block_bulk_portable::<true, true>(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx2::classify_block_bulk::<false, true>(&buf, off),
                        classify_block_bulk_portable::<false, true>(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx2::classify_block_ds::<false>(&buf, off),
                        classify_block_ds_scalar::<false>(&buf, off),
                        "{ctx}"
                    );
                    assert_eq!(
                        avx2::classify_block_ds::<true>(&buf, off),
                        classify_block_ds_scalar::<true>(&buf, off),
                        "{ctx}"
                    );
                }
            }
        }
    }

    #[test]
    fn movemask_ordering() {
        // byte 0 high bit -> bit 0, byte 7 -> bit 7.
        assert_eq!(movemask8(0x0000_0000_0000_0080), 0b0000_0001);
        assert_eq!(movemask8(0x8000_0000_0000_0000), 0b1000_0000);
        assert_eq!(movemask8(HI), 0xFF);
        assert_eq!(movemask8(0), 0);
    }
}
