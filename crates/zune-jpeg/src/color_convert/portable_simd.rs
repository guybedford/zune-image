/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Portable SIMD YCbCr to RGB conversion (`core::simd`, nightly).
//!
//! Sixteen pixels per call: the arithmetic is done in `i32x16`, narrowed to
//! `u8x16` planes, then interleaved with swizzles. Bit identical to the
//! scalar routines.

use core::simd::prelude::*;

use crate::color_convert::scalar::{
    CB_CF, CR_CF, C_G_CB_COEF_2, C_G_CR_COEF_1, YUV_PREC, YUV_RND, Y_CF
};

type Wide = i32x16;

#[inline]
fn coeff(c: i16) -> Wide {
    Wide::splat(i32::from(c))
}

#[inline]
fn narrow(v: Wide) -> u8x16 {
    (v >> (YUV_PREC as i32))
        .simd_clamp(Wide::splat(0), Wide::splat(255))
        .cast::<u8>()
}

/// Convert sixteen samples into planar R, G and B.
#[inline]
fn ycbcr_to_rgb_planes(y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16]) -> (u8x16, u8x16, u8x16) {
    let y = i16x16::from_array(*y).cast::<i32>();
    let cb = i16x16::from_array(*cb).cast::<i32>() - Wide::splat(128);
    let cr = i16x16::from_array(*cr).cast::<i32>() - Wide::splat(128);

    let y0 = y * coeff(Y_CF) + coeff(YUV_RND);

    let r = y0 + cr * coeff(CR_CF);
    let g = y0 + cr * coeff(C_G_CR_COEF_1) + cb * coeff(C_G_CB_COEF_2);
    let b = y0 + cb * coeff(CB_CF);

    (narrow(r), narrow(g), narrow(b))
}

/// Interleave three planes into 48 bytes of `abcabc...`.
///
/// Two 16-lane shuffles per output vector: `a` and `b` are first placed in
/// their final slots, then `c` fills the gaps. Wider intermediates make LLVM
/// scalarise the preceding narrowing casts on wasm32.
#[inline]
fn store3(out: &mut [u8; 48], a: u8x16, b: u8x16, c: u8x16) {
    let t0: u8x16 = simd_swizzle!(a, b, [0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5]);
    let o0: u8x16 = simd_swizzle!(t0, c, [0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15]);

    let t1: u8x16 = simd_swizzle!(a, b, [21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26]);
    let o1: u8x16 = simd_swizzle!(t1, c, [0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15]);

    let t2: u8x16 = simd_swizzle!(a, b, [0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0]);
    let o2: u8x16 = simd_swizzle!(t2, c, [26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31]);

    o0.copy_to_slice(&mut out[0..16]);
    o1.copy_to_slice(&mut out[16..32]);
    o2.copy_to_slice(&mut out[32..48]);
}

/// Interleave four planes into 64 bytes of `abcdabcd...`.
#[inline]
fn store4(out: &mut [u8; 64], a: u8x16, b: u8x16, c: u8x16, d: u8x16) {
    let (ab_lo, ab_hi) = a.interleave(b);
    let (cd_lo, cd_hi) = c.interleave(d);
    // ab and cd now hold byte pairs; interleave them as 16-bit units
    let (o0, o1) = pairs(ab_lo, cd_lo);
    let (o2, o3) = pairs(ab_hi, cd_hi);

    o0.copy_to_slice(&mut out[0..16]);
    o1.copy_to_slice(&mut out[16..32]);
    o2.copy_to_slice(&mut out[32..48]);
    o3.copy_to_slice(&mut out[48..64]);
}

/// Interleave `x` and `y` two bytes at a time.
#[inline]
fn pairs(x: u8x16, y: u8x16) -> (u8x16, u8x16) {
    (
        simd_swizzle!(
            x,
            y,
            [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23]
        ),
        simd_swizzle!(
            x,
            y,
            [8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31]
        )
    )
}

#[inline]
fn out3(out: &mut [u8], offset: usize) -> &mut [u8; 48] {
    out.get_mut(offset..offset + 48)
        .expect("Slice to small cannot write")
        .try_into()
        .unwrap()
}

#[inline]
fn out4(out: &mut [u8], offset: usize) -> &mut [u8; 64] {
    out.get_mut(offset..offset + 64)
        .expect("Slice to small cannot write")
        .try_into()
        .unwrap()
}

pub fn ycbcr_to_rgb_simd(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let (r, g, b) = ycbcr_to_rgb_planes(y, cb, cr);
    store3(out3(out, *offset), r, g, b);
    *offset += 48;
}

pub fn ycbcr_to_bgr_simd(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let (r, g, b) = ycbcr_to_rgb_planes(y, cb, cr);
    store3(out3(out, *offset), b, g, r);
    *offset += 48;
}

pub fn ycbcr_to_rgba_simd(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let (r, g, b) = ycbcr_to_rgb_planes(y, cb, cr);
    store4(out4(out, *offset), r, g, b, u8x16::splat(255));
    *offset += 64;
}

pub fn ycbcr_to_bgra_simd(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let (r, g, b) = ycbcr_to_rgb_planes(y, cb, cr);
    store4(out4(out, *offset), b, g, r, u8x16::splat(255));
    *offset += 64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_convert::scalar::{
        ycbcr_to_rgb_inner_16_scalar, ycbcr_to_rgba_inner_16_scalar
    };
    use crate::decoder::ColorConvert16Ptr;

    fn samples(seed: i16) -> ([i16; 16], [i16; 16], [i16; 16]) {
        let gen = |k: i16| core::array::from_fn(|i| (i as i16 * 37 + seed * k) & 0xff);
        (gen(1), gen(3), gen(7))
    }

    fn check(scalar: ColorConvert16Ptr, fast: ColorConvert16Ptr, stride: usize) {
        for seed in 0..64 {
            let (y, cb, cr) = samples(seed);
            let mut expected = vec![0u8; stride + 8];
            let mut actual = vec![0u8; stride + 8];
            let (mut pe, mut pa) = (4, 4);
            scalar(&y, &cb, &cr, &mut expected, &mut pe);
            fast(&y, &cb, &cr, &mut actual, &mut pa);
            assert_eq!(pe, pa);
            assert_eq!(expected, actual, "seed {seed}");
        }
    }

    #[test]
    fn rgb_matches_scalar() {
        check(ycbcr_to_rgb_inner_16_scalar::<false>, ycbcr_to_rgb_simd, 48);
        check(ycbcr_to_rgb_inner_16_scalar::<true>, ycbcr_to_bgr_simd, 48);
    }

    #[test]
    fn rgba_matches_scalar() {
        check(
            ycbcr_to_rgba_inner_16_scalar::<false>,
            ycbcr_to_rgba_simd,
            64
        );
        check(
            ycbcr_to_rgba_inner_16_scalar::<true>,
            ycbcr_to_bgra_simd,
            64
        );
    }

    #[test]
    fn saturates() {
        for (y, cb, cr) in [
            ([255i16; 16], [0i16; 16], [255i16; 16]),
            ([0; 16], [255; 16], [0; 16])
        ] {
            let mut expected = [0u8; 48];
            let mut actual = [0u8; 48];
            ycbcr_to_rgb_inner_16_scalar::<false>(&y, &cb, &cr, &mut expected, &mut 0);
            ycbcr_to_rgb_simd(&y, &cb, &cr, &mut actual, &mut 0);
            assert_eq!(expected, actual);
        }
    }
}
