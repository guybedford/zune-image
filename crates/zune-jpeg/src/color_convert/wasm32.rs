/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! WebAssembly simd128 color conversion routines.
//!
//! Sixteen pixels per call, following the NEON version: widening multiplies
//! into `i32` accumulators, a saturating narrow back to `u8`, then byte
//! shuffles to interleave the planes since there is no `vst3q`/`vst4q`.

#![cfg(all(
    feature = "simd128",
    target_arch = "wasm32",
    target_feature = "simd128"
))]

use core::arch::wasm32::*;

use crate::color_convert::scalar::{
    CB_CF, CR_CF, C_G_CB_COEF_2, C_G_CR_COEF_1, YUV_PREC, YUV_RND, Y_CF
};

/// Widening multiply-accumulate of eight `i16` lanes by a constant into
/// two `i32x4` accumulators.
#[inline]
fn mla(acc: (v128, v128), a: v128, coeff: i16) -> (v128, v128) {
    let c = i16x8_splat(coeff);
    (
        i32x4_add(acc.0, i32x4_extmul_low_i16x8(a, c)),
        i32x4_add(acc.1, i32x4_extmul_high_i16x8(a, c))
    )
}

/// `>> YUV_PREC` then saturate eight `i32` lanes to `u16`.
#[inline]
fn shift_narrow(a: (v128, v128)) -> v128 {
    u16x8_narrow_i32x4(
        i32x4_shr(a.0, YUV_PREC as u32),
        i32x4_shr(a.1, YUV_PREC as u32)
    )
}

/// Convert sixteen YCbCr samples into planar `u8x16` R, G and B vectors.
#[inline]
fn ycbcr_to_rgb_planes(y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16]) -> (v128, v128, v128) {
    // SAFETY: each input holds 16 i16 (32 bytes); both loads stay in bounds.
    let (y0, y1, cb0, cb1, cr0, cr1) = unsafe {
        (
            v128_load(y.as_ptr().cast()),
            v128_load(y[8..].as_ptr().cast()),
            v128_load(cb.as_ptr().cast()),
            v128_load(cb[8..].as_ptr().cast()),
            v128_load(cr.as_ptr().cast()),
            v128_load(cr[8..].as_ptr().cast())
        )
    };

    let bias = i16x8_splat(128);
    let cb0 = i16x8_sub(cb0, bias);
    let cb1 = i16x8_sub(cb1, bias);
    let cr0 = i16x8_sub(cr0, bias);
    let cr1 = i16x8_sub(cr1, bias);

    let rnd = i32x4_splat(i32::from(YUV_RND));
    let acc0 = mla((rnd, rnd), y0, Y_CF);
    let acc1 = mla((rnd, rnd), y1, Y_CF);

    let r0 = shift_narrow(mla(acc0, cr0, CR_CF));
    let r1 = shift_narrow(mla(acc1, cr1, CR_CF));

    let b0 = shift_narrow(mla(acc0, cb0, CB_CF));
    let b1 = shift_narrow(mla(acc1, cb1, CB_CF));

    let g0 = shift_narrow(mla(mla(acc0, cb0, C_G_CB_COEF_2), cr0, C_G_CR_COEF_1));
    let g1 = shift_narrow(mla(mla(acc1, cb1, C_G_CB_COEF_2), cr1, C_G_CR_COEF_1));

    // The u16 lanes are at most a few hundred, so the i16 view saturates to u8 correctly.
    (
        u8x16_narrow_i16x8(r0, r1),
        u8x16_narrow_i16x8(g0, g1),
        u8x16_narrow_i16x8(b0, b1)
    )
}

/// Interleave three planes into 48 bytes of `abcabc...`.
#[inline]
fn store3(out: &mut [u8; 48], a: v128, b: v128, c: v128) {
    // Pull a and b into their final slots, then drop c into the gaps.
    let t0 = i8x16_shuffle::<0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5>(a, b);
    let o0 = i8x16_shuffle::<0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15>(t0, c);

    let t1 = i8x16_shuffle::<21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26>(a, b);
    let o1 = i8x16_shuffle::<0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15>(t1, c);

    let t2 = i8x16_shuffle::<0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0>(a, b);
    let o2 = i8x16_shuffle::<26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31>(t2, c);

    // SAFETY: `out` is exactly 48 bytes.
    unsafe {
        v128_store(out.as_mut_ptr().cast(), o0);
        v128_store(out.as_mut_ptr().add(16).cast(), o1);
        v128_store(out.as_mut_ptr().add(32).cast(), o2);
    }
}

/// Interleave four planes into 64 bytes of `abcdabcd...`.
#[inline]
fn store4(out: &mut [u8; 64], a: v128, b: v128, c: v128, d: v128) {
    let ab_lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b);
    let ab_hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b);
    let cd_lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(c, d);
    let cd_hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(c, d);

    let o0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(ab_lo, cd_lo);
    let o1 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(ab_lo, cd_lo);
    let o2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(ab_hi, cd_hi);
    let o3 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(ab_hi, cd_hi);

    // SAFETY: `out` is exactly 64 bytes.
    unsafe {
        v128_store(out.as_mut_ptr().cast(), o0);
        v128_store(out.as_mut_ptr().add(16).cast(), o1);
        v128_store(out.as_mut_ptr().add(32).cast(), o2);
        v128_store(out.as_mut_ptr().add(48).cast(), o3);
    }
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

pub fn ycbcr_to_rgb_simd128(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let (r, g, b) = ycbcr_to_rgb_planes(y, cb, cr);
    store3(out3(out, *offset), r, g, b);
    *offset += 48;
}

pub fn ycbcr_to_bgr_simd128(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let (r, g, b) = ycbcr_to_rgb_planes(y, cb, cr);
    store3(out3(out, *offset), b, g, r);
    *offset += 48;
}

pub fn ycbcr_to_rgba_simd128(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let (r, g, b) = ycbcr_to_rgb_planes(y, cb, cr);
    store4(out4(out, *offset), r, g, b, u8x16_splat(255));
    *offset += 64;
}

pub fn ycbcr_to_bgra_simd128(
    y: &[i16; 16], cb: &[i16; 16], cr: &[i16; 16], out: &mut [u8], offset: &mut usize
) {
    let (r, g, b) = ycbcr_to_rgb_planes(y, cb, cr);
    store4(out4(out, *offset), b, g, r, u8x16_splat(255));
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
        check(
            ycbcr_to_rgb_inner_16_scalar::<false>,
            ycbcr_to_rgb_simd128,
            48
        );
        check(
            ycbcr_to_rgb_inner_16_scalar::<true>,
            ycbcr_to_bgr_simd128,
            48
        );
    }

    #[test]
    fn rgba_matches_scalar() {
        check(
            ycbcr_to_rgba_inner_16_scalar::<false>,
            ycbcr_to_rgba_simd128,
            64
        );
        check(
            ycbcr_to_rgba_inner_16_scalar::<true>,
            ycbcr_to_bgra_simd128,
            64
        );
    }

    #[test]
    fn saturates() {
        let y = [255i16; 16];
        let cb = [0i16; 16];
        let cr = [255i16; 16];
        let mut expected = [0u8; 48];
        let mut actual = [0u8; 48];
        ycbcr_to_rgb_inner_16_scalar::<false>(&y, &cb, &cr, &mut expected, &mut 0);
        ycbcr_to_rgb_simd128(&y, &cb, &cr, &mut actual, &mut 0);
        assert_eq!(expected, actual);
    }
}
