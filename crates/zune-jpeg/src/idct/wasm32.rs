/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! WebAssembly simd128 IDCT.
//!
//! A direct translation of the NEON implementation: eight `i32` lanes per
//! `YmmRegister` as a pair of `v128`s, two 1-D passes with an in-register
//! transpose between them, bit identical to the scalar code.

#![cfg(all(
    feature = "simd128",
    target_arch = "wasm32",
    target_feature = "simd128"
))]

use core::arch::wasm32::*;

use crate::unsafe_utils::{transpose, YmmRegister};

const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

/// Saturate eight `i32` lanes to `i16` and clamp to `0..=255`.
#[inline]
fn pack_clamp(a: (v128, v128)) -> v128 {
    let packed = i16x8_narrow_i32x4(a.0, a.1);
    i16x8_min(i16x8_max(packed, i16x8_splat(0)), i16x8_splat(255))
}

/// Store eight `i16` values at `out[pos..pos + 8]`, into a scratch buffer if
/// the slice is too short so malformed images cannot write out of bounds.
#[inline]
fn store8(out: &mut [i16], pos: usize, value: v128) {
    let mut tmp = [0i16; 8];
    let dst: &mut [i16; 8] = out
        .get_mut(pos..pos + 8)
        .map(|s| s.try_into().unwrap())
        .unwrap_or(&mut tmp);
    // SAFETY: `dst` is exactly eight i16, 16 bytes; v128_store is unaligned.
    unsafe { v128_store(dst.as_mut_ptr().cast(), value) };
}

#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::op_ref,
    unused_assignments,
    clippy::zero_prefixed_literal
)]
pub fn idct_simd128(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    let mut pos = 0;

    // SAFETY: `in_vector` holds 64 i32, every load below stays within it.
    let (mut row0, mut row1, mut row2, mut row3, mut row4, mut row5, mut row6, mut row7, row8) = unsafe {
        (
            YmmRegister::load(in_vector[00..].as_ptr()),
            YmmRegister::load(in_vector[08..].as_ptr()),
            YmmRegister::load(in_vector[16..].as_ptr()),
            YmmRegister::load(in_vector[24..].as_ptr()),
            YmmRegister::load(in_vector[32..].as_ptr()),
            YmmRegister::load(in_vector[40..].as_ptr()),
            YmmRegister::load(in_vector[48..].as_ptr()),
            YmmRegister::load(in_vector[56..].as_ptr()),
            // First row shifted by one so the DC term is excluded from the AC test.
            YmmRegister::load(in_vector[1..].as_ptr())
        )
    };

    // When every AC term is zero the block is flat: (dc / 8) + 128, clamped.
    let or_tree = ((row1 | row8) | (row2 | row3)) | ((row4 | row5) | (row6 | row7));

    if or_tree.all_zero() {
        let coeff = ((in_vector[0] + 4 + 1024) >> 3).clamp(0, 255) as i16;
        let idct_value = i16x8_splat(coeff);
        for _ in 0..8 {
            store8(out_vector, pos, idct_value);
            pos += stride;
        }
        return;
    }

    macro_rules! dct_pass {
        ($SCALE_BITS:tt,$scale:tt) => {
            // even part
            let p1 = (row2 + row6) * 2217;

            let mut t2 = p1 + row6 * -7567;
            let mut t3 = p1 + row2 * 3135;

            let mut t0 = (row0 + row4).const_shl::<12>();
            let mut t1 = (row0 - row4).const_shl::<12>();

            let x0 = t0 + t3 + $SCALE_BITS;
            let x3 = t0 - t3 + $SCALE_BITS;
            let x1 = t1 + t2 + $SCALE_BITS;
            let x2 = t1 - t2 + $SCALE_BITS;

            let p3 = row7 + row3;
            let p4 = row5 + row1;
            let p1 = row7 + row1;
            let p2 = row5 + row3;
            let p5 = (p3 + p4) * 4816;

            t0 = row7 * 1223;
            t1 = row5 * 8410;
            t2 = row3 * 12586;
            t3 = row1 * 6149;

            let p1 = p5 + p1 * -3685;
            let p2 = p5 + (p2 * -10497);
            let p3 = p3 * -8034;
            let p4 = p4 * -1597;

            t3 += p1 + p4;
            t2 += p2 + p3;
            t1 += p2 + p4;
            t0 += p1 + p3;

            row0 = (x0 + t3).const_shra::<$scale>();
            row1 = (x1 + t2).const_shra::<$scale>();
            row2 = (x2 + t1).const_shra::<$scale>();
            row3 = (x3 + t0).const_shra::<$scale>();

            row4 = (x3 - t0).const_shra::<$scale>();
            row5 = (x2 - t1).const_shra::<$scale>();
            row6 = (x1 - t2).const_shra::<$scale>();
            row7 = (x0 - t3).const_shra::<$scale>();
        };
    }

    // rows
    dct_pass!(512, 10);
    transpose(
        &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6, &mut row7
    );

    // columns
    dct_pass!(SCALE_BITS, 17);
    transpose(
        &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6, &mut row7
    );

    for row in [row0, row1, row2, row3, row4, row5, row6, row7] {
        store8(out_vector, pos, pack_clamp(row.mm256));
        pos += stride;
    }
}

/// IDCT for blocks whose coefficients outside the top-left 4x4 are all zero.
///
/// Translation of the AVX2 version: the zero rows and columns drop out of the
/// butterflies, so each pass only reads four inputs.
#[allow(clippy::similar_names, clippy::zero_prefixed_literal)]
pub fn idct_simd128_4x4(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    // SAFETY: `in_vector` holds 64 i32, the loads read rows 0..4.
    let (rw0, rw1, rw2, rw3) = unsafe {
        (
            YmmRegister::load(in_vector[00..].as_ptr()),
            YmmRegister::load(in_vector[08..].as_ptr()),
            YmmRegister::load(in_vector[16..].as_ptr()),
            YmmRegister::load(in_vector[24..].as_ptr())
        )
    };

    let (mut row0, mut row1, mut row2, mut row3) = (rw0, rw1, rw2, rw3);
    let (mut row4, mut row5, mut row6, mut row7);

    {
        row0 = row0.const_shl::<12>() + 512;

        let i2 = row2;

        let p1 = i2 * 2217;
        let p3 = i2 * 5352;

        let x0 = row0 + p3;
        let x1 = row0 + p1;
        let x2 = row0 - p1;
        let x3 = row0 - p3;

        let i4 = row3;
        let i3 = row1;

        let p5 = (i4 + i3) * 4816;

        let p1 = p5 + i3 * -3685;
        let p2 = p5 + i4 * -10497;

        let t3 = p5 + i3 * 867;
        let t2 = p5 + i4 * -5945;

        let t1 = p2 + i3 * -1597;
        let t0 = p1 + i4 * -8034;

        row0 = (x0 + t3).const_shra::<10>();
        row1 = (x1 + t2).const_shra::<10>();
        row2 = (x2 + t1).const_shra::<10>();
        row3 = (x3 + t0).const_shra::<10>();

        row4 = (x3 - t0).const_shra::<10>();
        row5 = (x2 - t1).const_shra::<10>();
        row6 = (x1 - t2).const_shra::<10>();
        row7 = (x0 - t3).const_shra::<10>();
    }

    transpose(
        &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6, &mut row7
    );

    {
        let i2 = row2;
        let i0 = row0;

        let t0 = i0.const_shl::<12>() + SCALE_BITS;

        let t2 = i2 * 2217;
        let t3 = i2 * 5352;

        let x0 = t0 + t3;
        let x3 = t0 - t3;
        let x1 = t0 + t2;
        let x2 = t0 - t2;

        let i3 = row3;
        let i1 = row1;

        let p5 = (i3 + i1) * 4816;

        let p1 = p5 + i1 * -3685;
        let p2 = p5 + i3 * -10497;

        let t3 = p5 + i1 * 867;
        let t2 = p5 + i3 * -5945;

        let t1 = p2 + i1 * -1597;
        let t0 = p1 + i3 * -8034;

        row0 = (x0 + t3).const_shra::<17>();
        row1 = (x1 + t2).const_shra::<17>();
        row2 = (x2 + t1).const_shra::<17>();
        row3 = (x3 + t0).const_shra::<17>();
        row4 = (x3 - t0).const_shra::<17>();
        row5 = (x2 - t1).const_shra::<17>();
        row6 = (x1 - t2).const_shra::<17>();
        row7 = (x0 - t3).const_shra::<17>();
    }

    transpose(
        &mut row0, &mut row1, &mut row2, &mut row3, &mut row4, &mut row5, &mut row6, &mut row7
    );

    let mut pos = 0;
    for row in [row0, row1, row2, row3, row4, row5, row6, row7] {
        store8(out_vector, pos, pack_clamp(row.mm256));
        pos += stride;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idct::scalar::{idct4x4, idct_int};

    fn coefficients(seed: u32, limit: usize) -> [i32; 64] {
        let mut state = seed;
        core::array::from_fn(|i| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let row = i / 8;
            let col = i % 8;
            if row < limit && col < limit {
                ((state >> 16) as i32 % 2048) - 1024
            } else {
                0
            }
        })
    }

    fn check(fast: fn(&mut [i32; 64], &mut [i16], usize), limit: usize) {
        for seed in 0..64 {
            let coeffs = coefficients(seed, limit);
            let (mut a, mut b) = (coeffs, coeffs);
            let mut expected = [0i16; 64];
            let mut actual = [0i16; 64];
            idct_int(&mut a, &mut expected, 8);
            fast(&mut b, &mut actual, 8);
            assert_eq!(expected, actual, "seed {seed}");
        }
    }

    #[test]
    fn matches_scalar() {
        check(idct_simd128, 8);
    }

    #[test]
    fn flat_block_matches_scalar() {
        let mut expected = [0i16; 64];
        let mut actual = [0i16; 64];
        let mut a = [0i32; 64];
        a[0] = 300;
        let mut b = a;
        idct_int(&mut a, &mut expected, 8);
        idct_simd128(&mut b, &mut actual, 8);
        assert_eq!(expected, actual);
    }

    #[test]
    fn matches_scalar_4x4() {
        for seed in 0..64 {
            let coeffs = coefficients(seed, 4);
            let (mut a, mut b) = (coeffs, coeffs);
            let mut expected = [0i16; 64];
            let mut actual = [0i16; 64];
            idct4x4(&mut a, &mut expected, 8);
            idct_simd128_4x4(&mut b, &mut actual, 8);
            assert_eq!(expected, actual, "seed {seed}");
        }
    }
}
