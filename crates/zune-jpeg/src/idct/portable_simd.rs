/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! Portable SIMD IDCT (`core::simd`, nightly).
//!
//! The same two-pass algorithm as the AVX2 and NEON versions with one `i32x8`
//! per row and an in-register transpose between passes. Bit identical to the
//! scalar code. On wasm32 with simd128 an `i32x8` is a pair of `v128`s.

use core::simd::prelude::*;

type Row = i32x8;

const SCALE_BITS: i32 = 512 + 65536 + (128 << 17);

#[inline]
fn load(src: &[i32]) -> Row {
    Row::from_slice(src)
}

/// Clamp to `0..=255`, narrow to `i16` and store at `out[pos..pos + 8]`;
/// malformed images with a short output land in a scratch buffer instead.
#[inline]
fn store_clamped(out: &mut [i16], pos: usize, row: Row) {
    let mut tmp = [0i16; 8];
    let dst = out.get_mut(pos..pos + 8).unwrap_or(&mut tmp);
    row.simd_clamp(Row::splat(0), Row::splat(255))
        .cast::<i16>()
        .copy_to_slice(dst);
}

/// Transpose an 8x8 matrix held as eight rows.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
#[inline]
fn transpose(rows: &mut [Row; 8]) {
    let [r0, r1, r2, r3, r4, r5, r6, r7] = *rows;

    // 32-bit interleave of row pairs
    let (t0, t1) = r0.interleave(r1);
    let (t2, t3) = r2.interleave(r3);
    let (t4, t5) = r4.interleave(r5);
    let (t6, t7) = r6.interleave(r7);

    // 64-bit interleave: each u holds two columns of four rows
    let u0 = simd_swizzle!(t0, t2, [0, 1, 8, 9, 2, 3, 10, 11]);
    let u1 = simd_swizzle!(t0, t2, [4, 5, 12, 13, 6, 7, 14, 15]);
    let u2 = simd_swizzle!(t1, t3, [0, 1, 8, 9, 2, 3, 10, 11]);
    let u3 = simd_swizzle!(t1, t3, [4, 5, 12, 13, 6, 7, 14, 15]);
    let u4 = simd_swizzle!(t4, t6, [0, 1, 8, 9, 2, 3, 10, 11]);
    let u5 = simd_swizzle!(t4, t6, [4, 5, 12, 13, 6, 7, 14, 15]);
    let u6 = simd_swizzle!(t5, t7, [0, 1, 8, 9, 2, 3, 10, 11]);
    let u7 = simd_swizzle!(t5, t7, [4, 5, 12, 13, 6, 7, 14, 15]);

    // 128-bit interleave: join the upper and lower four rows of each column
    *rows = [
        simd_swizzle!(u0, u4, [0, 1, 2, 3, 8, 9, 10, 11]),
        simd_swizzle!(u0, u4, [4, 5, 6, 7, 12, 13, 14, 15]),
        simd_swizzle!(u1, u5, [0, 1, 2, 3, 8, 9, 10, 11]),
        simd_swizzle!(u1, u5, [4, 5, 6, 7, 12, 13, 14, 15]),
        simd_swizzle!(u2, u6, [0, 1, 2, 3, 8, 9, 10, 11]),
        simd_swizzle!(u2, u6, [4, 5, 6, 7, 12, 13, 14, 15]),
        simd_swizzle!(u3, u7, [0, 1, 2, 3, 8, 9, 10, 11]),
        simd_swizzle!(u3, u7, [4, 5, 6, 7, 12, 13, 14, 15])
    ];
}

/// One 1-D pass over the eight rows, operating lanewise so that it combines
/// rows (the vertical pass); run after a transpose it is the horizontal pass.
#[inline]
fn dct_pass(rows: &mut [Row; 8], bias: i32, shift: i32) {
    let [row0, row1, row2, row3, row4, row5, row6, row7] = *rows;
    let bias = Row::splat(bias);

    // even part
    let p1 = (row2 + row6) * Row::splat(2217);

    let mut t2 = p1 + row6 * Row::splat(-7567);
    let mut t3 = p1 + row2 * Row::splat(3135);

    let mut t0 = (row0 + row4) << 12;
    let mut t1 = (row0 - row4) << 12;

    let x0 = t0 + t3 + bias;
    let x3 = t0 - t3 + bias;
    let x1 = t1 + t2 + bias;
    let x2 = t1 - t2 + bias;

    // odd part
    let p3 = row7 + row3;
    let p4 = row5 + row1;
    let p1 = row7 + row1;
    let p2 = row5 + row3;
    let p5 = (p3 + p4) * Row::splat(4816);

    t0 = row7 * Row::splat(1223);
    t1 = row5 * Row::splat(8410);
    t2 = row3 * Row::splat(12586);
    t3 = row1 * Row::splat(6149);

    let p1 = p5 + p1 * Row::splat(-3685);
    let p2 = p5 + p2 * Row::splat(-10497);
    let p3 = p3 * Row::splat(-8034);
    let p4 = p4 * Row::splat(-1597);

    t3 += p1 + p4;
    t2 += p2 + p3;
    t1 += p2 + p4;
    t0 += p1 + p3;

    *rows = [
        (x0 + t3) >> shift,
        (x1 + t2) >> shift,
        (x2 + t1) >> shift,
        (x3 + t0) >> shift,
        (x3 - t0) >> shift,
        (x2 - t1) >> shift,
        (x1 - t2) >> shift,
        (x0 - t3) >> shift
    ];
}

#[allow(clippy::cast_possible_truncation)]
pub fn idct_simd(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    let mut rows: [Row; 8] = core::array::from_fn(|r| load(&in_vector[r * 8..]));

    // When every AC term is zero the block is flat: (dc / 8) + 128, clamped.
    // The first row is reloaded shifted by one to exclude the DC term.
    let ac = rows[1..]
        .iter()
        .fold(load(&in_vector[1..]), |acc, r| acc | *r);
    if !ac.simd_ne(Row::splat(0)).any() {
        let coeff = ((in_vector[0] + 4 + 1024) >> 3).clamp(0, 255) as i16;
        let mut tmp = [0i16; 8];
        for r in 0..8 {
            out_vector
                .get_mut(r * stride..r * stride + 8)
                .unwrap_or(&mut tmp)
                .fill(coeff);
        }
        return;
    }

    dct_pass(&mut rows, 512, 10);
    transpose(&mut rows);
    dct_pass(&mut rows, SCALE_BITS, 17);
    transpose(&mut rows);

    for (r, row) in rows.into_iter().enumerate() {
        store_clamped(out_vector, r * stride, row);
    }
}

/// IDCT for blocks whose coefficients outside the top-left 4x4 are all zero,
/// so each pass only reads four inputs. Same math as the AVX2 version.
pub fn idct_simd_4x4(in_vector: &mut [i32; 64], out_vector: &mut [i16], stride: usize) {
    #[inline]
    fn pass(i0: Row, i1: Row, i2: Row, i3: Row, bias: i32, shift: i32) -> [Row; 8] {
        let t0 = (i0 << 12) + Row::splat(bias);

        let t2 = i2 * Row::splat(2217);
        let t3 = i2 * Row::splat(5352);

        let x0 = t0 + t3;
        let x3 = t0 - t3;
        let x1 = t0 + t2;
        let x2 = t0 - t2;

        let p5 = (i3 + i1) * Row::splat(4816);

        let p1 = p5 + i1 * Row::splat(-3685);
        let p2 = p5 + i3 * Row::splat(-10497);

        let t3 = p5 + i1 * Row::splat(867);
        let t2 = p5 + i3 * Row::splat(-5945);

        let t1 = p2 + i1 * Row::splat(-1597);
        let t0 = p1 + i3 * Row::splat(-8034);

        [
            (x0 + t3) >> shift,
            (x1 + t2) >> shift,
            (x2 + t1) >> shift,
            (x3 + t0) >> shift,
            (x3 - t0) >> shift,
            (x2 - t1) >> shift,
            (x1 - t2) >> shift,
            (x0 - t3) >> shift
        ]
    }

    let mut rows = pass(
        load(&in_vector[0..]),
        load(&in_vector[8..]),
        load(&in_vector[16..]),
        load(&in_vector[24..]),
        512,
        10
    );
    transpose(&mut rows);
    rows = pass(rows[0], rows[1], rows[2], rows[3], SCALE_BITS, 17);
    transpose(&mut rows);

    for (r, row) in rows.into_iter().enumerate() {
        store_clamped(out_vector, r * stride, row);
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
            if i / 8 < limit && i % 8 < limit {
                ((state >> 16) as i32 % 2048) - 1024
            } else {
                0
            }
        })
    }

    #[test]
    fn transpose_8x8() {
        let input: [i32; 64] = core::array::from_fn(|i| i as i32);
        let mut rows: [Row; 8] = core::array::from_fn(|r| load(&input[r * 8..]));
        transpose(&mut rows);
        for (r, row) in rows.iter().enumerate() {
            for c in 0..8 {
                assert_eq!(row[c], input[c * 8 + r]);
            }
        }
    }

    #[test]
    fn matches_scalar() {
        for seed in 0..64 {
            let coeffs = coefficients(seed, 8);
            let (mut a, mut b) = (coeffs, coeffs);
            let mut expected = [0i16; 64];
            let mut actual = [0i16; 64];
            idct_int(&mut a, &mut expected, 8);
            idct_simd(&mut b, &mut actual, 8);
            assert_eq!(expected, actual, "seed {seed}");
        }
    }

    #[test]
    fn flat_block_matches_scalar() {
        let mut a = [0i32; 64];
        a[0] = 300;
        let mut b = a;
        let mut expected = [0i16; 64];
        let mut actual = [0i16; 64];
        idct_int(&mut a, &mut expected, 8);
        idct_simd(&mut b, &mut actual, 8);
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
            idct_simd_4x4(&mut b, &mut actual, 8);
            assert_eq!(expected, actual, "seed {seed}");
        }
    }

    #[test]
    fn short_output_does_not_panic() {
        let mut coeffs = coefficients(1, 8);
        let mut out = [0i16; 20];
        idct_simd(&mut coeffs, &mut out, 8);
        idct_int(&mut coefficients(1, 8), &mut [0i16; 64], 8);
    }
}
