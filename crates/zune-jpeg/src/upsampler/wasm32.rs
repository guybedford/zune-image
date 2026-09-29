/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! WebAssembly simd128 upsampling, a translation of the NEON routines.

#![cfg(all(
    feature = "simd128",
    target_arch = "wasm32",
    target_feature = "simd128"
))]

use core::arch::wasm32::*;

#[inline]
fn load8(src: &[i16]) -> v128 {
    debug_assert!(src.len() >= 8);
    // SAFETY: caller passes at least eight i16; loads may be unaligned.
    unsafe { v128_load(src.as_ptr().cast()) }
}

#[inline]
fn store8(dst: &mut [i16; 8], v: v128) {
    // SAFETY: `dst` is exactly 16 bytes.
    unsafe { v128_store(dst.as_mut_ptr().cast(), v) }
}

pub fn upsample_horizontal_simd128(
    input: &[i16], in_near: &[i16], in_far: &[i16], scratch: &mut [i16], output: &mut [i16]
) {
    assert_eq!(input.len() * 2, output.len());
    assert!(input.len() > 2);

    let len = input.len();
    if len < 10 {
        return super::scalar::upsample_horizontal(input, in_near, in_far, scratch, output);
    }

    output[0] = input[0];
    output[1] = (input[0] * 3 + input[1] + 2) >> 2;

    let v_three = i16x8_splat(3);
    let v_two = i16x8_splat(2);

    // Ten inputs give the eight centre samples plus one neighbour on each side.
    let upsample8 = |input: &[i16; 10], output: &mut [i16; 16]| {
        let v_prev = load8(&input[0..]);
        let v_curr = load8(&input[1..]);
        let v_next = load8(&input[2..]);

        let v_common = i16x8_add(i16x8_mul(v_curr, v_three), v_two);
        let v_even = i16x8_shr(i16x8_add(v_common, v_prev), 2);
        let v_odd = i16x8_shr(i16x8_add(v_common, v_next), 2);

        let (lo, hi) = output.split_at_mut(8);
        store8(
            lo.try_into().unwrap(),
            i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v_even, v_odd)
        );
        store8(
            hi.try_into().unwrap(),
            i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v_even, v_odd)
        );
    };

    for (input, output) in input
        .windows(10)
        .step_by(8)
        .zip(output[2..].chunks_exact_mut(16))
    {
        upsample8(input.try_into().unwrap(), output.try_into().unwrap());
    }

    // Redo the tail so the last partial chunk is covered.
    if let Some(rest_input) = input.last_chunk::<10>() {
        let end = output.len() - 2;
        if let Some(rest_output) = output[..end].last_chunk_mut::<16>() {
            upsample8(rest_input, rest_output);
        }
    }

    output[output.len() - 2] = (3 * input[len - 1] + input[len - 2] + 2) >> 2;
    output[output.len() - 1] = input[len - 1];
}

pub fn upsample_vertical_simd128(
    input: &[i16], in_near: &[i16], in_far: &[i16], scratch: &mut [i16], output: &mut [i16]
) {
    assert_eq!(input.len() * 2, output.len());
    assert_eq!(in_near.len(), input.len());
    assert_eq!(in_far.len(), input.len());

    let len = input.len();
    if len < 16 {
        return super::scalar::upsample_vertical(input, in_near, in_far, scratch, output);
    }

    let middle = output.len() / 2;
    let (out_top, out_bottom) = output.split_at_mut(middle);

    let v_three = i16x8_splat(3);
    let v_two = i16x8_splat(2);

    let upsample8 = |input: &[i16; 8],
                     in_near: &[i16; 8],
                     in_far: &[i16; 8],
                     out_top: &mut [i16; 8],
                     out_bottom: &mut [i16; 8]| {
        let v_in = load8(input);
        let v_near = load8(in_near);
        let v_far = load8(in_far);

        let v_common = i16x8_add(i16x8_mul(v_in, v_three), v_two);
        store8(out_top, i16x8_shr(i16x8_add(v_common, v_near), 2));
        store8(out_bottom, i16x8_shr(i16x8_add(v_common, v_far), 2));
    };

    let chunks = input
        .chunks_exact(8)
        .zip(in_near.chunks_exact(8))
        .zip(in_far.chunks_exact(8))
        .zip(out_top.chunks_exact_mut(8))
        .zip(out_bottom.chunks_exact_mut(8));

    for ((((input, in_near), in_far), out_top), out_bottom) in chunks {
        upsample8(
            input.try_into().unwrap(),
            in_near.try_into().unwrap(),
            in_far.try_into().unwrap(),
            out_top.try_into().unwrap(),
            out_bottom.try_into().unwrap()
        );
    }

    if let (Some(rest), Some(rest_near), Some(rest_far)) = (
        input.last_chunk::<8>(),
        in_near.last_chunk::<8>(),
        in_far.last_chunk::<8>()
    ) {
        if let (Some(rest_top), Some(rest_bottom)) = (
            out_top.last_chunk_mut::<8>(),
            out_bottom.last_chunk_mut::<8>()
        ) {
            upsample8(rest, rest_near, rest_far, rest_top, rest_bottom);
        }
    }
}

pub fn upsample_hv_simd128(
    input: &[i16], in_near: &[i16], in_far: &[i16], scratch_space: &mut [i16], output: &mut [i16]
) {
    assert_eq!(input.len() * 4, output.len());
    assert!(input.len() * 2 <= scratch_space.len());

    let scratch_space = &mut scratch_space[..input.len() * 2];
    upsample_vertical_simd128(input, in_near, in_far, &mut [], scratch_space);

    let scratch_half = scratch_space.len() / 2;
    let output_half = output.len() / 2;
    let (scratch_top, scratch_bottom) = scratch_space.split_at_mut(scratch_half);
    let (out_top, out_bottom) = output.split_at_mut(output_half);

    let mut t = [0];
    upsample_horizontal_simd128(scratch_top, &[], &[], &mut t, out_top);
    upsample_horizontal_simd128(scratch_bottom, &[], &[], &mut t, out_bottom);
}
