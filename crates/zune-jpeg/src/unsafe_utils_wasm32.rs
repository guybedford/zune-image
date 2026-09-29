/*
 * Copyright (c) 2026.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! WebAssembly simd128 helpers mirroring the AVX2 and NEON ones.
//!
//! A `YmmRegister` is a pair of 128-bit vectors holding eight `i32` lanes, so
//! the IDCT reads the same across all three backends.

#![cfg(all(
    feature = "simd128",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
#![allow(clippy::wildcard_imports)]

use core::arch::wasm32::*;
use core::ops::{Add, AddAssign, BitOr, BitOrAssign, Mul, MulAssign, Sub};

pub type VecType = (v128, v128);

/// Load eight `i32` values from an unaligned pointer.
pub unsafe fn loadu(src: *const i32) -> VecType {
    unsafe { (v128_load(src.cast()), v128_load(src.add(4).cast())) }
}

#[derive(Clone, Copy)]
pub struct YmmRegister {
    pub(crate) mm256: VecType
}

impl YmmRegister {
    #[inline]
    pub unsafe fn load(src: *const i32) -> Self {
        unsafe { loadu(src).into() }
    }

    #[inline]
    pub fn map2(self, other: Self, f: impl Fn(v128, v128) -> v128) -> Self {
        YmmRegister {
            mm256: (
                f(self.mm256.0, other.mm256.0),
                f(self.mm256.1, other.mm256.1)
            )
        }
    }

    #[inline]
    pub fn all_zero(self) -> bool {
        !v128_any_true(v128_or(self.mm256.0, self.mm256.1))
    }

    #[inline]
    pub fn const_shl<const N: u32>(self) -> Self {
        YmmRegister {
            mm256: (i32x4_shl(self.mm256.0, N), i32x4_shl(self.mm256.1, N))
        }
    }

    #[inline]
    pub fn const_shra<const N: u32>(self) -> Self {
        YmmRegister {
            mm256: (i32x4_shr(self.mm256.0, N), i32x4_shr(self.mm256.1, N))
        }
    }
}

impl<T> Add<T> for YmmRegister
where
    T: Into<Self>
{
    type Output = YmmRegister;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        self.map2(rhs.into(), |a, b| i32x4_add(a, b))
    }
}

impl<T> Sub<T> for YmmRegister
where
    T: Into<Self>
{
    type Output = YmmRegister;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        self.map2(rhs.into(), |a, b| i32x4_sub(a, b))
    }
}

impl<T> AddAssign<T> for YmmRegister
where
    T: Into<Self>
{
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs.into();
    }
}

impl<T> Mul<T> for YmmRegister
where
    T: Into<Self>
{
    type Output = YmmRegister;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.map2(rhs.into(), |a, b| i32x4_mul(a, b))
    }
}

impl<T> MulAssign<T> for YmmRegister
where
    T: Into<Self>
{
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs.into();
    }
}

impl<T> BitOr<T> for YmmRegister
where
    T: Into<Self>
{
    type Output = YmmRegister;

    #[inline]
    fn bitor(self, rhs: T) -> Self::Output {
        self.map2(rhs.into(), |a, b| v128_or(a, b))
    }
}

impl<T> BitOrAssign<T> for YmmRegister
where
    T: Into<Self>
{
    #[inline]
    fn bitor_assign(&mut self, rhs: T) {
        *self = *self | rhs.into();
    }
}

impl From<i32> for YmmRegister {
    #[inline]
    fn from(val: i32) -> Self {
        let dup = i32x4_splat(val);
        YmmRegister { mm256: (dup, dup) }
    }
}

impl From<VecType> for YmmRegister {
    #[inline]
    fn from(mm256: VecType) -> Self {
        YmmRegister { mm256 }
    }
}

/// Transpose a 4x4 matrix of `i32` held in four vectors.
#[inline]
fn transpose4(v0: &mut v128, v1: &mut v128, v2: &mut v128, v3: &mut v128) {
    // a0 b0 a1 b1 | a2 b2 a3 b3 | c0 d0 c1 d1 | c2 d2 c3 d3
    let t0 = i32x4_shuffle::<0, 4, 1, 5>(*v0, *v1);
    let t1 = i32x4_shuffle::<2, 6, 3, 7>(*v0, *v1);
    let t2 = i32x4_shuffle::<0, 4, 1, 5>(*v2, *v3);
    let t3 = i32x4_shuffle::<2, 6, 3, 7>(*v2, *v3);

    *v0 = i64x2_shuffle::<0, 2>(t0, t2);
    *v1 = i64x2_shuffle::<1, 3>(t0, t2);
    *v2 = i64x2_shuffle::<0, 2>(t1, t3);
    *v3 = i64x2_shuffle::<1, 3>(t1, t3);
}

/// Transpose an 8x8 matrix of `i32` held in eight registers.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn transpose(
    v0: &mut YmmRegister, v1: &mut YmmRegister, v2: &mut YmmRegister, v3: &mut YmmRegister,
    v4: &mut YmmRegister, v5: &mut YmmRegister, v6: &mut YmmRegister, v7: &mut YmmRegister
) {
    use core::mem::swap;

    // Swap the upper-right and lower-left quadrants, then transpose each
    // quadrant in place.
    swap(&mut v0.mm256.1, &mut v4.mm256.0);
    swap(&mut v1.mm256.1, &mut v5.mm256.0);
    swap(&mut v2.mm256.1, &mut v6.mm256.0);
    swap(&mut v3.mm256.1, &mut v7.mm256.0);

    transpose4(
        &mut v0.mm256.0,
        &mut v1.mm256.0,
        &mut v2.mm256.0,
        &mut v3.mm256.0
    );
    transpose4(
        &mut v0.mm256.1,
        &mut v1.mm256.1,
        &mut v2.mm256.1,
        &mut v3.mm256.1
    );
    transpose4(
        &mut v4.mm256.0,
        &mut v5.mm256.0,
        &mut v6.mm256.0,
        &mut v7.mm256.0
    );
    transpose4(
        &mut v4.mm256.1,
        &mut v5.mm256.1,
        &mut v6.mm256.1,
        &mut v7.mm256.1
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_8x8() {
        let input: [i32; 64] = core::array::from_fn(|i| i as i32);
        let mut rows: [YmmRegister; 8] =
            core::array::from_fn(|r| unsafe { YmmRegister::load(input[r * 8..].as_ptr()) });
        let [mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7] = rows;
        transpose(
            &mut r0, &mut r1, &mut r2, &mut r3, &mut r4, &mut r5, &mut r6, &mut r7
        );
        rows = [r0, r1, r2, r3, r4, r5, r6, r7];

        let mut output = [0i32; 64];
        for (r, reg) in rows.iter().enumerate() {
            unsafe {
                v128_store(output[r * 8..].as_mut_ptr().cast(), reg.mm256.0);
                v128_store(output[r * 8 + 4..].as_mut_ptr().cast(), reg.mm256.1);
            }
        }
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(output[r * 8 + c], input[c * 8 + r]);
            }
        }
    }
}
