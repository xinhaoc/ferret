# 17. Mathematical Functions[](#mathematical-functions-appendix "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

The reference manual lists, along with their description, all the functions of the C/C++ standard library mathematical functions that are supported in device code, as well as all intrinsic functions (that are only supported in device code).

This section provides accuracy information for some of these functions when applicable. It uses ULP for quantification. For further information on the definition of the Unit in the Last Place (ULP), please see Jean-Michel Muller’s paper *On the definition of ulp(x)*, RR-5504, LIP RR-2005-09, INRIA, LIP. 2005, pp.16 at <https://hal.inria.fr/inria-00070503/document>.

Mathematical functions supported in device code do not set the global `errno` variable, nor report any floating-point exceptions to indicate errors; thus, if error diagnostic mechanisms are required, the user should implement additional screening for inputs and outputs of the functions. The user is responsible for the validity of pointer arguments. The user must not pass uninitialized parameters to the Mathematical functions as this may result in undefined behavior: functions are inlined in the user program and thus are subject to compiler optimizations.

## 17.1. Standard Functions[](#standard-functions "Permalink to this headline")

The functions from this section can be used in both host and device code.

This section specifies the error bounds of each function when executed on the device and also when executed on the host in the case where the host does not supply the function.

The error bounds are generated from extensive but not exhaustive tests, so they are not guaranteed bounds.

**Single-Precision Floating-Point Functions**

Addition and multiplication are IEEE-compliant, so have a maximum error of 0.5 ulp.

The recommended way to round a single-precision floating-point operand to an integer, with the result being a single-precision floating-point number is `rintf()`, not `roundf()`. The reason is that `roundf()` maps to a 4-instruction sequence on the device, whereas `rintf()` maps to a single instruction. `truncf()`, `ceilf()`, and `floorf()` each map to a single instruction as well.

Table 17 Single-Precision Mathematical Standard Library Functions with Maximum ULP Error. The maximum error is stated as the absolute value of the difference in ulps between the result returned by the CUDA library function and a correctly rounded single-precision result obtained according to the round-to-nearest ties-to-even rounding mode.[](#id479 "Permalink to this table")


| Function | Maximum ulp error |
| --- | --- |
| `x+y` | 0 (IEEE-754 round-to-nearest-even) |
| `x*y` | 0 (IEEE-754 round-to-nearest-even) |
| `x/y` | 0 for compute capability \(\ge 2\) when compiled with `-prec-div=true`  2 (full range), otherwise |
| `1/x` | 0 for compute capability \(\ge 2\) when compiled with `-prec-div=true`  1 (full range), otherwise |
| `rsqrtf(x)`  `1/sqrtf(x)` | 2 (full range)  Applies to `1/sqrtf(x)` only when it is converted to `rsqrtf(x)` by the compiler. |
| `sqrtf(x)` | 0 when compiled with `-prec-sqrt=true`  Otherwise 1 for compute capability \(\ge 5.2\)  and 3 for older architectures |
| `cbrtf(x)` | 1 (full range) |
| `rcbrtf(x)` | 1 (full range) |
| `hypotf(x,y)` | 3 (full range) |
| `rhypotf(x,y)` | 2 (full range) |
| `norm3df(x,y,z)` | 3 (full range) |
| `rnorm3df(x,y,z)` | 2 (full range) |
| `norm4df(x,y,z,t)` | 3 (full range) |
| `rnorm4df(x,y,z,t)` | 2 (full range) |
| `normf(dim,arr)` | An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off. . |
| `rnormf(dim,arr)` | An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off. . |
| `expf(x)` | 2 (full range) |
| `exp2f(x)` | 2 (full range) |
| `exp10f(x)` | 2 (full range) |
| `expm1f(x)` | 1 (full range) |
| `logf(x)` | 1 (full range) |
| `log2f(x)` | 1 (full range) |
| `log10f(x)` | 2 (full range) |
| `log1pf(x)` | 1 (full range) |
| `sinf(x)` | 2 (full range) |
| `cosf(x)` | 2 (full range) |
| `tanf(x)` | 4 (full range) |
| `sincosf(x,sptr,cptr)` | 2 (full range) |
| `sinpif(x)` | 1 (full range) |
| `cospif(x)` | 1 (full range) |
| `sincospif(x,sptr,cptr)` | 1 (full range) |
| `asinf(x)` | 2 (full range) |
| `acosf(x)` | 2 (full range) |
| `atanf(x)` | 2 (full range) |
| `atan2f(y,x)` | 3 (full range) |
| `sinhf(x)` | 3 (full range) |
| `coshf(x)` | 2 (full range) |
| `tanhf(x)` | 2 (full range) |
| `asinhf(x)` | 3 (full range) |
| `acoshf(x)` | 4 (full range) |
| `atanhf(x)` | 3 (full range) |
| `powf(x,y)` | 4 (full range) |
| `erff(x)` | 2 (full range) |
| `erfcf(x)` | 4 (full range) |
| `erfinvf(x)` | 2 (full range) |
| `erfcinvf(x)` | 4 (full range) |
| `erfcxf(x)` | 4 (full range) |
| `normcdff(x)` | 5 (full range) |
| `normcdfinvf(x)` | 5 (full range) |
| `lgammaf(x)` | 6 (outside interval -10.001 … -2.264; larger inside) |
| `tgammaf(x)` | 5 (full range) |
| `fmaf(x,y,z)` | 0 (full range) |
| `frexpf(x,exp)` | 0 (full range) |
| `ldexpf(x,exp)` | 0 (full range) |
| `scalbnf(x,n)` | 0 (full range) |
| `scalblnf(x,l)` | 0 (full range) |
| `logbf(x)` | 0 (full range) |
| `ilogbf(x)` | 0 (full range) |
| `j0f(x)` | 9 for |x| < 8  otherwise, the maximum absolute error is 2.2 x 10-6 |
| `j1f(x)` | 9 for |x| < 8  otherwise, the maximum absolute error is 2.2 x 10-6 |
| `jnf(n,x)` | For n = 128, the maximum absolute error is 2.2 x 10-6 |
| `y0f(x)` | 9 for |x| < 8  otherwise, the maximum absolute error is 2.2 x 10-6 |
| `y1f(x)` | 9 for |x| < 8  otherwise, the maximum absolute error is 2.2 x 10-6 |
| `ynf(n,x)` | ceil(2 + 2.5n) for |x| < n  otherwise, the maximum absolute error is 2.2 x 10-6 |
| `cyl_bessel_i0f(x)` | 6 (full range) |
| `cyl_bessel_i1f(x)` | 6 (full range) |
| `fmodf(x,y)` | 0 (full range) |
| `remainderf(x,y)` | 0 (full range) |
| `remquof(x,y,iptr)` | 0 (full range) |
| `modff(x,iptr)` | 0 (full range) |
| `fdimf(x,y)` | 0 (full range) |
| `truncf(x)` | 0 (full range) |
| `roundf(x)` | 0 (full range) |
| `rintf(x)` | 0 (full range) |
| `nearbyintf(x)` | 0 (full range) |
| `ceilf(x)` | 0 (full range) |
| `floorf(x)` | 0 (full range) |
| `lrintf(x)` | 0 (full range) |
| `lroundf(x)` | 0 (full range) |
| `llrintf(x)` | 0 (full range) |
| `llroundf(x)` | 0 (full range) |

**Double-Precision Floating-Point Functions**

The recommended way to round a double-precision floating-point operand to an integer, with the result being a double-precision floating-point number is `rint()`, not `round()`. The reason is that `round()` maps to a 5-instruction sequence on the device, whereas `rint()` maps to a single instruction. `trunc()`, `ceil()`, and `floor()` each map to a single instruction as well.

Table 18 Double-Precision Mathematical Standard Library Functions with Maximum ULP Error. The maximum error is stated as the absolute value of the difference in ulps between the result returned by the CUDA library function and a correctly rounded double-precision result obtained according to the round-to-nearest ties-to-even rounding mode.[](#id480 "Permalink to this table")


| Function | Maximum ulp error |
| --- | --- |
| `x+y` | 0 (IEEE-754 round-to-nearest-even) |
| `x*y` | 0 (IEEE-754 round-to-nearest-even) |
| `x/y` | 0 (IEEE-754 round-to-nearest-even) |
| `1/x` | 0 (IEEE-754 round-to-nearest-even) |
| `sqrt(x)` | 0 (IEEE-754 round-to-nearest-even) |
| `rsqrt(x)` | 1 (full range) |
| `cbrt(x)` | 1 (full range) |
| `rcbrt(x)` | 1 (full range) |
| `hypot(x,y)` | 2 (full range) |
| `rhypot(x,y)` | 1 (full range) |
| `norm3d(x,y,z)` | 2 (full range) |
| `rnorm3d(x,y,z)` | 1 (full range) |
| `norm4d(x,y,z,t)` | 2 (full range) |
| `rnorm4d(x,y,z,t)` | 1 (full range) |
| `norm(dim,arr)` | An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off. |
| `rnorm(dim,arr)` | An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off. |
| `exp(x)` | 1 (full range) |
| `exp2(x)` | 1 (full range) |
| `exp10(x)` | 1 (full range) |
| `expm1(x)` | 1 (full range) |
| `log(x)` | 1 (full range) |
| `log2(x)` | 1 (full range) |
| `log10(x)` | 1 (full range) |
| `log1p(x)` | 1 (full range) |
| `sin(x)` | 2 (full range) |
| `cos(x)` | 2 (full range) |
| `tan(x)` | 2 (full range) |
| `sincos(x,sptr,cptr)` | 2 (full range) |
| `sinpi(x)` | 2 (full range) |
| `cospi(x)` | 2 (full range) |
| `sincospi(x,sptr,cptr)` | 2 (full range) |
| `asin(x)` | 2 (full range) |
| `acos(x)` | 2 (full range) |
| `atan(x)` | 2 (full range) |
| `atan2(y,x)` | 2 (full range) |
| `sinh(x)` | 2 (full range) |
| `cosh(x)` | 1 (full range) |
| `tanh(x)` | 1 (full range) |
| `asinh(x)` | 3 (full range) |
| `acosh(x)` | 3 (full range) |
| `atanh(x)` | 2 (full range) |
| `pow(x,y)` | 2 (full range) |
| `erf(x)` | 2 (full range) |
| `erfc(x)` | 5 (full range) |
| `erfinv(x)` | 5 (full range) |
| `erfcinv(x)` | 6 (full range) |
| `erfcx(x)` | 4 (full range) |
| `normcdf(x)` | 5 (full range) |
| `normcdfinv(x)` | 8 (full range) |
| `lgamma(x)` | 4 (outside interval -23.0001 … -2.2637; larger inside) |
| `tgamma(x)` | 10 (full range) |
| `fma(x,y,z)` | 0 (IEEE-754 round-to-nearest-even) |
| `frexp(x,exp)` | 0 (full range) |
| `ldexp(x,exp)` | 0 (full range) |
| `scalbn(x,n)` | 0 (full range) |
| `scalbln(x,l)` | 0 (full range) |
| `logb(x)` | 0 (full range) |
| `ilogb(x)` | 0 (full range) |
| `j0(x)` | 7 for |x| < 8  otherwise, the maximum absolute error is 5 x 10-12 |
| `j1(x)` | 7 for |x| < 8  otherwise, the maximum absolute error is 5 x 10-12 |
| `jn(n,x)` | For n = 128, the maximum absolute error is 5 x 10-12 |
| `y0(x)` | 7 for |x| < 8  otherwise, the maximum absolute error is 5 x 10-12 |
| `y1(x)` | 7 for |x| < 8  otherwise, the maximum absolute error is 5 x 10-12 |
| `yn(n,x)` | For |x| > 1.5n, the maximum absolute error is 5 x 10-12 |
| `cyl_bessel_i0(x)` | 6 (full range) |
| `cyl_bessel_i1(x)` | 6 (full range) |
| `fmod(x,y)` | 0 (full range) |
| `remainder(x,y)` | 0 (full range) |
| `remquo(x,y,iptr)` | 0 (full range) |
| `modf(x,iptr)` | 0 (full range) |
| `fdim(x,y)` | 0 (full range) |
| `trunc(x)` | 0 (full range) |
| `round(x)` | 0 (full range) |
| `rint(x)` | 0 (full range) |
| `nearbyint(x)` | 0 (full range) |
| `ceil(x)` | 0 (full range) |
| `floor(x)` | 0 (full range) |
| `lrint(x)` | 0 (full range) |
| `lround(x)` | 0 (full range) |
| `llrint(x)` | 0 (full range) |
| `llround(x)` | 0 (full range) |

**Quad-Precision Floating-Point Functions**

Note that the quad-precision mathematical functions are currently only available to devices with compute capability 10.0 and later.
Due to the specifics of implementation, the support of `__float128` and `_Float128` types in device code is also limited to select combinations of host platforms, see also [Host Compiler Extensions](#host-compiler-extensions).

Table 19 Quad-Precision Mathematical Standard Library Functions with Maximum ULP Error. The maximum error is stated as the absolute value of the difference in ulps between the result returned by the CUDA library function and a correctly rounded quad-precision result obtained according to the round-to-nearest ties-to-even rounding mode.[](#id481 "Permalink to this table")


| Function | Maximum ulp error |
| --- | --- |
| `x+y` `__nv_fp128_add(x, y)` | 0 (IEEE-754 round-to-nearest-even) |
| `x-y` `__nv_fp128_sub(x, y)` | 0 (IEEE-754 round-to-nearest-even) |
| `x*y` `__nv_fp128_mul(x, y)` | 0 (IEEE-754 round-to-nearest-even) |
| `x/y` `__nv_fp128_div(x, y)` | 0 (IEEE-754 round-to-nearest-even) |
| `__nv_fp128_sqrt(x)` | 0 (IEEE-754 round-to-nearest-even) |
| `__nv_fp128_fma(x, y, z)` | 0 (IEEE-754 round-to-nearest-even) |
| `__nv_fp128_sin(x)` | 1 (full range) |
| `__nv_fp128_cos(x)` | 1 (full range) |
| `__nv_fp128_tan(x)` | 1 (full range) |
| `__nv_fp128_asin(x)` | 1 (full range) |
| `__nv_fp128_acos(x)` | 1 (full range) |
| `__nv_fp128_atan(x)` | 1 (full range) |
| `__nv_fp128_exp(x)` | 1 (full range) |
| `__nv_fp128_exp2(x)` | 1 (full range) |
| `__nv_fp128_exp10(x)` | 1 (full range) |
| `__nv_fp128_expm1(x)` | 1 (full range) |
| `__nv_fp128_log(x)` | 1 (full range) |
| `__nv_fp128_log2(x)` | 1 (full range) |
| `__nv_fp128_log10(x)` | 1 (full range) |
| `__nv_fp128_log1p(x)` | 1 (full range) |
| `__nv_fp128_pow(x, y)` | 1 (full range) |
| `__nv_fp128_sinh(x)` | 1 (full range) |
| `__nv_fp128_cosh(x)` | 1 (full range) |
| `__nv_fp128_tanh(x)` | 1 (full range) |
| `__nv_fp128_asinh(x)` | 1 (full range) |
| `__nv_fp128_acosh(x)` | 1 (full range) |
| `__nv_fp128_atanh(x)` | 1 (full range) |
| `__nv_fp128_hypot(x, y)` | 1 (full range) |
| `__nv_fp128_ceil(x)` | 0 (full range) |
| `__nv_fp128_trunc(x)` | 0 (full range) |
| `__nv_fp128_floor(x)` | 0 (full range) |
| `__nv_fp128_round(x)` | 0 (full range) |
| `__nv_fp128_rint(x)` | 0 (full range) |
| `__nv_fp128_fabs(x)` | 0 (full range) |
| `__nv_fp128_copysign(x, y)` | 0 (full range) |
| `__nv_fp128_fmax(x, y)` | 0 (full range) |
| `__nv_fp128_fmin(x, y)` | 0 (full range) |
| `__nv_fp128_fdim(x, y)` | 0 (full range) |
| `__nv_fp128_fmod(x, y)` | 0 (full range) |
| `__nv_fp128_remainder(x, y)` | 0 (full range) |
| `__nv_fp128_frexp(x, nptr)` | 0 (full range) |
| `__nv_fp128_modf(x, iptr)` | 0 (full range) |
| `__nv_fp128_ldexp(x, exp)` | 0 (full range) |
| `__nv_fp128_ilogb(x)` | 0 (full range) |

## 17.2. Intrinsic Functions[](#intrinsic-functions "Permalink to this headline")

The functions from this section can only be used in device code.

Among these functions are the less accurate, but faster versions of some of the functions of [Standard Functions](#mathematical-functions-appendix-standard-functions).
They have the same name prefixed with `__` (such as `__sinf(x)`).
They are faster as they map to fewer native instructions.
The compiler has an option (`-use_fast_math`) that forces each function in [Table 20](#intrinsic-functions-functions-affected-use-fast-math)
to compile to its intrinsic counterpart. In addition to reducing the accuracy of the affected functions,
it may also cause some differences in special case handling. A more robust approach is to selectively replace
mathematical function calls by calls to intrinsic functions only where it is merited by the performance gains
and where changed properties such as reduced accuracy and different special case handling can be tolerated.

Table 20 Functions Affected by -use\_fast\_math[](#intrinsic-functions-functions-affected-use-fast-math "Permalink to this table")


| Operator/Function | Device Function |
| --- | --- |
| `x/y` | `__fdividef(x,y)` |
| `sinf(x)` | `__sinf(x)` |
| `cosf(x)` | `__cosf(x)` |
| `tanf(x)` | `__tanf(x)` |
| `sincosf(x,sptr,cptr)` | `__sincosf(x,sptr,cptr)` |
| `logf(x)` | `__logf(x)` |
| `log2f(x)` | `__log2f(x)` |
| `log10f(x)` | `__log10f(x)` |
| `expf(x)` | `__expf(x)` |
| `exp10f(x)` | `__exp10f(x)` |
| `powf(x,y)` | `__powf(x,y)` |
| `tanhf(x)` | `__tanhf(x)` |

**Single-Precision Floating-Point Functions**

`__fadd_[rn,rz,ru,rd]()` and `__fmul_[rn,rz,ru,rd]()` map to addition and multiplication operations that the compiler never merges into FMADs. By contrast, additions and multiplications generated from the ‘\*’ and ‘+’ operators will frequently be combined into FMADs.

Functions suffixed with `_rn` operate using the round to nearest even rounding mode.

Functions suffixed with `_rz` operate using the round towards zero rounding mode.

Functions suffixed with `_ru` operate using the round up (to positive infinity) rounding mode.

Functions suffixed with `_rd` operate using the round down (to negative infinity) rounding mode.

The accuracy of floating-point division varies depending on whether the code is compiled with `-prec-div=false`
or `-prec-div=true`. When the code is compiled with `-prec-div=false`, both the regular division `/`
operator and `__fdividef(x,y)` have the same accuracy, but for 2126 < `|y|` < 2128,
`__fdividef(x,y)` delivers a result of zero, whereas the `/` operator delivers the correct result to
within the accuracy stated in [Table 21](#intrinsic-functions-single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library).
Also, for 2126 < `|y|` < 2128, if `x` is infinity, `__fdividef(x,y)` delivers
a `NaN` (as a result of multiplying infinity by zero), while the `/` operator returns infinity.
On the other hand, the `/` operator is IEEE-compliant when the code is compiled with `-prec-div=true`
or without any `-prec-div` option at all since its default value is true.

Table 21 Single-Precision Floating-Point Intrinsic Functions. (Supported by the CUDA Runtime Library with Respective Error Bounds)[](#intrinsic-functions-single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library "Permalink to this table")


| Function | Error bounds |
| --- | --- |
| `__fadd_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fsub_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fmul_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fmaf_[rn,rz,ru,rd](x,y,z)` | IEEE-compliant. |
| `__frcp_[rn,rz,ru,rd](x)` | IEEE-compliant. |
| `__fsqrt_[rn,rz,ru,rd](x)` | IEEE-compliant. |
| `__frsqrt_rn(x)` | IEEE-compliant. |
| `__fdiv_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fdividef(x,y)` | For `|y|` in [\(2^{-126}, 2^{126}\)], the maximum ulp error is 2. |
| `__expf(x)` | The maximum ulp error is `2 + floor(abs(1.173 * x))`. |
| `__exp10f(x)` | The maximum ulp error is `2 + floor(abs(2.97 * x))`. |
| `__logf(x)` | For `x` in [0.5, 2], the maximum absolute error is \(2^{-21.41}\), otherwise, the maximum ulp error is 3. |
| `__log2f(x)` | For `x` in [0.5, 2], the maximum absolute error is \(2^{-22}\), otherwise, the maximum ulp error is 2. |
| `__log10f(x)` | For `x` in [0.5, 2], the maximum absolute error is \(2^{-24}\), otherwise, the maximum ulp error is 3. |
| `__sinf(x)` | For `x` in [\(-\pi, \pi\)], the maximum absolute error is \(2^{-21.41}\), and larger otherwise. |
| `__cosf(x)` | For `x` in [\(-\pi, \pi\)], the maximum absolute error is \(2^{-21.19}\), and larger otherwise. |
| `__sincosf(x,sptr,cptr)` | Same as `__sinf(x)` and `__cosf(x)`. |
| `__tanf(x)` | Derived from its implementation as `__sinf(x) * (1/__cosf(x))`. |
| `__powf(x, y)` | Derived from its implementation as `exp2f(y * __log2f(x))`. |
| `__tanhf(x)` | The maximum relative error of the current implementation is \(2^{-11}\). Subnormal results of this fast intrinsic are not flushed to zero even under `-ftz=true` compiler setting. Available for devices with compute capability of at least 7.5; and defaults to regular `tanhf()` function behavior on other devices. |

**Double-Precision Floating-Point Functions**

`__dadd_rn()` and `__dmul_rn()` map to addition and multiplication operations that the compiler never merges into FMADs. By contrast, additions and multiplications generated from the ‘\*’ and ‘+’ operators will frequently be combined into FMADs.

Table 22 Double-Precision Floating-Point Intrinsic Functions. (Supported by the CUDA Runtime Library with Respective Error Bounds)[](#id482 "Permalink to this table")


| Function | Error bounds |
| --- | --- |
| `__dadd_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__dsub_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__dmul_[rn,rz,ru,rd](x,y)` | IEEE-compliant. |
| `__fma_[rn,rz,ru,rd](x,y,z)` | IEEE-compliant. |
| `__ddiv_[rn,rz,ru,rd](x,y)(x,y)` | IEEE-compliant.  Requires compute capability *>* 2. |
| `__drcp_[rn,rz,ru,rd](x)` | IEEE-compliant.  Requires compute capability *>* 2. |
| `__dsqrt_[rn,rz,ru,rd](x)` | IEEE-compliant.  Requires compute capability *>* 2. |
