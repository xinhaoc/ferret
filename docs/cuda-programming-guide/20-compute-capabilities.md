# 20. Compute Capabilities[](#compute-capabilities "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

The general specifications and features of a compute device depend on its compute capability (see [Compute Capability](#compute-capability)).

[Table 26](#features-and-technical-specifications-feature-support-per-compute-capability) and
[Table 27](#features-and-technical-specifications-technical-specifications-per-compute-capability) show the features and technical specifications associated with each compute capability that is currently supported.

Section [Floating-Point Standard](#floating-point-standard) reviews compliance with the IEEE floating-point standard.

Sections [Compute Capability 5.x](#compute-capability-5-x), [Compute Capability 6.x](#compute-capability-6-x), [Compute Capability 7.x](#compute-capability-7-x), [Compute Capability 8.x](#compute-capability-8-x), [Compute Capability 9.0](#compute-capability-9-0), [Compute Capability 10.0](#compute-capability-10-0), and [Compute Capability 12.0](#compute-capability-12-0) give more details on the architecture of devices with these respective compute capabilities.

## 20.1. Feature Availability[](#feature-availability "Permalink to this headline")

Most compute features introduced with a compute architecture are intended to be available on all subsequent architectures. This is shown in [Table 26](#features-and-technical-specifications-feature-support-per-compute-capability) by the “yes” for availability of a feature on compute capabilities subsequent to its introduction.

### 20.1.1. Architecture-Specific Features[](#architecture-specific-features "Permalink to this headline")

Beginning with devices of Compute Capability 9.0, specialized compute features that are introduced with an architecture may not be guaranteed to be available on all subsequent compute capabilities. These features are called *architecture-specific* features and target acceleration of specialized operations, such as Tensor Core operations, which are not intended for all classes of compute capabilities or may significantly change on future generations. Code must be compiled with an architecture-specific compiler target (see [Feature Set Compiler Targets](#feature-set-compiler-targets)) to enable architecture-specific features. Code compiled with an architecture-specific compiler target can only be run on the exact compute capability it was compiled for.

### 20.1.2. Family-Specific Features[](#family-specific-features "Permalink to this headline")

Beginning with devices of Compute Capability 10.0, some architecture-specific features are common to devices of more than one compute capability. The devices that contain these features are part of the same family and these features can also be called *family-specific* features. Family-specific features are guaranteed to be available on all devices in the same family. A family-specific compiler target is required to enable family-specific features. See [Section 20.1.3](#feature-set-compiler-targets). Code compiled for a family-specific target can only be run on GPUs which are members of that family.

### 20.1.3. Feature Set Compiler Targets[](#feature-set-compiler-targets "Permalink to this headline")

There are three sets of compute features which the compiler can target:

**Baseline Feature Set**: The predominant set of compute features that are introduced with the intent to be available for subsequent compute architectures. These features and their availability are summarized in [Table 26](#features-and-technical-specifications-feature-support-per-compute-capability).

**Architecture-Specific Feature Set**: A small and highly specialized set of features called architecture-specific, that are introduced to accelerate specialized operations, which are not guaranteed to be available or might change significantly on subsequent compute architectures. These features are summarized in the respective “Compute Capability #.#” subsections. The architecture-specific feature set is a superset of the family-specific feature set. Architecture-specific compiler targets were introduced with Compute Capability 9.0 devices and are selected by using an **a** suffix in the compilation target, for example by specifying `compute_100a` or `compute_120a` as the compute target.

**Family-Specific Feature Set**: Some architecture-specific features are common to GPUs of more than one compute capability. These features are summarized in the respective “Compute Capability #.#” subsections. With a few exceptions, later generation devices with the same major compute capability are in the same family. [Table 25](#family-specific-compatibility) indicates the compatibility of family-specific targets with device compute capability, including exceptions. The family-specific feature set is a superset of the baseline feature set. Family-specific compiler targets were introduced with Compute Capability 10.0 devices and are selected by using a **f** suffix in the compilation target, for example by specifying `compute_100f` or `compute_120f` as the compute target.

All devices starting from compute capability 9.0 have a set of features that are architecture-specific. To utilize the complete set of these features on a specific GPU, the architecture-specific compiler target with the suffix **a** must be used. Additionally, starting from compute capability 10.0, there are sets of features that appear in multiple devices with different minor compute capability. These sets of instructions are called family-specific features, and the devices which share these features are said to be part of the same family. The family-specific features are a subset of the architecture-specific features that are shared by all members of that GPU family. The family-specific compiler target with the suffix **f** allows the compiler to generate code which uses this common subset of architecture-specific features.

For example:

* The `compute_100` compilation target does not allow use of architecture-specific features. This target will be compatible with all devices of compute capability 10.0 and later.
* The `compute_100f` *family-specific* compilation target allows the use of the subset of architecture-specific features that are common across the GPU family. This target will only be compatible with devices that are part of the GPU family. In this example it is compatible with devices of Compute Capability 10.0 and Compute Capability 10.3. The features available in the family-specific `compute_100f` target is a superset of the features available in the baseline `compute_100` target.
* The `compute_100a` *architecture-specific* compilation target allows use of the complete set of architecture-specific features in Compute Capability 10.0 devices. This target will only be compatible with devices of Compute Capability 10.0 and no others. The features available in the `compute_100a` target form a superset of the features available in the `compute_100f` target.

Table 25 Family-Specific Compatibility[](#family-specific-compatibility "Permalink to this table")


| Compilation Target | Compatible with Compute Capability | |
| --- | --- | --- |
| `compute_100f` | 10.0 | 10.3 |
| `compute_103f` | 10.3 [26](#family2) | |
| `compute_110f` | 11.0 [26](#family2) | |
| `compute_120f` | 12.0 | 12.1 |
| `compute_121f` | 12.1 [26](#family2) | |

26([1](#id395),[2](#id396),[3](#id397))
:   Some families only contain a single member when they are created. They may be expanded in the future to include more devices.

## 20.2. Features and Technical Specifications[](#features-and-technical-specifications "Permalink to this headline")

Table 26 Feature Support per Compute Capability[](#features-and-technical-specifications-feature-support-per-compute-capability "Permalink to this table")


| **Feature Support** | **Compute Capability** | | | | | |
| --- | --- | --- | --- | --- | --- | --- |
| (Unlisted features are supported for all compute capabilities) | 7.x | 8.x | 9.0 | 10.0 | 11.0 | 12.0 |
| Atomic functions operating on 128-bit integer values in global memory ([Atomic Functions](#atomic-functions)) | No | | Yes | | | |
| Atomic functions operating on 128-bit integer values in shared memory ([Atomic Functions](#atomic-functions)) | No | | Yes | | | |
| Atomic addition operating on float2 and float4 floating point vectors in global memory ([atomicAdd()](#atomicadd)) | No | | Yes | | | |
| Bfloat16-precision floating-point operations: addition, subtraction, multiplication, comparison, warp shuffle functions, conversion | No | | Yes | | | |
| Hardware-accelerated `memcpy_async` ([Asynchronous Data Copies using cuda::pipeline](#memcpy-async-pipeline)) | No | | Yes | | | |
| Hardware-accelerated Split Arrive/Wait Barrier ([Asynchronous Barrier](#aw-barrier)) | No | Yes | | | | |
| L2 Cache Residency Management ([Device Memory L2 Access Management](#l2-access-intro)) | No | Yes | | | | |
| DPX Instructions for Accelerated Dynamic Programming | No | | Yes | | | |
| Distributed Shared Memory | No | | Yes | | | |
| Thread Block Cluster | No | | Yes | | | |
| Tensor Memory Accelerator (TMA) unit | No | | Yes | | | |

Note that the KB and K units used in the following table correspond to 1024 bytes (i.e., a KiB) and 1024 respectively.

Table 27 Technical Specifications per Compute Capability[](#features-and-technical-specifications-technical-specifications-per-compute-capability "Permalink to this table")


|  | **Compute Capability** | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Technical Specifications | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 | 10.0 | 11.0 | 12.0 |
| Maximum number of resident grids per device (Concurrent Kernel Execution) | 128 | | | | | | | | |
| Maximum dimensionality of grid of thread blocks | 3 | | | | | | | | |
| Maximum x -dimension of a grid of thread blocks | 231-1 | | | | | | | | |
| Maximum y- or z-dimension of a grid of thread blocks | 65535 | | | | | | | | |
| Maximum dimensionality of thread block | 3 | | | | | | | | |
| Maximum x- or y-dimensionality of a block | 1024 | | | | | | | | |
| Maximum z-dimension of a block | 64 | | | | | | | | |
| Maximum number of threads per block | 1024 | | | | | | | | |
| Warp size | 32 | | | | | | | | |
| Maximum number of resident blocks per SM | 16 | 32 | 16 | | 24 | 32 | | 24 | |
| Maximum number of resident warps per SM | 32 | 64 | 48 | | | 64 | | 48 | |
| Maximum number of resident threads per SM | 1024 | 2048 | 1536 | | | 2048 | | 1536 | |
| Number of 32-bit registers per SM | 64 K | | | | | | | | |
| Maximum number of 32-bit registers per thread block | 64 K | | | | | | | | |
| Maximum number of 32-bit registers per thread | 255 | | | | | | | | |
| Maximum amount of shared memory per SM | 64 KB | 164 KB | 100 KB | 164 KB | 100 KB | 228 KB | | | 100 KB |
| Maximum amount of shared memory per thread block [27](#fn33) | 64 KB | 163 KB | 99 KB | 163 KB | 99 KB | 227 KB | | | 99 KB |
| Number of shared memory banks | 32 | | | | | | | | |
| Maximum amount of local memory per thread | 512 KB | | | | | | | | |
| Constant memory size | 64 KB | | | | | | | | |
| Cache working set per SM for constant memory | 8 KB | | | | | | | | |
| Cache working set per SM for texture memory | 32 or 64 KB | 28 KB ~ 192 KB | 28 KB ~ 128 KB | 28 KB ~ 192 KB | 28 KB ~ 128 KB | 28 KB ~ 256 KB | | | 28 KB ~ 128 KB |
| Maximum width for a 1D texture object using a CUDA array | 131072 | | | | | | | | |
| Maximum width for a 1D texture object using linear memory | 228 | | | | | | | | |
| Maximum width and number of layers for a 1D layered texture object | 32768 x 2048 | | | | | | | | |
| Maximum width and height for a 2D texture object using a CUDA array | 131072 x 65536 | | | | | | | | |
| Maximum width and height for a 2D texture object using linear memory | 131072 x 65000 | | | | | | | | |
| Maximum width and height for a 2D texture object using a CUDA array supporting texture gather | 32768 x 32768 | | | | | | | | |
| Maximum width, height, and number of layers for a 2D layered texture object | 32768 x 32768 x 2048 | | | | | | | | |
| Maximum width, height, and depth for a 3D texture object using to a CUDA array | 16384 x 16384 x 16384 | | | | | | | | |
| Maximum width (and height) for a cubemap texture object | 32768 | | | | | | | | |
| Maximum width (and height) and number of layers for a cubemap layered texture object | 32768 x 2046 | | | | | | | | |
| Maximum number of textures that can be bound to a kernel | 256 | | | | | | | | |
| Maximum width for a 1D surface object using a CUDA array | 32768 | | | | | | | | |
| Maximum width and number of layers for a 1D layered surface object | 32768 x 2048 | | | | | | | | |
| Maximum width and height for a 2D surface object using a CUDA array | 131072 x 65536 | | | | | | | | |
| Maximum width, height, and number of layers for a 2D layered surface object | 32768 x 32768 x 1048 | | | | | | | | |
| Maximum width, height, and depth for a 3D surface object using a CUDA array | 16384 x 16384 x 16384 | | | | | | | | |
| Maximum width (and height) for a cubemap surface object using a CUDA array | 32768 | | | | | | | | |
| Maximum width (and height) and number of layers for a cubemap layered surface object | 32768 x 2046 | | | | | | | | |
| Maximum number of surfaces that can use a kernel | 32 | | | | | | | | |

## 20.3. Floating-Point Standard[](#floating-point-standard "Permalink to this headline")

All compute devices follow the IEEE 754-2008 standard for binary floating-point arithmetic with the following deviations:

* There is no dynamically configurable rounding mode; however, most of the operations support multiple IEEE rounding modes, exposed via device intrinsics.
* There is no mechanism for detecting that a floating-point exception has occurred and all operations behave as if the IEEE-754 exceptions are always masked, and deliver the masked response as defined by IEEE-754 if there is an exceptional event. For the same reason, while SNaN encodings are supported, they are not signaling and are handled as quiet.
* The result of a single-precision floating-point operation involving one or more input NaNs is the quiet NaN of bit pattern 0x7fffffff.
* Double-precision floating-point absolute value and negation are not compliant with IEEE-754 with respect to NaNs; these are passed through unchanged.

Code must be compiled with `-ftz=false`, `-prec-div=true`, and `-prec-sqrt=true` to ensure IEEE compliance (this is the default setting; see the `nvcc` user manual for description of these compilation flags).

Regardless of the setting of the compiler flag `-ftz`,

* atomic single-precision floating-point adds on global memory always operate in flush-to-zero mode, i.e., behave equivalent to `FADD.F32.FTZ.RN`,
* atomic single-precision floating-point adds on shared memory always operate with denormal support, i.e., behave equivalent to `FADD.F32.RN`.

In accordance to the IEEE-754R standard, if one of the input parameters to `fminf()`, `fmin()`, `fmaxf()`, or `fmax()` is NaN, but not the other, the result is the non-NaN parameter.

The conversion of a floating-point value to an integer value in the case where the floating-point value falls outside the range of the integer format is left undefined by IEEE-754. For compute devices, the behavior is to clamp to the end of the supported range. This is unlike the x86 architecture behavior.

The behavior of integer division by zero and integer overflow is left undefined by IEEE-754. For compute devices, there is no mechanism for detecting that such integer operation exceptions have occurred. Integer division by zero yields an unspecified, machine-specific value.

<https://developer.nvidia.com/content/precision-performance-floating-point-and-ieee-754-compliance-nvidia-gpus> includes more information on the floating point accuracy and compliance of NVIDIA GPUs.

## 20.4. Compute Capability 5.x[](#compute-capability-5-x "Permalink to this headline")

### 20.4.1. Architecture[](#architecture "Permalink to this headline")

An SM consists of:

* 128 CUDA cores for arithmetic operations (see [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions) for throughputs of arithmetic operations),
* 32 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

When an SM is given warps to execute, it first distributes them among the four schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified L1/texture cache of 24 KB used to cache reads from global memory,
* 64 KB of shared memory for devices of compute capability 5.0 or 96 KB of shared memory for devices of compute capability 5.2.

The unified L1/texture cache is also used by the texture unit that implements the various addressing modes and data filtering mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

There is also an L2 cache shared by all SMs that is used to cache accesses to local or global memory, including temporary register spills. Applications may query the L2 cache size by checking the `l2CacheSize` device property (see [Device Enumeration](#device-enumeration)).

The cache behavior (e.g., whether reads are cached in both the unified L1/texture cache and L2 or in L2 only) can be partially configured on a per-access basis using modifiers to the load instruction.

### 20.4.2. Global Memory[](#global-memory-5-x "Permalink to this headline")

Global memory accesses are always cached in L2.

Data that is read-only for the entire lifetime of the kernel can also be cached in the unified L1/texture cache described in the previous section by reading it using the `__ldg()` function (see [Read-Only Data Cache Load Function](#ldg-function)). When the compiler detects that the read-only condition is satisfied for some data, it will use `__ldg()` to read it. The compiler might not always be able to detect that the read-only condition is satisfied for some data. Marking pointers used for loading such data with both the `const` and `__restrict__` qualifiers increases the likelihood that the compiler will detect the read-only condition.

Data that is not read-only for the entire lifetime of the kernel cannot be cached in the unified L1/texture cache for devices of compute capability 5.0. For devices of compute capability 5.2, it is, by default, not cached in the unified L1/texture cache, but caching may be enabled using the following mechanisms:

* Perform the read using inline assembly with the appropriate modifier as described in the PTX reference manual;
* Compile with the `-Xptxas -dlcm=ca` compilation flag, in which case all reads are cached, except reads that are performed using inline assembly with a modifier that disables caching;
* Compile with the `-Xptxas -fscm=ca` compilation flag, in which case all reads are cached, including reads that are performed using inline assembly regardless of the modifier used.

When caching is enabled using one of the three mechanisms listed above, devices of compute capability 5.2 will cache global memory reads in the unified L1/texture cache for all kernel launches except for the kernel launches for which thread blocks consume too much of the SM’s register file. These exceptions are reported by the profiler.

### 20.4.3. Shared Memory[](#shared-memory-5-x "Permalink to this headline")

Shared memory has 32 banks that are organized such that successive 32-bit words map to successive banks. Each bank has a bandwidth of 32 bits per clock cycle.

A shared memory request for a warp does not generate a bank conflict between two threads that access any address within the same 32-bit word (even though the two addresses fall in the same bank). In that case, for read accesses, the word is broadcast to the requesting threads and for write accesses, each address is written by only one of the threads (which thread performs the write is undefined).

[Figure 39](#shared-memory-5-x-examples-of-strided-shared-memory-accesses) shows some examples of strided access.

[Figure 40](#shared-memory-5-x-examples-of-irregular-shared-memory-accesses) shows some examples of memory read accesses that involve the broadcast mechanism.

![Strided Shared Memory Accesses in 32 bit bank size mode.](_images/examples-of-strided-shared-memory-accesses.png)


Figure 39 Strided Shared Memory Accesses in 32 bit bank size mode.[](#shared-memory-5-x-examples-of-strided-shared-memory-accesses "Permalink to this image")

Left
:   Linear addressing with a stride of one 32-bit word (no bank conflict).

Middle
:   Linear addressing with a stride of two 32-bit words (two-way bank conflict).

Right
:   Linear addressing with a stride of three 32-bit words (no bank conflict).


![Irregular Shared Memory Accesses.](_images/examples-of-irregular-shared-memory-accesses.png)


Figure 40 Irregular Shared Memory Accesses.[](#shared-memory-5-x-examples-of-irregular-shared-memory-accesses "Permalink to this image")

Left
:   Conflict-free access via random permutation.

Middle
:   Conflict-free access since threads 3, 4, 6, 7, and 9 access the same word within bank 5.

Right
:   Conflict-free broadcast access (threads access the same word within a bank).

## 20.5. Compute Capability 6.x[](#compute-capability-6-x "Permalink to this headline")

### 20.5.1. Architecture[](#architecture-6-x "Permalink to this headline")

An SM consists of:

* 64 (compute capability 6.0) or 128 (6.1 and 6.2) CUDA cores for arithmetic operations,
* 16 (6.0) or 32 (6.1 and 6.2) special function units for single-precision floating-point transcendental functions,
* 2 (6.0) or 4 (6.1 and 6.2) warp schedulers.

When an SM is given warps to execute, it first distributes them among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified L1/texture cache for reads from global memory of size 24 KB (6.0 and 6.2) or 48 KB (6.1),
* a shared memory of size 64 KB (6.0 and 6.2) or 96 KB (6.1).

The unified L1/texture cache is also used by the texture unit that implements the various addressing modes and data filtering mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

There is also an L2 cache shared by all SMs that is used to cache accesses to local or global memory, including temporary register spills. Applications may query the L2 cache size by checking the `l2CacheSize` device property (see [Device Enumeration](#device-enumeration)).

The cache behavior (for example, whether reads are cached in both the unified L1/texture cache and L2 or in L2 only) can be partially configured on a per-access basis using modifiers to the load instruction.

### 20.5.2. Global Memory[](#global-memory-6-x "Permalink to this headline")

Global memory behaves the same way as in devices of compute capability 5.x (See [Global Memory](#global-memory-5-x)).

### 20.5.3. Shared Memory[](#shared-memory-6-x "Permalink to this headline")

Shared memory behaves the same way as in devices of compute capability 5.x (See [Shared Memory](#shared-memory-5-x)).

## 20.6. Compute Capability 7.x[](#compute-capability-7-x "Permalink to this headline")

### 20.6.1. Architecture[](#architecture-7-x "Permalink to this headline")

An SM consists of:

* 64 FP32 cores for single-precision arithmetic operations,
* 32 FP64 cores for double-precision arithmetic operations,[28](#fn35)
* 64 INT32 cores for integer math,
* 8 mixed-precision Tensor Cores for deep learning matrix arithmetic
* 16 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified data cache and shared memory with a total size of 128 KB (*Volta*) or 96 KB (*Turing*).

Shared memory is partitioned out of unified data cache, and can be configured to various sizes (See [Shared Memory](#shared-memory-7-x).) The remaining data cache serves as an L1 cache and is also used by the texture unit that implements the various addressing and data filtering modes mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

### 20.6.2. Independent Thread Scheduling[](#independent-thread-scheduling "Permalink to this headline")

The **NVIDIA Volta GPU Architecture** introduces *Independent Thread Scheduling* among threads in a warp, enabling intra-warp synchronization patterns previously unavailable and simplifying code changes when porting CPU code. However, this can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity of previous hardware architectures.

Below are code patterns of concern and suggested corrective actions for Volta-safe code.

1. For applications using warp intrinsics (`__shfl*`, `__any`, `__all`, `__ballot`), it is necessary that developers port their code to the new, safe, synchronizing counterpart, with the `*_sync` suffix. The new warp intrinsics take in a mask of threads that explicitly define which lanes (threads of a warp) must participate in the warp intrinsic. See [Warp Vote Functions](#warp-vote-functions) and [Warp Shuffle Functions](#warp-shuffle-functions) for details.

Since the intrinsics are available with CUDA 9.0+, (if necessary) code can be executed conditionally with the following preprocessor macro:

```
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
// *_sync intrinsic
#endif
```

These intrinsics are available on all architectures, not just **NVIDIA Volta GPU Architecture** or **NVIDIA Turing GPU Architecture**, and in most cases a single code-base will suffice for all architectures. Note, however, that for *Pascal* and earlier architectures, all threads in mask must execute the same warp intrinsic instruction in convergence, and the union of all values in mask must be equal to the warp’s active mask. The following code pattern is valid on **NVIDIA Volta GPU Architecture**, but not on *Pascal* or earlier architectures.

> ```
> if (tid % warpSize < 16) {
>     ...
>     float swapped = __shfl_xor_sync(0xffffffff, val, 16);
>     ...
> } else {
>     ...
>     float swapped = __shfl_xor_sync(0xffffffff, val, 16);
>     ...
> }
> ```

The replacement for `__ballot(1)` is `__activemask()`. Note that threads within a warp can diverge even within a single code path. As a result, `__activemask()` and `__ballot(1)` may return only a subset of the threads on the current code path. The following invalid code example sets bit `i` of `output` to 1 when `data[i]` is greater than `threshold`. `__activemask()` is used in an attempt to enable cases where `dataLen` is not a multiple of 32.

> ```
> // Sets bit in output[] to 1 if the correspond element in data[i]
> // is greater than 'threshold', using 32 threads in a warp.
>
> for (int i = warpLane; i < dataLen; i += warpSize) {
>     unsigned active = __activemask();
>     unsigned bitPack = __ballot_sync(active, data[i] > threshold);
>     if (warpLane == 0) {
>         output[i / 32] = bitPack;
>     }
> }
> ```

This code is invalid because CUDA does not guarantee that the warp will diverge ONLY at the loop condition. When divergence happens for other reasons, conflicting results will be computed for the same 32-bit output element by different subsets of threads in the warp. A correct code might use a non-divergent loop condition together with `__ballot_sync()` to safely enumerate the set of threads in the warp participating in the threshold calculation as follows.

> ```
> for (int i = warpLane; i - warpLane < dataLen; i += warpSize) {
>     unsigned active = __ballot_sync(0xFFFFFFFF, i < dataLen);
>     if (i < dataLen) {
>         unsigned bitPack = __ballot_sync(active, data[i] > threshold);
>         if (warpLane == 0) {
>             output[i / 32] = bitPack;
>         }
>     }
> }
> ```

[Discovery Pattern](#discovery-pattern-cg) demonstrates a valid use case for `__activemask()`.

1. If applications have warp-synchronous codes, they will need to insert the new `__syncwarp()` warp-wide barrier synchronization instruction between any steps where data is exchanged between threads via global or shared memory. Assumptions that code is executed in lockstep or that reads/writes from separate threads are visible across a warp without synchronization are invalid.

   ```
   __shared__ float s_buff[BLOCK_SIZE];
   s_buff[tid] = val;
   __syncthreads();

   // Inter-warp reduction
   for (int i = BLOCK_SIZE / 2; i >= 32; i /= 2) {
       if (tid < i) {
           s_buff[tid] += s_buff[tid+i];
       }
       __syncthreads();
   }

   // Intra-warp reduction
   // Butterfly reduction simplifies syncwarp mask
   if (tid < 32) {
       float temp;
       temp = s_buff[tid ^ 16]; __syncwarp();
       s_buff[tid] += temp;     __syncwarp();
       temp = s_buff[tid ^ 8];  __syncwarp();
       s_buff[tid] += temp;     __syncwarp();
       temp = s_buff[tid ^ 4];  __syncwarp();
       s_buff[tid] += temp;     __syncwarp();
       temp = s_buff[tid ^ 2];  __syncwarp();
       s_buff[tid] += temp;     __syncwarp();
   }

   if (tid == 0) {
       *output = s_buff[0] + s_buff[1];
   }
   __syncthreads();
   ```
2. Although `__syncthreads()` has been consistently documented as synchronizing all threads in the thread block, *Pascal* and prior architectures could only enforce synchronization at the warp level. In certain cases, this allowed a barrier to succeed without being executed by every thread as long as at least some thread in every warp reached the barrier. Starting with **NVIDIA Volta GPU Architecture**, the CUDA built-in `__syncthreads()` and PTX instruction `bar.sync` (and their derivatives) are enforced per thread and thus will not succeed until reached by all non-exited threads in the block. Code exploiting the previous behavior will likely deadlock and must be modified to ensure that all non-exited threads reach the barrier.

The `racecheck` and `synccheck` tools provided by `compute-saniter` can help with locating violations.

To aid migration while implementing the above-mentioned corrective actions, developers can opt-in to the Pascal scheduling model that does not support independent thread scheduling. See [Application Compatibility](#application-compatibility) for details.

### 20.6.3. Global Memory[](#global-memory-7-x "Permalink to this headline")

Global memory behaves the same way as in devices of compute capability 5.x (See [Global Memory](#global-memory-5-x)).

### 20.6.4. Shared Memory[](#shared-memory-7-x "Permalink to this headline")

The amount of the unified data cache reserved for shared memory is configurable on a per kernel basis. For the *Volta* architecture (compute capability 7.0), the unified data cache has a size of 128 KB, and the shared memory capacity can be set to 0, 8, 16, 32, 64 or 96 KB. For the *Turing* architecture (compute capability 7.5), the unified data cache has a size of 96 KB, and the shared memory capacity can be set to either 32 KB or 64 KB. Unlike Kepler, the driver automatically configures the shared memory capacity for each kernel to avoid shared memory occupancy bottlenecks while also allowing concurrent execution with already launched kernels where possible. In most cases, the driver’s default behavior should provide optimal performance.

Because the driver is not always aware of the full workload, it is sometimes useful for applications to provide additional hints regarding the desired shared memory configuration. For example, a kernel with little or no shared memory use may request a larger carveout in order to encourage concurrent execution with later kernels that require more shared memory. The new `cudaFuncSetAttribute()` API allows applications to set a preferred shared memory capacity, or `carveout`, as a percentage of the maximum supported shared memory capacity (96 KB for *Volta*, and 64 KB for *Turing*).

`cudaFuncSetAttribute()` relaxes enforcement of the preferred shared capacity compared to the legacy `cudaFuncSetCacheConfig()` API introduced with Kepler. The legacy API treated shared memory capacities as hard requirements for kernel launch. As a result, interleaving kernels with different shared memory configurations would needlessly serialize launches behind shared memory reconfigurations. With the new API, the carveout is treated as a hint. The driver may choose a different configuration if required to execute the function or to avoid thrashing.

```
// Device code
__global__ void MyKernel(...)
{
    __shared__ float buffer[BLOCK_DIM];
    ...
}

// Host code
int carveout = 50; // prefer shared memory capacity 50% of maximum
// Named Carveout Values:
// carveout = cudaSharedmemCarveoutDefault;   //  (-1)
// carveout = cudaSharedmemCarveoutMaxL1;     //   (0)
// carveout = cudaSharedmemCarveoutMaxShared; // (100)
cudaFuncSetAttribute(MyKernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
MyKernel <<<gridDim, BLOCK_DIM>>>(...);
```

In addition to an integer percentage, several convenience enums are provided as listed in the code comments above. Where a chosen integer percentage does not map exactly to a supported capacity (SM 7.0 devices support shared capacities of 0, 8, 16, 32, 64, or 96 KB), the next larger capacity is used. For instance, in the example above, 50% of the 96 KB maximum is 48 KB, which is not a supported shared memory capacity. Thus, the preference is rounded up to 64 KB.

Compute capability 7.x devices allow a single thread block to address the full capacity of shared memory: 96 KB on *Volta*, 64 KB on *Turing*. Kernels relying on shared memory allocations over 48 KB per block are architecture-specific, as such they must use dynamic shared memory (rather than statically sized arrays) and require an explicit opt-in using `cudaFuncSetAttribute()` as follows.

```
// Device code
__global__ void MyKernel(...)
{
    extern __shared__ float buffer[];
    ...
}

// Host code
int maxbytes = 98304; // 96 KB
cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
MyKernel <<<gridDim, blockDim, maxbytes>>>(...);
```

Otherwise, shared memory behaves the same way as for devices of compute capability 5.x (See [Shared Memory](#shared-memory-5-x)).

## 20.7. Compute Capability 8.x[](#compute-capability-8-x "Permalink to this headline")

### 20.7.1. Architecture[](#architecture-8-x "Permalink to this headline")

A Streaming Multiprocessor (SM) consists of:

* 64 FP32 cores for single-precision arithmetic operations in devices of compute capability 8.0 and 128 FP32 cores in devices of compute capability 8.6, 8.7 and 8.9,
* 32 FP64 cores for double-precision arithmetic operations in devices of compute capability 8.0 and 2 FP64 cores in devices of compute capability 8.6, 8.7 and 8.9
* 64 INT32 cores for integer math,
* 4 mixed-precision Third-Generation Tensor Cores supporting half-precision (fp16), `__nv_bfloat16`, `tf32`, sub-byte and double precision (fp64) matrix arithmetic for compute capabilities 8.0, 8.6 and 8.7 (see [Warp Matrix Functions](#wmma) for details),
* 4 mixed-precision Fourth-Generation Tensor Cores supporting `fp8`, `fp16`, `__nv_bfloat16`, `tf32`, sub-byte and `fp64` for compute capability 8.9 (see [Warp Matrix Functions](#wmma) for details),
* 16 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified data cache and shared memory with a total size of 192 KB for devices of compute capability 8.0 and 8.7 (1.5x *Volta*’s 128 KB capacity) and 128 KB for devices of compute capabilities 8.6 and 8.9.

Shared memory is partitioned out of the unified data cache, and can be configured to various sizes (see [Shared Memory](#shared-memory-8-x)). The remaining data cache serves as an L1 cache and is also used by the texture unit that implements the various addressing and data filtering modes mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

### 20.7.2. Global Memory[](#global-memory-8-x "Permalink to this headline")

Global memory behaves the same way as for devices of compute capability 5.x (See [Global Memory](#global-memory-5-x)).

### 20.7.3. Shared Memory[](#shared-memory-8-x "Permalink to this headline")

Similar to the [Volta architecture](#architecture-7-x), the amount of the unified data cache reserved for shared memory is configurable on a per kernel basis. For the **NVIDIA Ampere GPU Architecture**, the unified data cache has a size of 192 KB for devices of compute capability 8.0 and 8.7 and 128 KB for devices of compute capabilities 8.6 and 8.9. The shared memory capacity can be set to 0, 8, 16, 32, 64, 100, 132 or 164 KB for devices of compute capability 8.0 and 8.7, and to 0, 8, 16, 32, 64 or 100 KB for devices of compute capabilities 8.6 and 8.9.

An application can set the `carveout`, i.e., the preferred shared memory capacity, with the `cudaFuncSetAttribute()`.

```
cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
```

The API can specify the carveout either as an integer percentage of the maximum supported shared memory capacity of 164 KB for devices of compute capability 8.0 and 8.7 and 100 KB for devices of compute capabilities 8.6 and 8.9 respectively, or as one of the following values: `{cudaSharedmemCarveoutDefault`, `cudaSharedmemCarveoutMaxL1`, or `cudaSharedmemCarveoutMaxShared`. When using a percentage, the carveout is rounded up to the nearest supported shared memory capacity. For example, for devices of compute capability 8.0, 50% will map to a 100 KB carveout instead of an 82 KB one. Setting the `cudaFuncAttributePreferredSharedMemoryCarveout` is considered a hint by the driver; the driver may choose a different configuration, if needed.

Devices of compute capability 8.0 and 8.7 allow a single thread block to address up to 163 KB of shared memory, while devices of compute capabilities 8.6 and 8.9 allow up to 99 KB of shared memory. Kernels relying on shared memory allocations over 48 KB per block are architecture-specific, and must use dynamic shared memory rather than statically sized shared memory arrays. These kernels require an explicit opt-in by using `cudaFuncSetAttribute()` to set the `cudaFuncAttributeMaxDynamicSharedMemorySize`; see [Shared Memory](#shared-memory-7-x) for the **NVIDIA Volta GPU Architecture**.

Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory partition available per SM. The 1 KB of shared memory not made available to a thread block is reserved for system use.

## 20.8. Compute Capability 9.0[](#compute-capability-9-0 "Permalink to this headline")

### 20.8.1. Architecture[](#architecture-9-0 "Permalink to this headline")

A Streaming Multiprocessor (SM) consists of:

* 128 FP32 cores for single-precision arithmetic operations,
* 64 FP64 cores for double-precision arithmetic operations,
* 64 INT32 cores for integer math,
* 4 mixed-precision fourth-generation Tensor Cores supporting the new `FP8` input type in either `E4M3` or `E5M2` for exponent (E) and mantissa (M), half-precision (fp16), `__nv_bfloat16`, `tf32`, INT8 and double precision (fp64) matrix arithmetic (see [Warp Matrix Functions](#wmma) for details) with sparsity support,
* 16 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified data cache and shared memory with a total size of 256 KB for devices of compute capability 9.0 (1.33x **NVIDIA Ampere GPU Architecture’s** 192 KB capacity).

Shared memory is partitioned out of the unified data cache, and can be configured to various sizes (see [Shared Memory](#shared-memory-9-0)). The remaining data cache serves as an L1 cache and is also used by the texture unit that implements the various addressing and data filtering modes mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

### 20.8.2. Global Memory[](#global-memory-9-0 "Permalink to this headline")

Global memory behaves the same way as for devices of compute capability 5.x (See [Global Memory](#global-memory-5-x)).

### 20.8.3. Shared Memory[](#shared-memory-9-0 "Permalink to this headline")

Similar to the [NVIDIA Ampere GPU architecture](#architecture-8-x), the amount of the unified data cache reserved for shared memory is configurable on a per kernel basis. For the *NVIDIA H100 Tensor Core GPU architecture*, the unified data cache has a size of 256 KB for devices of compute capability 9.0. The shared memory capacity can be set to 0, 8, 16, 32, 64, 100, 132, 164, 196 or 228 KB.

As with the [NVIDIA Ampere GPU architecture](#shared-memory-8-x), an application can configure its preferred shared memory capacity, i.e., the `carveout`. Devices of compute capability 9.0 allow a single thread block to address up to 227 KB of shared memory. Kernels relying on shared memory allocations over 48 KB per block are architecture-specific, and must use dynamic shared memory rather than statically sized shared memory arrays. These kernels require an explicit opt-in by using `cudaFuncSetAttribute()` to set the `cudaFuncAttributeMaxDynamicSharedMemorySize`; see [Shared Memory](#shared-memory-7-x) for the **NVIDIA Volta GPU Architecture**.

Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory partition available per SM. The 1 KB of shared memory not made available to a thread block is reserved for system use.

### 20.8.4. Features Accelerating Specialized Computations[](#features-accelerating-specialized-computations "Permalink to this headline")

The NVIDIA Hopper GPU architecture includes features to accelerate matrix multiply-accumulate (MMA) computations with:

* asynchronous execution of MMA instructions
* MMA instructions acting on large matrices spanning a warp-group
* dynamic reassignment of register capacity among warp-groups to support even larger matrices, and
* operand matrices accessed directly from shared memory

See the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set) for more details.

This feature set is only available within the CUDA compilation toolchain through inline PTX.

It is strongly recommended that applications utilize this complex feature set through CUDA-X libraries such as cuBLAS, cuDNN, or cuFFT.

It is strongly recommended that device kernels utilize this complex feature set through [CUTLASS](https://github.com/NVIDIA/cutlass), a collection of CUDA C++ template abstractions for implementing high-performance matrix-multiplication (GEMM) and related computations at all levels and scales within CUDA.

## 20.9. Compute Capability 10.0[](#compute-capability-10-0 "Permalink to this headline")

### 20.9.1. Architecture[](#architecture-10-0 "Permalink to this headline")

A Streaming Multiprocessor (SM) consists of:

* 128 FP32 cores for single-precision arithmetic operations,
* 64 FP64 cores for double-precision arithmetic operations,
* 64 INT32 cores for integer math,
* 4 mixed-precision fifth-generation Tensor Cores supporting `FP8` input type in either `E4M3` or `E5M2` for exponent (E) and mantissa (M), half-precision (fp16), `__nv_bfloat16`, `tf32`, INT8 and double precision (fp64) matrix arithmetic (see [Warp Matrix Functions](#wmma) for details) with sparsity support,
* 16 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified data cache and shared memory with a total size of 256 KB for devices of compute capability 10.0

Shared memory is partitioned out of the unified data cache, and can be configured to various sizes (see [Shared Memory](#shared-memory-10-0)). The remaining data cache serves as an L1 cache and is also used by the texture unit that implements the various addressing and data filtering modes mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

### 20.9.2. Global Memory[](#global-memory-10-0 "Permalink to this headline")

Global memory behaves the same way as for devices of compute capability 5.x (See [Global Memory](#global-memory-5-x)).

### 20.9.3. Shared Memory[](#shared-memory-10-0 "Permalink to this headline")

The amount of the unified data cache reserved for shared memory is configurable on a per kernel basis and is identical to [compute capability 9.0](#shared-memory-9-0). The unified data cache has a size of 256 KB for devices of compute capability 10.0. The shared memory capacity can be set to 0, 8, 16, 32, 64, 100, 132, 164, 196 or 228 KB.

As with the [NVIDIA Ampere GPU architecture](#shared-memory-8-x), an application can configure its preferred shared memory capacity, i.e., the `carveout`. Devices of compute capability 10.0 allow a single thread block to address up to 227 KB of shared memory. Kernels relying on shared memory allocations over 48 KB per block are architecture-specific, and must use dynamic shared memory rather than statically sized shared memory arrays. These kernels require an explicit opt-in by using `cudaFuncSetAttribute()` to set the `cudaFuncAttributeMaxDynamicSharedMemorySize`; see [Shared Memory](#shared-memory-7-x) for the Volta architecture.

Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory partition available per SM. The 1 KB of shared memory not made available to a thread block is reserved for system use.

### 20.9.4. Features Accelerating Specialized Computations[](#features-accelerating-specialized-computations-10-0 "Permalink to this headline")

The NVIDIA Blackwell GPU architecture extends features to accelerate matrix multiply-accumulate (MMA) from the NVIDIA Hopper GPU architecture.

See the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set) for more details.

This feature set is only available within the CUDA compilation toolchain through inline PTX.

It is strongly recommended that applications utilize this complex feature set through CUDA-X libraries such as cuBLAS, cuDNN, or cuFFT.

It is strongly recommended that device kernels utilize this complex feature set through [CUTLASS](https://github.com/NVIDIA/cutlass), a collection of CUDA C++ template abstractions for implementing high-performance matrix-multiplication (GEMM) and related computations at all levels and scales within CUDA.

## 20.10. Compute Capability 12.0[](#compute-capability-12-0 "Permalink to this headline")

### 20.10.1. Architecture[](#architecture-12-0 "Permalink to this headline")

A Streaming Multiprocessor (SM) consists of:

* 128 FP32 cores for single-precision arithmetic operations,
* 2 FP64 cores for double-precision arithmetic operations,
* 64 INT32 cores for integer math,
* Mixed-precision fifth-generation Tensor Core(s) supporting `FP8` input type in either `E4M3` or `E5M2` for exponent (E) and mantissa (M), half-precision (fp16), `__nv_bfloat16`, `tf32`, INT8 and double precision (fp64) matrix arithmetic (see [Warp Matrix Functions](#wmma) for details) with sparsity support,
* 16 special function units for single-precision floating-point transcendental functions,
* 4 warp schedulers.

An SM statically distributes its warps among its schedulers. Then, at every instruction issue time, each scheduler issues one instruction for one of its assigned warps that is ready to execute, if any.

An SM has:

* a read-only constant cache that is shared by all functional units and speeds up reads from the constant memory space, which resides in device memory,
* a unified data cache and shared memory with a total size of 100 KB for devices of compute capability 12.0

Shared memory is partitioned out of the unified data cache, and can be configured to various sizes (see [Shared Memory](#shared-memory-12-0)). The remaining data cache serves as an L1 cache and is also used by the texture unit that implements the various addressing and data filtering modes mentioned in [Texture and Surface Memory](#texture-and-surface-memory).

### 20.10.2. Global Memory[](#global-memory-12-0 "Permalink to this headline")

Global memory behaves the same way as for devices of compute capability 5.x (See [Global Memory](#global-memory-5-x)).

### 20.10.3. Shared Memory[](#shared-memory-12-0 "Permalink to this headline")

The amount of the unified data cache reserved for shared memory is configurable on a per kernel basis. The unified data cache has a size of 100 KB for devices of compute capability 12.0. The shared memory capacity can be set to 0, 8, 16, 32, 64, or 100 KB.

As with the [NVIDIA Ampere GPU architecture](#shared-memory-8-x), an application can configure its preferred shared memory capacity, i.e., the `carveout`. Devices of compute capability 12.0 allow a single thread block to address up to 99 KB of shared memory. Kernels relying on shared memory allocations over 48 KB per block are architecture-specific, and must use dynamic shared memory rather than statically sized shared memory arrays. These kernels require an explicit opt-in by using `cudaFuncSetAttribute()` to set the `cudaFuncAttributeMaxDynamicSharedMemorySize`; see [Shared Memory](#shared-memory-7-x) for the Volta architecture.

Note that the maximum amount of shared memory per thread block is smaller than the maximum shared memory partition available per SM. The 1 KB of shared memory not made available to a thread block is reserved for system use.

### 20.10.4. Features Accelerating Specialized Computations[](#features-accelerating-specialized-computations-12-0 "Permalink to this headline")

The NVIDIA Blackwell GPU architecture extends features to accelerate matrix multiply-accumulate (MMA) from the NVIDIA Hopper GPU architecture.

See the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set) for more details.

This feature set is only available within the CUDA compilation toolchain through inline PTX.

It is strongly recommended that applications utilize this complex feature set through CUDA-X libraries such as cuBLAS, cuDNN, or cuFFT.

It is strongly recommended that device kernels utilize this complex feature set through [CUTLASS](https://github.com/NVIDIA/cutlass), a collection of CUDA C++ template abstractions for implementing high-performance matrix-multiplication (GEMM) and related computations at all levels and scales within CUDA.

[27](#id399)
:   above 48 KB requires dynamic shared memory

[28](#id410)
:   2 FP64 cores for double-precision arithmetic operations for devices of compute capabilities 7.5
