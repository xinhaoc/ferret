# 18. C++ Language Support[](#c-language-support "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

As described in [Compilation with NVCC](#compilation-with-nvcc), CUDA source files compiled with `nvcc` can include a mix of host code and device code. The CUDA front-end compiler aims to emulate the host compiler behavior with respect to C++ input code. The input source code is processed according to the C++ ISO/IEC 14882:2003, C++ ISO/IEC 14882:2011, C++ ISO/IEC 14882:2014 or C++ ISO/IEC 14882:2017 specifications, and the CUDA front-end compiler aims to emulate any host compiler divergences from the ISO specification. In addition, the supported language is extended with CUDA-specific constructs described in this document [6](#fn13), and is subject to the restrictions described below.

[C++11 Language Features](#cpp11-language-features), [C++14 Language Features](#cpp14-language-features) and [C++17 Language Features](#cpp17-language-features) provide support matrices for the C++11, C++14, C++17 and C++20 features, respectively. [Restrictions](#language-restrictions) lists the language restrictions. [Polymorphic Function Wrappers](#polymorphic-function-wrappers) and [Extended Lambdas](#extended-lambda) describe additional features. [Code Samples](#code-samples) gives code samples.

## 18.1. C++11 Language Features[](#c-11-language-features "Permalink to this headline")

The following table lists new language features that have been accepted into the C++11 standard. The “Proposal” column provides a link to the ISO C++ committee proposal that describes the feature, while the “Available in nvcc (device code)” column indicates the first version of nvcc that contains an implementation of this feature (if it has been implemented) for device code.

Table 23 C++11 Language Features[](#id483 "Permalink to this table")


| Language Feature | C++11 Proposal | Available in nvcc (device code) |
| --- | --- | --- |
| Rvalue references | [N2118](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html) | 7.0 |
| Rvalue references for `*this` | [N2439](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm) | 7.0 |
| Initialization of class objects by rvalues | [N1610](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html) | 7.0 |
| Non-static data member initializers | [N2756](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm) | 7.0 |
| Variadic templates | [N2242](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf) | 7.0 |
| Extending variadic template template parameters | [N2555](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf) | 7.0 |
| Initializer lists | [N2672](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm) | 7.0 |
| Static assertions | [N1720](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html) | 7.0 |
| `auto`-typed variables | [N1984](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf) | 7.0 |
| Multi-declarator `auto` | [N1737](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf) | 7.0 |
| Removal of auto as a storage-class specifier | [N2546](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm) | 7.0 |
| New function declarator syntax | [N2541](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm) | 7.0 |
| Lambda expressions | [N2927](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf) | 7.0 |
| Declared type of an expression | [N2343](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf) | 7.0 |
| Incomplete return types | [N3276](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf) | 7.0 |
| Right angle brackets | [N1757](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html) | 7.0 |
| Default template arguments for function templates | [DR226](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226) | 7.0 |
| Solving the SFINAE problem for expressions | [DR339](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html) | 7.0 |
| Alias templates | [N2258](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf) | 7.0 |
| Extern templates | [N1987](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm) | 7.0 |
| Null pointer constant | [N2431](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf) | 7.0 |
| Strongly-typed enums | [N2347](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf) | 7.0 |
| Forward declarations for enums | [N2764](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf) [DR1206](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206) | 7.0 |
| Standardized attribute syntax | [N2761](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf) | 7.0 |
| Generalized constant expressions | [N2235](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf) | 7.0 |
| Alignment support | [N2341](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf) | 7.0 |
| Conditionally-support behavior | [N1627](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf) | 7.0 |
| Changing undefined behavior into diagnosable errors | [N1727](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf) | 7.0 |
| Delegating constructors | [N1986](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf) | 7.0 |
| Inheriting constructors | [N2540](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm) | 7.0 |
| Explicit conversion operators | [N2437](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf) | 7.0 |
| New character types | [N2249](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html) | 7.0 |
| Unicode string literals | [N2442](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm) | 7.0 |
| Raw string literals | [N2442](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm) | 7.0 |
| Universal character names in literals | [N2170](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html) | 7.0 |
| User-defined literals | [N2765](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf) | 7.0 |
| Standard Layout Types | [N2342](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm) | 7.0 |
| Defaulted functions | [N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm) | 7.0 |
| Deleted functions | [N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm) | 7.0 |
| Extended friend declarations | [N1791](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf) | 7.0 |
| Extending `sizeof` | [N2253](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html) [DR850](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#850) | 7.0 |
| Inline namespaces | [N2535](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm) | 7.0 |
| Unrestricted unions | [N2544](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf) | 7.0 |
| Local and unnamed types as template arguments | [N2657](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm) | 7.0 |
| Range-based for | [N2930](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html) | 7.0 |
| Explicit virtual overrides | [N2928](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm) [N3206](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm) [N3272](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm) | 7.0 |
| Minimal support for garbage collection and reachability-based leak detection | [N2670](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm) | N/A (see [Restrictions](#language-restrictions)) |
| Allowing move constructors to throw [noexcept] | [N3050](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html) | 7.0 |
| Defining move special member functions | [N3053](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html) | 7.0 |
| **Concurrency** | | |
| Sequence points | [N2239](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html) |  |
| Atomic operations | [N2427](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html) |  |
| Strong Compare and Exchange | [N2748](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html) |  |
| Bidirectional Fences | [N2752](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm) |  |
| Memory model | [N2429](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm) |  |
| Data-dependency ordering: atomics and memory model | [N2664](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm) |  |
| Propagating exceptions | [N2179](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html) |  |
| Allow atomics use in signal handlers | [N2547](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm) |  |
| Thread-local storage | [N2659](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm) |  |
| Dynamic initialization and destruction with concurrency | [N2660](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm) |  |
| **C99 Features in C++11** | | |
| `__func__` predefined identifier | [N2340](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm) | 7.0 |
| C99 preprocessor | [N1653](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm) | 7.0 |
| `long long` | [N1811](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf) | 7.0 |
| Extended integral types | [N1988](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf) |  |

## 18.2. C++14 Language Features[](#c-14-language-features "Permalink to this headline")

The following table lists new language features that have been accepted into the C++14 standard.

Table 24 C++14 Language Features[](#id484 "Permalink to this table")


| Language Feature | C++14 Proposal | Available in nvcc (device code) |
| --- | --- | --- |
| Tweak to certain C++ contextual conversions | [N3323](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf) | 9.0 |
| Binary literals | [N3472](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf) | 9.0 |
| Functions with deduced return type | [N3638](https://isocpp.org/files/papers/N3638.html) | 9.0 |
| Generalized lambda capture (init-capture) | [N3648](https://isocpp.org/files/papers/N3648.html) | 9.0 |
| Generic (polymorphic) lambda expressions | [N3649](https://isocpp.org/files/papers/N3649.html) | 9.0 |
| Variable templates | [N3651](https://isocpp.org/files/papers/N3651.pdf) | 9.0 |
| Relaxing requirements on constexpr functions | [N3652](https://isocpp.org/files/papers/N3652.html) | 9.0 |
| Member initializers and aggregates | [N3653](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html) | 9.0 |
| Clarifying memory allocation | [N3664](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html) |  |
| Sized deallocation | [N3778](https://isocpp.org/files/papers/n3778.html) |  |
| `[[deprecated]]` attribute | [N3760](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html) | 9.0 |
| Single-quotation-mark as a digit separator | [N3781](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf) | 9.0 |

## 18.3. C++17 Language Features[](#c-17-language-features "Permalink to this headline")

All C++17 language features are supported in nvcc version 11.0 and later, subject to restrictions described [here](#cpp17).

## 18.4. C++20 Language Features[](#c-20-language-features "Permalink to this headline")

All C++20 language features are supported in nvcc version 12.0 and later, subject to restrictions described [here](#cpp20).

## 18.5. Restrictions[](#language-restrictions "Permalink to this headline")

### 18.5.1. Host Compiler Extensions[](#host-compiler-extensions "Permalink to this headline")

Host compiler specific language extensions are not supported in device code.

`__Complex` types are only supported in host code.

`__int128` type is supported in device code when compiled in conjunction with a host compiler that supports it.

`__float128` type is supported for devices with compute capability 10.0 and later, when compiled in conjunction with a host compiler that supports the type. A constant expression of `__float128` type may be processed by the compiler in a floating point representation with lower precision.

### 18.5.2. Preprocessor Symbols[](#preprocessor-symbols "Permalink to this headline")

#### 18.5.2.1. \_\_CUDA\_ARCH\_\_[](#cuda-arch "Permalink to this headline")

1. The type signature of the following entities shall not depend on whether `__CUDA_ARCH__` is defined or not, or on a particular value of `__CUDA_ARCH__`:

   * `__global__` functions and function templates
   * `__device__` and `__constant__` variables
   * textures and surfaces

   Example:

   ```
   #if !defined(__CUDA_ARCH__)
   typedef int mytype;
   #else
   typedef double mytype;
   #endif

   __device__ mytype xxx;         // error: xxx's type depends on __CUDA_ARCH__
   __global__ void foo(mytype in, // error: foo's type depends on __CUDA_ARCH__
                       mytype *ptr)
   {
     *ptr = in;
   }
   ```
2. If a `__global__` function template is instantiated and launched from the host, then the function template must be instantiated with the same template arguments irrespective of whether `__CUDA_ARCH__` is defined and regardless of the value of `__CUDA_ARCH__`.

   Example:

   ```
   __device__ int result;
   template <typename T>
   __global__ void kern(T in)
   {
     result = in;
   }

   __host__ __device__ void foo(void)
   {
   #if !defined(__CUDA_ARCH__)
     kern<<<1,1>>>(1);      // error: "kern<int>" instantiation only
                            // when __CUDA_ARCH__ is undefined!
   #endif
   }

   int main(void)
   {
     foo();
     cudaDeviceSynchronize();
     return 0;
   }
   ```
3. In separate compilation mode, the presence or absence of a definition of a function or variable with external linkage shall not depend on whether `__CUDA_ARCH__` is defined or on a particular value of `__CUDA_ARCH__`[7](#fn14).

   Example:

   ```
   #if !defined(__CUDA_ARCH__)
   void foo(void) { }                  // error: The definition of foo()
                                       // is only present when __CUDA_ARCH__
                                       // is undefined
   #endif
   ```
4. In separate compilation, `__CUDA_ARCH__` must not be used in headers such that different objects could contain different behavior. Or, it must be guaranteed that all objects will compile for the same compute\_arch. If a weak function or template function is defined in a header and its behavior depends on `__CUDA_ARCH__`, then the instances of that function in the objects could conflict if the objects are compiled for different compute arch.

   For example, if an a.h contains:

   ```
   template<typename T>
   __device__ T* getptr(void)
   {
   #if __CUDA_ARCH__ == 700
     return NULL; /* no address */
   #else
     __shared__ T arr[256];
     return arr;
   #endif
   }
   ```

   Then if `a.cu` and `b.cu` both include `a.h` and instantiate `getptr` for the same type, and `b.cu` expects a non-NULL address, and compile with:

   ```
   nvcc –arch=compute_70 –dc a.cu
   nvcc –arch=compute_80 –dc b.cu
   nvcc –arch=sm_80 a.o b.o
   ```

   At link time only one version of the `getptr` is used, so the behavior would depend on which version is chosen. To avoid this, either `a.cu` and `b.cu` must be compiled for the same compute arch, or `__CUDA_ARCH__` should not be used in the shared header function.

The compiler does not guarantee that a diagnostic will be generated for the unsupported uses of `__CUDA_ARCH__` described above.

### 18.5.3. Qualifiers[](#qualifiers "Permalink to this headline")

#### 18.5.3.1. Device Memory Space Specifiers[](#device-memory-space-specifiers "Permalink to this headline")

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on:

* `class`, `struct`, and `union` data members,
* formal parameters,
* non-extern variable declarations within a function that executes on the host.

The `__device__`, `__constant__` and `__managed__` memory space specifiers are not allowed on variable declarations that are neither extern nor static within a function that executes on the device.

A `__device__`, `__constant__`, `__managed__` or `__shared__` variable definition cannot have a class type with a non-empty constructor or a non-empty destructor. A constructor for a class type is considered empty at a point in the translation unit, if it is either a trivial constructor or it satisfies all of the following conditions:

* The constructor function has been defined.
* The constructor function has no parameters, the initializer list is empty and the function body is an empty compound statement.
* Its class has no virtual functions, no virtual base classes and no non-static data member initializers.
* The default constructors of all base classes of its class can be considered empty.
* For all the nonstatic data members of its class that are of class type (or array thereof), the default constructors can be considered empty.

A destructor for a class is considered empty at a point in the translation unit, if it is either a trivial destructor or it satisfies all of the following conditions:

* The destructor function has been defined.
* The destructor function body is an empty compound statement.
* Its class has no virtual functions and no virtual base classes.
* The destructors of all base classes of its class can be considered empty.
* For all the nonstatic data members of its class that are of class type (or array thereof), the destructor can be considered empty.

When compiling in the whole program compilation mode (see the nvcc user manual for a description of this mode), `__device__`, `__shared__`, `__managed__` and `__constant__` variables cannot be defined as external using the `extern` keyword. The only exception is for dynamically allocated `__shared__` variables as described in [\_\_shared\_\_](#shared).

When compiling in the separate compilation mode (see the nvcc user manual for a description of this mode), `__device__`, `__shared__`, `__managed__` and `__constant__` variables can be defined as external using the `extern` keyword. `nvlink` will generate an error when it cannot find a definition for an external variable (unless it is a dynamically allocated `__shared__` variable).

#### 18.5.3.2. \_\_managed\_\_ Memory Space Specifier[](#managed-memory-space-specifier "Permalink to this headline")

Variables marked with the `__managed__` memory space specifier (“managed” variables) have the following restrictions:

* The address of a managed variable is not a constant expression.
* A managed variable shall not have a const qualified type.
* A managed variable shall not have a reference type.
* The address or value of a managed variable shall not be used when the CUDA runtime may not be in a valid state, including the following cases:

  + In static/dynamic initialization or destruction of an object with static or thread local storage duration.
  + In code that executes after exit() has been called (for example, a function marked with gcc’s “`__attribute__((destructor))`”).
  + In code that executes when CUDA runtime may not be initialized (for example, a function marked with gcc’s “`__attribute__((constructor))`”).
* A managed variable cannot be used as an unparenthesized id-expression argument to a `decltype()` expression.
* Managed variables have the same coherence and consistency behavior as specified for dynamically allocated managed memory.
* When a CUDA program containing managed variables is run on an execution platform with multiple GPUs, the variables are allocated only once, and not per GPU.
* A managed variable declaration without the extern linkage is not allowed within a function that executes on the host.
* A managed variable declaration without the extern or static linkage is not allowed within a function that executes on the device.

Here are examples of legal and illegal uses of managed variables:

```
__device__ __managed__ int xxx = 10;         // OK

int *ptr = &xxx;                             // error: use of managed variable
                                             // (xxx) in static initialization
struct S1_t {
  int field;
  S1_t(void) : field(xxx) { };
};
struct S2_t {
  ~S2_t(void) { xxx = 10; }
};

S1_t temp1;                                 // error: use of managed variable
                                            // (xxx) in dynamic initialization

S2_t temp2;                                 // error: use of managed variable
                                            // (xxx) in the destructor of
                                            // object with static storage
                                            // duration

__device__ __managed__ const int yyy = 10;  // error: const qualified type

__device__ __managed__ int &zzz = xxx;      // error: reference type

template <int *addr> struct S3_t { };
S3_t<&xxx> temp;                            // error: address of managed
                                            // variable(xxx) not a
                                            // constant expression

__global__ void kern(int *ptr)
{
  assert(ptr == &xxx);                      // OK
  xxx = 20;                                 // OK
}
int main(void)
{
  int *ptr = &xxx;                          // OK
  kern<<<1,1>>>(ptr);
  cudaDeviceSynchronize();
  xxx++;                                    // OK
  decltype(xxx) qqq;                        // error: managed variable(xxx) used
                                            // as unparenthized argument to
                                            // decltype

  decltype((xxx)) zzz = yyy;                // OK
}
```

#### 18.5.3.3. Volatile Qualifier[](#volatile-qualifier "Permalink to this headline")

Note

The `volatile` keyword is supported to maintain compatibility with ISO C++; however, few if any of its [remaining non-deprecated uses](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1152r0.html#prop) apply to GPUs.

Reads and writes to volatile qualified objects are not atomic, and are compiled to one or more [.volatile instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation) which do NOT guarantee:

* ordering of memory operations, or
* that the number of memory operations performed by the HW matches the number of PTX instructions.

That is, CUDA C++ volatile is not suitable for:

* **Inter-Thread Synchronization**: use atomic operations via [cuda::atomic\_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html), or [Atomic Functions](#atomic-functions) instead.
  Atomic memory operations provide inter-thread synchronization guarantees and deliver much better performance than volatile operations.
  CUDA C++ volatile operations do not provide any inter-thread synchronization guarantees and are therefore not correct for inter-thread synchronization.
  The following example shows how to pass a message across two threads using atomic operations.

  > cuda::atomic\_ref
  >
  > |  |
  > | --- |
  > | ``` __global__ void kernel(int* flag, int* data) {   cuda::atomic_ref<int, cuda::thread_scope_device> f{*flag};   if (threadIdx.x == 0) {     // Consumer: blocks until flag is set by producer, then reads data     while(f.load(cuda::memory_order_acquire) == 0);     if (*data != 42) __trap(); // Errors if wrong data read   } else if (threadIdx.x == 1) {     // Producer: writes data then sets flag     *data = 42;     f.store(1, cuda::memory_order_release);   } } ``` |
  >
  >
  > cuda::atomic
  >
  > |  |
  > | --- |
  > | ``` __global__ void kernel(cuda::atomic<int, cuda::thread_scope_device>* flag, int* data) {   if (threadIdx.x == 0) {     // Consumer: blocks until flag is set by producer, then reads data     while(flag->load(cuda::memory_order_acquire) == 0);     if (*data != 42) __trap(); // Errors if wrong data read   } else if (threadIdx.x == 1) {     // Producer: writes data then sets flag     *data = 42;     flag->store(1, cuda::memory_order_release);   } } ``` |
  >
  >
  > Atomic Functions (`atomicAdd` and `atomicExch`)
  >
  > |  |
  > | --- |
  > | ``` __global__ void kernel(int* flag, int* data) {   if (threadIdx.x == 0) {     // Consumer: blocks until flag is set by producer, then reads data     while(atomicAdd(flag, 0) == 0); // Load with Relaxed Read-Modify-Write     __threadfence();                // SequentiallyConsistent fence     if (*data != 42) __trap();      // Errors if wrong data read   } else if (threadIdx.x == 1) {     // Producer: writes data then sets flag     *data = 42;     __threadfence();     // SequentiallyConsistent fence     atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write   } } ``` |
* **Memory Mapped IO** (MMIO): use [PTX MMIO operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation) via inline PTX instead.
  PTX MMIO operations strictly preserve the number of memory accesses performed.
  CUDA C++ `volatile` operations do not preserve the number of memory accesses performed, and may perform more or less accesses than requested in a non-deterministic way, making them incorrect for MMIO.
  The following example shows how to read and write from a register using PTX mmio operations.

  > ```
  > __global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
  >   // Write to MMIO register:
  >   int value = 13;
  >   asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;" :: "l"(mmio_reg0), "r"(value) : "memory");
  >
  >   // Read MMIO register:
  >   asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];" : "=r"(value) : "l"(mmio_reg1) : "memory");
  >   
  >   if (value != 42) __trap(); // Errors if wrong data read
  > }
  > ```

### 18.5.4. Pointers[](#pointers "Permalink to this headline")

Dereferencing a pointer either to global or shared memory in code that is executed on the host, or to host memory in code that is executed on the device results in an undefined behavior, most often in a segmentation fault and application termination.

The address obtained by taking the address of a `__device__`, `__shared__` or `__constant__` variable can only be used in device code. The address of a `__device__` or `__constant__` variable obtained through `cudaGetSymbolAddress()` as described in [Device Memory](#device-memory) can only be used in host code.

### 18.5.5. Operators[](#operators "Permalink to this headline")

#### 18.5.5.1. Assignment Operator[](#assignment-operator "Permalink to this headline")

`__constant__` variables can only be assigned from the host code through runtime functions ([Device Memory](#device-memory)); they cannot be assigned from the device code.

`__shared__` variables cannot have an initialization as part of their declaration.

It is not allowed to assign values to any of the built-in variables defined in [Built-in Variables](#built-in-variables).

#### 18.5.5.2. Address Operator[](#address-operator "Permalink to this headline")

It is not allowed to take the address of any of the built-in variables defined in [Built-in Variables](#built-in-variables).

### 18.5.6. Run Time Type Information (RTTI)[](#run-time-type-information-rtti "Permalink to this headline")

The following RTTI-related features are supported in host code, but not in device code.

* `typeid` operator
* `std::type_info`
* `dynamic_cast` operator

### 18.5.7. Exception Handling[](#exception-handling "Permalink to this headline")

Exception handling is only supported in host code, but not in device code.

Exception specification is not supported for `__global__` functions.

### 18.5.8. Standard Library[](#standard-library "Permalink to this headline")

Standard libraries are only supported in host code, but not in device code, unless specified otherwise.

### 18.5.9. Namespace Reservations[](#namespace-reservations "Permalink to this headline")

Unless an exception is otherwise noted, it is undefined behavior to add any declarations or definitions to `cuda::`, `nv::`, `cooperative_groups::` or any namespace nested within.

Examples:

```
namespace cuda{
   // Bad: class declaration added to namespace cuda
   struct foo{};

   // Bad: function definition added to namespace cuda
   cudaStream_t make_stream(){
      cudaStream_t s;
      cudaStreamCreate(&s);
      return s;
   }
} // namespace cuda

namespace cuda{
   namespace utils{
      // Bad: function definition added to namespace nested within cuda
      cudaStream_t make_stream(){
          cudaStream_t s;
          cudaStreamCreate(&s);
          return s;
      }
   } // namespace utils
} // namespace cuda

namespace utils{
   namespace cuda{
     // Okay: namespace cuda may be used nested within a non-reserved namespace
     cudaStream_t make_stream(){
          cudaStream_t s;
          cudaStreamCreate(&s);
          return s;
      }
   } // namespace cuda
} // namespace utils

// Bad: Equivalent to adding symbols to namespace cuda at global scope
using namespace utils;
```

### 18.5.10. Functions[](#functions "Permalink to this headline")

#### 18.5.10.1. External Linkage[](#external-linkage "Permalink to this headline")

A call within some device code of a function declared with the extern qualifier is only allowed if the function is defined within the same compilation unit as the device code, i.e., a single file or several files linked together with relocatable device code and nvlink.

#### 18.5.10.2. Implicitly-declared and non-virtual explicitly-defaulted functions[](#implicitly-declared-and-non-virtual-explicitly-defaulted-functions "Permalink to this headline")

Let `F` denote a function that is either implicitly-declared or is a non-virtual function that is explicitly-defaulted on its first declaration. The execution space specifiers (`__host__`, `__device__`) for `F` are the union of the execution space specifiers of all the functions that invoke it (note that a `__global__` caller will be treated as a `__device__` caller for this analysis). For example:

```
class Base {
  int x;
public:
  __host__ __device__ Base(void) : x(10) {}
};

class Derived : public Base {
  int y;
};

class Other: public Base {
  int z;
};

__device__ void foo(void)
{
  Derived D1;
  Other D2;
}

__host__ void bar(void)
{
  Other D3;
}
```

Here, the implicitly-declared constructor function “Derived::Derived” will be treated as a `__device__` function, since it is invoked only from the `__device__` function “foo”. The implicitly-declared constructor function “Other::Other” will be treated as a `__host__ __device__` function, since it is invoked both from a `__device__` function “foo” and a `__host__` function “bar”.

In addition, if `F` is an implicitly declared virtual function (e.g.,a virtual destructor), then the execution spaces of each virtual function `D` overridden by `F` are added to the set of execution spaces for `F`, if `D` is not implicitly declared.

For example:

```
struct Base1 { virtual __host__ __device__ ~Base1() { } };
struct Derived1 : Base1 { }; // implicitly-declared virtual destructor
                             // ~Derived1 has __host__ __device__
                             // execution space specifiers

struct Base2 { virtual __device__ ~Base2() = default; };
struct Derived2 : Base2 { }; // implicitly-declared virtual destructor
                             // ~Derived2 has __device__ execution
                             // space specifiers
```

#### 18.5.10.3. Function Parameters[](#function-parameters "Permalink to this headline")

`__global__` function parameters are passed to the device via constant memory and are limited to 32,764 bytes starting with Volta, and 4 KB on older architectures.

`__global__` functions cannot have a variable number of arguments.

`__global__` function parameters cannot be pass-by-reference.

In separate compilation mode, if a `__device__` or `__global__` function is ODR-used in a particular translation unit, then the parameter and return types of the function must be complete in that translation unit.

Example:

```
//first.cu:
struct S;
__device__ void foo(S); // error: type 'S' is incomplete
__device__ auto *ptr = foo;

int main() { }

//second.cu:
struct S { int x; };
__device__ void foo(S) { }
```

```
//compiler invocation
$nvcc -std=c++14 -rdc=true first.cu second.cu -o first
nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o', first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
nvlink fatal   : merge_elf failed
```

##### 18.5.10.3.1. `__global__` Function Argument Processing[](#global-function-argument-processing "Permalink to this headline")

When a `__global__` function is launched from device code, each argument must be trivially copyable and trivially destructible.

When a `__global__` function is launched from host code, each argument type is allowed to be non-trivially copyable or non-trivially-destructible, but the processing for such types does not follow the standard C++ model, as described below. User code must ensure that this workflow does not affect program correctness. The workflow diverges from standard C++ in two areas:

1. **Memcpy instead of copy constructor invocation**

   When lowering a `__global__` function launch from host code, the compiler generates stub functions that copy the parameters one or more times by value, before eventually using `memcpy` to copy the arguments to the `__global__` function’s parameter memory on the device. This occurs even if an argument was non-trivially-copyable, and therefore may break programs where the copy constructor has side effects.

   Example:

   ```
   #include <cassert>
   struct S {
    int x;
    int *ptr;
    __host__ __device__ S() { }
    __host__ __device__ S(const S &) { ptr = &x; }
   };

   __global__ void foo(S in) {
    // this assert may fail, because the compiler
    // generated code will memcpy the contents of "in"
    // from host to kernel parameter memory, so the
    // "in.ptr" is not initialized to "&in.x" because
    // the copy constructor is skipped.
    assert(in.ptr == &in.x);
   }

   int main() {
     S tmp;
     foo<<<1,1>>>(tmp);
     cudaDeviceSynchronize();
   }
   ```

   Example:

   ```
   #include <cassert>

   __managed__ int counter;
   struct S1 {
   S1() { }
   S1(const S1 &) { ++counter; }
   };

   __global__ void foo(S1) {

   /* this assertion may fail, because
      the compiler generates stub
      functions on the host for a kernel
      launch, and they may copy the
      argument by value more than once.
   */
   assert(counter == 1);
   }

   int main() {
   S1 V;
   foo<<<1,1>>>(V);
   cudaDeviceSynchronize();
   }
   ```
2. **Destructor may be invoked before the ``\_\_global\_\_`` function has finished**

   Kernel launches are asynchronous with host execution. As a result, if a `__global__` function argument has a non-trivial destructor, the destructor may execute in host code even before the `__global__` function has finished execution. This may break programs where the destructor has side effects.

   Example:

   ```
   struct S {
    int *ptr;
    S() : ptr(nullptr) { }
    S(const S &) { cudaMallocManaged(&ptr, sizeof(int)); }
    ~S() { cudaFree(ptr); }
   };

   __global__ void foo(S in) {

     //error: This store may write to memory that has already been
     //       freed (see below).
     *(in.ptr) = 4;

   }

   int main() {
    S V;

    /* The object 'V' is first copied by value to a compiler-generated
     * stub function that does the kernel launch, and the stub function
     * bitwise copies the contents of the argument to kernel parameter
     * memory.
     * However, GPU kernel execution is asynchronous with host
     * execution.
     * As a result, S::~S() will execute when the stub function   returns, releasing allocated memory, even though the kernel may not have finished execution.
     */
    foo<<<1,1>>>(V);
    cudaDeviceSynchronize();
   }
   ```

##### 18.5.10.3.2. Toolkit and Driver Compatibility[](#toolkit-and-driver-compatibility "Permalink to this headline")

Developers must use the 12.1 Toolkit and r530 driver or higher to compile, launch, and debug kernels that accept parameters larger than 4KB. If such kernels are launched on older drivers, CUDA will issue the error `CUDA_ERROR_NOT_SUPPORTED`.

##### 18.5.10.3.3. Link Compatibility across Toolkit Revisions[](#link-compatibility-across-toolkit-revisions "Permalink to this headline")

When linking device objects, if at least one device object contains a kernel with a parameter larger than 4KB, the developer must recompile all objects from their respective device sources with the 12.1 toolkit or higher before linking them together. Failure to do so will result in a linker error.

#### 18.5.10.4. Static Variables within Function[](#static-variables-within-function "Permalink to this headline")

Variable memory space specifiers are allowed in the declaration of a static variable `V` within the immediate or nested block scope of a function `F` where:

* `F` is a `__global__` or `__device__`-only function.
* `F` is a `__host__ __device__` function and `__CUDA_ARCH__` is defined [11](#fn17).

If no explicit memory space specifier is present in the declaration of `V`, an implicit `__device__` specifier is assumed during device compilation.

`V` has the same initialization restrictions as a variable with the same memory space specifiers declared in namespace scope for example a `__device__` variable cannot have a ‘non-empty’ constructor (see [Device Memory Space Specifiers](#device-memory-specifiers)).

Examples of legal and illegal uses of function-scope static variables are shown below.

```
struct S1_t {
  int x;
};

struct S2_t {
  int x;
  __device__ S2_t(void) { x = 10; }
};

struct S3_t {
  int x;
  __device__ S3_t(int p) : x(p) { }
};

__device__ void f1() {
  static int i1;              // OK, implicit __device__ memory space specifier
  static int i2 = 11;         // OK, implicit __device__ memory space specifier
  static __managed__ int m1;  // OK
  static __device__ int d1;   // OK
  static __constant__ int c1; // OK

  static S1_t i3;             // OK, implicit __device__ memory space specifier
  static S1_t i4 = {22};      // OK, implicit __device__ memory space specifier

  static __shared__ int i5;   // OK

  int x = 33;
  static int i6 = x;          // error: dynamic initialization is not allowed
  static S1_t i7 = {x};       // error: dynamic initialization is not allowed

  static S2_t i8;             // error: dynamic initialization is not allowed
  static S3_t i9(44);         // error: dynamic initialization is not allowed
}

__host__ __device__ void f2() {
  static int i1;              // OK, implicit __device__ memory space specifier
                              // during device compilation.
#ifdef __CUDA_ARCH__
  static __device__ int d1;   // OK, declaration is only visible during device
                              // compilation  (__CUDA_ARCH__ is defined)
#else
  static int d0;              // OK, declaration is only visible during host
                              // compilation (__CUDA_ARCH__ is not defined)
#endif

  static __device__ int d2;   // error: __device__ variable inside
                              // a host function during host compilation
                              // i.e. when __CUDA_ARCH__ is not defined

  static __shared__ int i2;  // error: __shared__ variable inside
                             // a host function during host compilation
                             // i.e. when __CUDA_ARCH__ is not defined
}
```

#### 18.5.10.5. Function Pointers[](#function-pointers "Permalink to this headline")

The address of a `__global__` function taken in host code cannot be used in device code (e.g. to launch the kernel). Similarly, the address of a `__global__` function taken in device code cannot be used in host code.

It is not allowed to take the address of a `__device__` function in host code.

#### 18.5.10.6. Function Recursion[](#function-recursion "Permalink to this headline")

`__global__` functions do not support recursion.

#### 18.5.10.7. Friend Functions[](#friend-functions "Permalink to this headline")

A `__global__` function or function template cannot be defined in a friend declaration.

Example:

```
struct S1_t {
  friend __global__
  void foo1(void);  // OK: not a definition
  template<typename T>
  friend __global__
  void foo2(void); // OK: not a definition

  friend __global__
  void foo3(void) { } // error: definition in friend declaration

  template<typename T>
  friend __global__
  void foo4(void) { } // error: definition in friend declaration
};
```

#### 18.5.10.8. Operator Function[](#operator-function "Permalink to this headline")

An operator function cannot be a `__global__` function.

#### 18.5.10.9. Allocation and Deallocation Functions[](#allocation-and-deallocation-functions "Permalink to this headline")

A user-defined `operator new`, `operator new[]`, `operator delete`, or `operator delete[]` cannot be used to replace the corresponding `__host__` or `__device__` builtins provided by the compiler.

### 18.5.11. Classes[](#classes "Permalink to this headline")

#### 18.5.11.1. Data Members[](#data-members "Permalink to this headline")

Static data members are not supported except for those that are also const-qualified (see [Const-qualified variables](#const-variables)).

#### 18.5.11.2. Function Members[](#function-members "Permalink to this headline")

Static member functions cannot be `__global__` functions.

#### 18.5.11.3. Virtual Functions[](#virtual-functions "Permalink to this headline")

When a function in a derived class overrides a virtual function in a base class, the execution space specifiers (i.e., `__host__`, `__device__`) on the overridden and overriding functions must match.

It is not allowed to pass as an argument to a `__global__` function an object of a class with virtual functions.

If an object is created in host code, invoking a virtual function for that object in device code has undefined behavior.

If an object is created in device code, invoking a virtual function for that object in host code has undefined behavior.

See [Windows-Specific](#windows-specific) for additional constraints when using the Microsoft host compiler.

Example:

```
struct S1 { virtual __host__ __device__ void foo() { } };

__managed__ S1 *ptr1, *ptr2;

__managed__ __align__(16) char buf1[128];
__global__ void kern() {
  ptr1->foo();     // error: virtual function call on a object
                   //        created in host code.
  ptr2 = new(buf1) S1();
}

int main(void) {
  void *buf;
  cudaMallocManaged(&buf, sizeof(S1), cudaMemAttachGlobal);
  ptr1 = new (buf) S1();
  kern<<<1,1>>>();
  cudaDeviceSynchronize();
  ptr2->foo();  // error: virtual function call on an object
                //        created in device code.
}
```

#### 18.5.11.4. Virtual Base Classes[](#virtual-base-classes "Permalink to this headline")

It is not allowed to pass as an argument to a `__global__` function an object of a class derived from virtual base classes.

See [Windows-Specific](#windows-specific) for additional constraints when using the Microsoft host compiler.

#### 18.5.11.5. Anonymous Unions[](#anonymous-unions "Permalink to this headline")

Member variables of a namespace scope anonymous union cannot be referenced in a `__global__` or `__device__` function.

#### 18.5.11.6. Windows-Specific[](#windows-specific "Permalink to this headline")

The CUDA compiler follows the IA64 ABI for class layout, while the Microsoft host compiler does not. Let `T` denote a pointer to member type, or a class type that satisfies any of the following conditions:

* `T` has virtual functions.
* `T` has a virtual base class.
* `T` has multiple inheritance with more than one direct or indirect empty base class.
* All direct and indirect base classes `B` of `T` are empty and the type of the first field `F` of `T` uses `B` in its definition, such that `B` is laid out at offset 0 in the definition of `F`.

Let `C` denote `T` or a class type that has `T` as a field type or as a base class type. The CUDA compiler may compute the class layout and size differently than the Microsoft host compiler for the type `C`.

As long as the type `C` is used exclusively in host or device code, the program should work correctly.

Passing an object of type `C` between host and device code has undefined behavior, for example, as an argument to a `__global__` function or through `cudaMemcpy*()` calls.

Accessing an object of type `C` or any subobject in device code, or invoking a member function in device code, has undefined behavior if the object is created in host code.

Accessing an object of type `C` or any subobject in host code, or invoking a member function in host code, has undefined behavior if the object is created in device code [12](#fn19).

### 18.5.12. Templates[](#templates "Permalink to this headline")

A type or template cannot be used in the type, non-type or template template argument of a `__global__` function template instantiation or a `__device__/__constant__` variable instantiation if either:

* The type or template is defined within a `__host__` or `__host__ __device__`.
* The type or template is a class member with `private` or `protected` access and its parent class is not defined within a `__device__` or `__global__` function.
* The type is unnamed.
* The type is compounded from any of the types above.

Example:

```
template <typename T>
__global__ void myKernel(void) { }

class myClass {
private:
    struct inner_t { };
public:
    static void launch(void)
    {
       // error: inner_t is used in template argument
       // but it is private
       myKernel<inner_t><<<1,1>>>();
    }
};

// C++14 only
template <typename T> __device__ T d1;

template <typename T1, typename T2> __device__ T1 d2;

void fn() {
  struct S1_t { };
  // error (C++14 only): S1_t is local to the function fn
  d1<S1_t> = {};

  auto lam1 = [] { };
  // error (C++14 only): a closure type cannot be used for
  // instantiating a variable template
  d2<int, decltype(lam1)> = 10;
}
```

### 18.5.13. Trigraphs and Digraphs[](#trigraphs-and-digraphs "Permalink to this headline")

Trigraphs are not supported on any platform. Digraphs are not supported on Windows.

### 18.5.14. Const-qualified variables[](#const-qualified-variables "Permalink to this headline")

Let ‘V’ denote a namespace scope variable or a class static member variable that has const qualified type and does not have execution space annotations (for example, `__device__, __constant__, __shared__`). V is considered to be a host code variable.

The value of V may be directly used in device code, if

* V has been initialized with a constant expression before the point of use,
* the type of V is not volatile-qualified, and
* it has one of the following types:

  + built-in floating point type except when the Microsoft compiler is used as the host compiler,
  + built-in integral type.

Device source code cannot contain a reference to V or take the address of V.

Example:

```
const int xxx = 10;
struct S1_t {  static const int yyy = 20; };

extern const int zzz;
const float www = 5.0;
__device__ void foo(void) {
  int local1[xxx];          // OK
  int local2[S1_t::yyy];    // OK

  int val1 = xxx;           // OK

  int val2 = S1_t::yyy;     // OK

  int val3 = zzz;           // error: zzz not initialized with constant
                            // expression at the point of use.

  const int &val3 = xxx;    // error: reference to host variable
  const int *val4 = &xxx;   // error: address of host variable
  const float val5 = www;   // OK except when the Microsoft compiler is used as
                            // the host compiler.
}
const int zzz = 20;
```

### 18.5.15. Long Double[](#long-double "Permalink to this headline")

The use of `long double` type is not supported in device code.

### 18.5.16. Deprecation Annotation[](#deprecation-annotation "Permalink to this headline")

nvcc supports the use of `deprecated` attribute when using `gcc`, `clang`, `xlC`, `icc` or `pgcc` host compilers, and the use of `deprecated` declspec when using the `cl.exe` host compiler. It also supports the `[[deprecated]]` standard attribute when the C++14 dialect has been enabled. The CUDA frontend compiler will generate a deprecation diagnostic for a reference to a deprecated entity from within the body of a `__device__`, `__global__` or `__host__ __device__` function when `__CUDA_ARCH__` is defined (i.e., during device compilation phase). Other references to deprecated entities will be handled by the host compiler, e.g., a reference from within a `__host__` function.

The CUDA frontend compiler does not support the `#pragma gcc diagnostic` or `#pragma warning` mechanisms supported by various host compilers. Therefore, deprecation diagnostics generated by the CUDA frontend compiler are not affected by these pragmas, but diagnostics generated by the host compiler will be affected. To suppress the warning for device-code, user can use NVIDIA specific pragma [#pragma nv\_diag\_suppress](#nv-diagnostic-pragmas). The `nvcc` flag `-Wno-deprecated-declarations` can be used to suppress all deprecation warnings, and the flag `-Werror=deprecated-declarations` can be used to turn deprecation warnings into errors.

### 18.5.17. Noreturn Annotation[](#noreturn-annotation "Permalink to this headline")

nvcc supports the use of `noreturn` attribute when using `gcc`, `clang`, `xlC`, `icc` or `pgcc` host compilers, and the use of `noreturn` declspec when using the `cl.exe` host compiler. It also supports the `[[noreturn]]` standard attribute when the C++11 dialect has been enabled.

The attribute/declspec can be used in both host and device code.

### 18.5.18. [[likely]] / [[unlikely]] Standard Attributes[](#likely-unlikely-standard-attributes "Permalink to this headline")

These attributes are accepted in all configurations that support the C++ standard attribute syntax. The attributes can be used to hint to the device compiler optimizer whether a statement is more or less likely to be executed compared to any alternative path that does not include the statement.

Example:

```
__device__ int foo(int x) {

 if (i < 10) [[likely]] { // the 'if' block will likely be entered
  return 4;
 }
 if (i < 20) [[unlikely]] { // the 'if' block will not likely be entered
  return 1;
 }
 return 0;
}
```

If these attributes are used in host code when `__CUDA_ARCH__` is undefined, then they will be present in the code parsed by the host compiler, which may generate a warning if the attributes are not supported. For example, `clang`11 host compiler will generate an ‘unknown attribute’ warning.

### 18.5.19. const and pure GNU Attributes[](#const-and-pure-gnu-attributes "Permalink to this headline")

These attributes are supported for both host and device functions, when using a language dialect and host compiler that also supports these attributes e.g. with g++ host compiler.

For a device function annotated with the `pure` attribute, the device code optimizer assumes that the function does not change any mutable state visible to caller functions (e.g. memory).

For a device function annotated with the `const` attribute, the device code optimizer assumes that the function does not access or change any mutable state visible to caller functions (e.g. memory).

Example:

```
__attribute__((const)) __device__ int get(int in);

__device__ int doit(int in) {
int sum = 0;

//because 'get' is marked with 'const' attribute
//device code optimizer can recognize that the
//second call to get() can be commoned out.
sum = get(in);
sum += get(in);

return sum;
}
```

### 18.5.20. \_\_nv\_pure\_\_ Attribute[](#nv-pure-attribute "Permalink to this headline")

The `__nv_pure__` attributed is supported for both host and device functions. For host functions, when using a language dialect that supports the `pure` GNU attribute, the `__nv_pure__` attribute is translated to the `pure` GNU attribute. Similarly when using MSVC as the host compiler, the attribute is translated to the MSVC `noalias` attribute.

When a device function is annotated with the `__nv_pure__` attribute, the device code optimizer assumes that the function does not change any mutable state visible to caller functions (e.g. memory).

### 18.5.21. Intel Host Compiler Specific[](#intel-host-compiler-specific "Permalink to this headline")

The CUDA frontend compiler parser does not recognize some of the intrinsic functions supported by the Intel compiler (e.g. `icc`). When using the Intel compiler as a host compiler, `nvcc` will therefore enable the macro `__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES` during preprocessing. This macro enables explicit declarations of the Intel compiler intrinsic functions in the associated header files, allowing `nvcc` to support use of such functions in host code[13](#fn20).

### 18.5.22. C++11 Features[](#c-11-features "Permalink to this headline")

C++11 features that are enabled by default by the host compiler are also supported by nvcc, subject to the restrictions described in this document. In addition, invoking nvcc with `-std=c++11` flag turns on all C++11 features and also invokes the host preprocessor, compiler and linker with the corresponding C++11 dialect option [14](#fn21).

#### 18.5.22.1. Lambda Expressions[](#lambda-expressions "Permalink to this headline")

The execution space specifiers for all member functions[15](#fn22) of the closure class associated with a lambda expression are derived by the compiler as follows. As described in the C++11 standard, the compiler creates a closure type in the smallest block scope, class scope or namespace scope that contains the lambda expression. The innermost function scope enclosing the closure type is computed, and the corresponding function’s execution space specifiers are assigned to the closure class member functions. If there is no enclosing function scope, the execution space specifier is `__host__`.

Examples of lambda expressions and computed execution space specifiers are shown below (in comments).

```
auto globalVar = [] { return 0; }; // __host__

void f1(void) {
  auto l1 = [] { return 1; };      // __host__
}

__device__ void f2(void) {
  auto l2 = [] { return 2; };      // __device__
}

__host__ __device__ void f3(void) {
  auto l3 = [] { return 3; };      // __host__ __device__
}

__device__ void f4(int (*fp)() = [] { return 4; } /* __host__ */) {
}

__global__ void f5(void) {
  auto l5 = [] { return 5; };      // __device__
}

__device__ void f6(void) {
  struct S1_t {
    static void helper(int (*fp)() = [] {return 6; } /* __device__ */) {
    }
  };
}
```

The closure type of a lambda expression cannot be used in the type or non-type argument of a `__global__` function template instantiation, unless the lambda is defined within a `__device__` or `__global__` function.

Example:

```
template <typename T>
__global__ void foo(T in) { };

template <typename T>
struct S1_t { };

void bar(void) {
  auto temp1 = [] { };

  foo<<<1,1>>>(temp1);                    // error: lambda closure type used in
                                          // template type argument
  foo<<<1,1>>>( S1_t<decltype(temp1)>()); // error: lambda closure type used in
                                          // template type argument
}
```

#### 18.5.22.2. std::initializer\_list[](#std-initializer-list "Permalink to this headline")

By default, the CUDA compiler will implicitly consider the member functions of `std::initializer_list` to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code. The nvcc flag `--no-host-device-initializer-list` will disable this behavior; member functions of `std::initializer_list` will then be considered as `__host__` functions and will not be directly invokable from device code.

Example:

```
#include <initializer_list>

__device__ int foo(std::initializer_list<int> in);

__device__ void bar(void)
  {
    foo({4,5,6});   // (a) initializer list containing only
                    // constant expressions.

    int i = 4;
    foo({i,5,6});   // (b) initializer list with at least one
                    // non-constant element.
                    // This form may have better performance than (a).
  }
```

#### 18.5.22.3. Rvalue references[](#rvalue-references "Permalink to this headline")

By default, the CUDA compiler will implicitly consider `std::move` and `std::forward` function templates to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code. The nvcc flag `--no-host-device-move-forward` will disable this behavior; `std::move` and `std::forward` will then be considered as `__host__` functions and will not be directly invokable from device code.

#### 18.5.22.4. Constexpr functions and function templates[](#constexpr-functions-and-function-templates "Permalink to this headline")

By default, a constexpr function cannot be called from a function with incompatible execution space [16](#fn23). The experimental nvcc flag `--expt-relaxed-constexpr` removes this restriction [17](#fn24). When this flag is specified, host code can invoke a `__device__` constexpr function and device code can invoke a `__host__` constexpr function. nvcc will define the macro `__CUDACC_RELAXED_CONSTEXPR__` when `--expt-relaxed-constexpr` has been specified. Note that a function template instantiation may not be a constexpr function even if the corresponding template is marked with the keyword `constexpr` (C++11 Standard Section `[dcl.constexpr.p6]`).

#### 18.5.22.5. Constexpr variables[](#constexpr-variables "Permalink to this headline")

Let ‘V’ denote a namespace scope variable or a class static member variable that has been marked constexpr and that does not have execution space annotations (e.g., `__device__, __constant__, __shared__`). V is considered to be a host code variable.

If V is of scalar type [18](#fn25) other than `long double` and the type is not volatile-qualified, the value of V can be directly used in device code. In addition, if V is of a non-scalar type then scalar elements of V can be used inside a constexpr `__device__` or `__host__ __device__` function, if the call to the function is a constant expression [19](#fn26). Device source code cannot contain a reference to V or take the address of V.

Example:

```
constexpr int xxx = 10;
constexpr int yyy = xxx + 4;
struct S1_t { static constexpr int qqq = 100; };

constexpr int host_arr[] = { 1, 2, 3};
constexpr __device__ int get(int idx) { return host_arr[idx]; }

__device__ int foo(int idx) {
  int v1 = xxx + yyy + S1_t::qqq;  // OK
  const int &v2 = xxx;             // error: reference to host constexpr
                                   // variable
  const int *v3 = &xxx;            // error: address of host constexpr
                                   // variable
  const int &v4 = S1_t::qqq;       // error: reference to host constexpr
                                   // variable
  const int *v5 = &S1_t::qqq;      // error: address of host constexpr
                                   // variable

  v1 += get(2);                    // OK: 'get(2)' is a constant
                                   // expression.
  v1 += get(idx);                  // error: 'get(idx)' is not a constant
                                   // expression
  v1 += host_arr[2];               // error: 'host_arr' does not have
                                   // scalar type.
  return v1;
}
```

#### 18.5.22.6. Inline namespaces[](#inline-namespaces "Permalink to this headline")

For an input CUDA translation unit, the CUDA compiler may invoke the host compiler for compiling the host code within the translation unit. In the code passed to the host compiler, the CUDA compiler will inject additional compiler generated code, if the input CUDA translation unit contained a definition of any of the following entities:

* `__global__` function or function template instantiation
* `__device__`, `__constant__`
* variables with surface or texture type

The compiler generated code contains a reference to the defined entity. If the entity is defined within an inline namespace and another entity of the same name and type signature is defined in an enclosing namespace, this reference may be considered ambiguous by the host compiler and host compilation will fail.

This limitation can be avoided by using unique names for such entities defined within an inline namespace.

Example:

```
__device__ int Gvar;
inline namespace N1 {
  __device__ int Gvar;
}

// <-- CUDA compiler inserts a reference to "Gvar" at this point in the
// translation unit. This reference will be considered ambiguous by the
// host compiler and compilation will fail.
```

Example:

```
inline namespace N1 {
  namespace N2 {
    __device__ int Gvar;
  }
}

namespace N2 {
  __device__ int Gvar;
}

// <-- CUDA compiler inserts reference to "::N2::Gvar" at this point in
// the translation unit. This reference will be considered ambiguous by
// the host compiler and compilation will fail.
```

##### 18.5.22.6.1. Inline unnamed namespaces[](#inline-unnamed-namespaces "Permalink to this headline")

The following entities cannot be declared in namespace scope within an inline unnamed namespace:

* `__managed__`, `__device__`, `__shared__` and `__constant__` variables
* `__global__` function and function templates
* variables with surface or texture type

Example:

```
inline namespace {
  namespace N2 {
    template <typename T>
    __global__ void foo(void);            // error

    __global__ void bar(void) { }         // error

    template <>
    __global__ void foo<int>(void) { }    // error

    __device__ int x1b;                   // error
    __constant__ int x2b;                 // error
    __shared__ int x3b;                   // error

    texture<int> q2;                      // error
    surface<int> s2;                      // error
  }
};
```

#### 18.5.22.7. thread\_local[](#thread-local "Permalink to this headline")

The `thread_local` storage specifier is not allowed in device code.

#### 18.5.22.8. \_\_global\_\_ functions and function templates[](#global-functions-and-function-templates "Permalink to this headline")

If the closure type associated with a lambda expression is used in a template argument of a `__global__` function template instantiation, the lambda expression must either be defined in the immediate or nested block scope of a `__device__` or `__global__` function, or must be an [extended lambda](#extended-lambda).

Example:

```
template <typename T>
__global__ void kernel(T in) { }

__device__ void foo_device(void)
{
  // All kernel instantiations in this function
  // are valid, since the lambdas are defined inside
  // a __device__ function.

  kernel<<<1,1>>>( [] __device__ { } );
  kernel<<<1,1>>>( [] __host__ __device__ { } );
  kernel<<<1,1>>>( []  { } );
}

auto lam1 = [] { };

auto lam2 = [] __host__ __device__ { };

void foo_host(void)
{
   // OK: instantiated with closure type of an extended __device__ lambda
   kernel<<<1,1>>>( [] __device__ { } );

   // OK: instantiated with closure type of an extended __host__ __device__
   // lambda
   kernel<<<1,1>>>( [] __host__ __device__ { } );

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( []  { } );

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( lam1);

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( lam2);
}
```

A `__global__` function or function template cannot be declared as `constexpr`.

A `__global__` function or function template cannot have a parameter of type `std::initializer_list` or `va_list`.

A `__global__` function cannot have a parameter of rvalue reference type.

A variadic `__global__` function template has the following restrictions:

* Only a single pack parameter is allowed.
* The pack parameter must be listed last in the template parameter list.

Example:

```
// ok
template <template <typename...> class Wrapper, typename... Pack>
__global__ void foo1(Wrapper<Pack...>);

// error: pack parameter is not last in parameter list
template <typename... Pack, template <typename...> class Wrapper>
__global__ void foo2(Wrapper<Pack...>);

// error: multiple parameter packs
template <typename... Pack1, int...Pack2, template<typename...> class Wrapper1,
          template<int...> class Wrapper2>
__global__ void foo3(Wrapper1<Pack1...>, Wrapper2<Pack2...>);
```

#### 18.5.22.9. \_\_managed\_\_ and \_\_shared\_\_ variables[](#managed-and-shared-variables "Permalink to this headline")

`` `__managed__ `` and `__shared__` variables cannot be marked with the keyword `constexpr`.

#### 18.5.22.10. Defaulted functions[](#defaulted-functions "Permalink to this headline")

Execution space specifiers on a non-virtual function that is explicitly-defaulted on its first declaration are ignored by the CUDA compiler. Instead, the CUDA compiler will infer the execution space specifiers as described in [Implicitly-declared and non-virtual explicitly-defaulted functions](#compiler-generated-functions).

Execution space specifiers are not ignored if the function is either:

* Explicitly-defaulted but not on its first declaration.
* Explicitly-defaulted and virtual.

Example:

```
 struct S1 {
   // warning: __host__ annotation is ignored on a non-virtual function that
   //          is explicitly-defaulted on its first declaration
   __host__ S1() = default;
 };

 __device__ void foo1() {
   //note: __device__ execution space is derived for S1::S1
   //       based on implicit call from within __device__ function
   //       foo1
   S1 s1;
 }

 struct S2 {
   __host__ S2();
 };

 //note: S2::S2 is not defaulted on its first declaration, and
 //      its execution space is fixed to __host__  based on its
 //      first declaration.
 S2::S2() = default;

 __device__ void foo2() {
    // error: call from __device__ function 'foo2' to
    //        __host__ function 'S2::S2'
    S2 s2;
 }

struct S3 {
  //note: S3::~S3 has __host__ execution space
  virtual __host__ ~S3() = default;
};

__device__ void foo3() {
  S3 qqq;
}  /*(implicit destructor call for 'qqq'):
      error: call from a __device__ fuction 'foo3' to a
     __host__ function 'S3::~S3' */
```

### 18.5.23. C++14 Features[](#c-14-features "Permalink to this headline")

C++14 features enabled by default by the host compiler are also supported by nvcc. Passing nvcc `-std=c++14` flag turns on all C++14 features and also invokes the host preprocessor, compiler and linker with the corresponding C++14 dialect option [20](#fn27). This section describes the restrictions on the supported C++14 features.

#### 18.5.23.1. Functions with deduced return type[](#functions-with-deduced-return-type "Permalink to this headline")

A `__global__` function cannot have a deduced return type.

If a `__device__` function has deduced return type, the CUDA frontend compiler will change the function declaration to have a `void` return type, before invoking the host compiler. This may cause issues for introspecting the deduced return type of the `__device__` function in host code. Thus, the CUDA compiler will issue compile-time errors for referencing such deduced return type outside device function bodies, except if the reference is absent when `__CUDA_ARCH__` is undefined.

Examples:

```
__device__ auto fn1(int x) {
  return x;
}

__device__ decltype(auto) fn2(int x) {
  return x;
}

__device__ void device_fn1() {
  // OK
  int (*p1)(int) = fn1;
}

// error: referenced outside device function bodies
decltype(fn1(10)) g1;

void host_fn1() {
  // error: referenced outside device function bodies
  int (*p1)(int) = fn1;

  struct S_local_t {
    // error: referenced outside device function bodies
    decltype(fn2(10)) m1;

    S_local_t() : m1(10) { }
  };
}

// error: referenced outside device function bodies
template <typename T = decltype(fn2)>
void host_fn2() { }

template<typename T> struct S1_t { };

// error: referenced outside device function bodies
struct S1_derived_t : S1_t<decltype(fn1)> { };
```

#### 18.5.23.2. Variable templates[](#variable-templates "Permalink to this headline")

A `__device__/__constant__` variable template cannot have a const qualified type when using the Microsoft host compiler.

Examples:

```
// error: a __device__ variable template cannot
// have a const qualified type on Windows
template <typename T>
__device__ const T d1(2);

int *const x = nullptr;
// error: a __device__ variable template cannot
// have a const qualified type on Windows
template <typename T>
__device__ T *const d2(x);

// OK
template <typename T>
__device__ const T *d3;

__device__ void fn() {
  int t1 = d1<int>;

  int *const t2 = d2<int>;

  const int *t3 = d3<int>;
}
```

### 18.5.24. C++17 Features[](#c-17-features "Permalink to this headline")

C++17 features enabled by default by the host compiler are also supported by nvcc. Passing nvcc `-std=c++17` flag turns on all C++17 features and also invokes the host preprocessor, compiler and linker with the corresponding C++17 dialect option [21](#fn28). This section describes the restrictions on the supported C++17 features.

#### 18.5.24.1. Inline Variable[](#inline-variable "Permalink to this headline")

* A namespace scope inline variable declared with `__device__` or `__constant__` or `__managed__` memory space specifier must have internal linkage, if the code is compiled with nvcc in whole program compilation mode.

  Examples:

  ```
  inline __device__ int xxx; //error when compiled with nvcc in
                             //whole program compilation mode.
                             //ok when compiled with nvcc in
                             //separate compilation mode.

  inline __shared__ int yyy0; // ok.

  static inline __device__ int yyy; // ok: internal linkage
  namespace {
  inline __device__ int zzz; // ok: internal linkage
  }
  ```
* When using g++ host compiler, an inline variable declared with `__managed__` memory space specifier may not be visible to the debugger.

#### 18.5.24.2. Structured Binding[](#structured-binding "Permalink to this headline")

A structured binding cannot be declared with a variable memory space specifier.

Example:

```
struct S { int x; int y; };
__device__ auto [a1, b1] = S{4,5}; // error
```

### 18.5.25. C++20 Features[](#c-20-features "Permalink to this headline")

C++20 features enabled by default by the host compiler are also supported by nvcc. Passing nvcc `-std=c++20` flag turns on all C++20 features and also invokes the host preprocessor, compiler and linker with the corresponding C++20 dialect option [22](#fn29). This section describes the restrictions on the supported C++20 features.

#### 18.5.25.1. Module support[](#module-support "Permalink to this headline")

Modules are not supported in CUDA C++, in either host or device code. Uses of the `module`, `export` and `import` keywords are diagnosed as errors.

#### 18.5.25.2. Coroutine support[](#coroutine-support "Permalink to this headline")

Coroutines are not supported in device code. Uses of the `co_await`, `co_yield` and `co_return` keywords in the scope of a device function are diagnosed as error during device compilation.

#### 18.5.25.3. Three-way comparison operator[](#three-way-comparison-operator "Permalink to this headline")

The three-way comparison operator is supported in both host and device code, but some uses implicitly rely on functionality from the Standard Template Library provided by the host implementation. Uses of those operators may require specifying the flag `--expt-relaxed-constexpr` to silence warnings and the functionality requires that the host implementation satisfies the requirements of device code.

Example:

```
#include<compare>
struct S {
  int x, y, z;
  auto operator<=>(const S& rhs) const = default;
  __host__ __device__ bool operator<=>(int rhs) const { return false; }
};
__host__ __device__ bool f(S a, S b) {
  if (a <=> 1) // ok, calls a user-defined host-device overload
    return true;
  return a < b; // call to an implicitly-declared function and requires
                // a device-compatible std::strong_ordering implementation
}
```

#### 18.5.25.4. Consteval functions[](#consteval-functions "Permalink to this headline")

Ordinarily, cross execution space calls are not allowed, and cause a compiler diagnostic (warning or error). This restriction does not apply when the called function is declared with the `consteval` specifier. Thus, a `__device__` or `__global__` function can call a `__host__``consteval` function, and a `__host__` function can call a `__device__ consteval` function.

Example:

```
namespace N1 {
//consteval host function
consteval int hcallee() { return 10; }

__device__ int dfunc() { return hcallee(); /* OK */ }
__global__ void gfunc() { (void)hcallee(); /* OK */ }
__host__ __device__ int hdfunc() { return hcallee();  /* OK */ }
int hfunc() { return hcallee(); /* OK */ }
} // namespace N1


namespace N2 {
//consteval device function
consteval __device__ int dcallee() { return 10; }

__device__ int dfunc() { return dcallee(); /* OK */ }
__global__ void gfunc() { (void)dcallee(); /* OK */ }
__host__ __device__ int hdfunc() { return dcallee();  /* OK */ }
int hfunc() { return dcallee(); /* OK */ }
}
```

## 18.6. Polymorphic Function Wrappers[](#polymorphic-function-wrappers "Permalink to this headline")

A polymorphic function wrapper class template `nvstd::function` is provided in the `nvfunctional` header. Instances of this class template can be used to store, copy and invoke any callable target, e.g., lambda expressions. `nvstd::function` can be used in both host and device code.

Example:

```
#include <nvfunctional>

__device__ int foo_d() { return 1; }
__host__ __device__ int foo_hd () { return 2; }
__host__ int foo_h() { return 3; }

__global__ void kernel(int *result) {
  nvstd::function<int()> fn1 = foo_d;
  nvstd::function<int()> fn2 = foo_hd;
  nvstd::function<int()> fn3 =  []() { return 10; };

  *result = fn1() + fn2() + fn3();
}

__host__ __device__ void hostdevice_func(int *result) {
  nvstd::function<int()> fn1 = foo_hd;
  nvstd::function<int()> fn2 =  []() { return 10; };

  *result = fn1() + fn2();
}

__host__ void host_func(int *result) {
  nvstd::function<int()> fn1 = foo_h;
  nvstd::function<int()> fn2 = foo_hd;
  nvstd::function<int()> fn3 =  []() { return 10; };

  *result = fn1() + fn2() + fn3();
}
```

Instances of `nvstd::function` in host code cannot be initialized with the address of a `__device__` function or with a functor whose `operator()` is a `__device__` function. Instances of `nvstd::function` in device code cannot be initialized with the address of a `__host__` function or with a functor whose `operator()` is a `__host__` function.

`nvstd::function` instances cannot be passed from host code to device code (and vice versa) at run time. `nvstd::function` cannot be used in the parameter type of a `__global__` function, if the `__global__` function is launched from host code.

Example:

```
#include <nvfunctional>

__device__ int foo_d() { return 1; }
__host__ int foo_h() { return 3; }
auto lam_h = [] { return 0; };

__global__ void k(void) {
  // error: initialized with address of __host__ function
  nvstd::function<int()> fn1 = foo_h;

  // error: initialized with address of functor with
  // __host__ operator() function
  nvstd::function<int()> fn2 = lam_h;
}

__global__ void kern(nvstd::function<int()> f1) { }

void foo(void) {
  // error: initialized with address of __device__ function
  nvstd::function<int()> fn1 = foo_d;

  auto lam_d = [=] __device__ { return 1; };

  // error: initialized with address of functor with
  // __device__ operator() function
  nvstd::function<int()> fn2 = lam_d;

  // error: passing nvstd::function from host to device
  kern<<<1,1>>>(fn2);
}
```

`nvstd::function` is defined in the `nvfunctional` header as follows:

```
namespace nvstd {
  template <class _RetType, class ..._ArgTypes>
  class function<_RetType(_ArgTypes...)>
  {
    public:
      // constructors
      __device__ __host__  function() noexcept;
      __device__ __host__  function(nullptr_t) noexcept;
      __device__ __host__  function(const function &);
      __device__ __host__  function(function &&);

      template<class _F>
      __device__ __host__  function(_F);

      // destructor
      __device__ __host__  ~function();

      // assignment operators
      __device__ __host__  function& operator=(const function&);
      __device__ __host__  function& operator=(function&&);
      __device__ __host__  function& operator=(nullptr_t);
      __device__ __host__  function& operator=(_F&&);

      // swap
      __device__ __host__  void swap(function&) noexcept;

      // function capacity
      __device__ __host__  explicit operator bool() const noexcept;

      // function invocation
      __device__ _RetType operator()(_ArgTypes...) const;
  };

  // null pointer comparisons
  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator==(const function<_R(_ArgTypes...)>&, nullptr_t) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator==(nullptr_t, const function<_R(_ArgTypes...)>&) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator!=(const function<_R(_ArgTypes...)>&, nullptr_t) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator!=(nullptr_t, const function<_R(_ArgTypes...)>&) noexcept;

  // specialized algorithms
  template <class _R, class... _ArgTypes>
  __device__ __host__
  void swap(function<_R(_ArgTypes...)>&, function<_R(_ArgTypes...)>&);
}
```

## 18.7. Extended Lambdas[](#extended-lambdas "Permalink to this headline")

The nvcc flag `'--extended-lambda'` allows explicit execution space annotations in a lambda expression [23](#fn30). The execution space annotations should be present after the ‘lambda-introducer’ and before the optional ‘lambda-declarator’. nvcc will define the macro `__CUDACC_EXTENDED_LAMBDA__` when the `'--extended-lambda'` flag has been specified.

An ‘extended `__device__` lambda’ is a lambda expression that is annotated explicitly with ‘`__device__`’, and is defined within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

An ‘extended `__host__ __device__` lambda’ is a lambda expression that is annotated explicitly with both ‘`__host__`’ and ‘`__device__`’, and is defined within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

An ‘extended lambda’ denotes either an extended `__device__` lambda or an extended `__host__ __device__` lambda. Extended lambdas can be used in the type arguments of [\_\_global\_\_ function template instantiation](#cpp11-global).

If the execution space annotations are not explicitly specified, they are computed based on the scopes enclosing the closure class associated with the lambda, as described in the section on C++11 support. The execution space annotations are applied to all methods of the closure class associated with the lambda.

Example:

```
void foo_host(void) {
  // not an extended lambda: no explicit execution space annotations
  auto lam1 = [] { };

  // extended __device__ lambda
  auto lam2 = [] __device__ { };

  // extended __host__ __device__ lambda
  auto lam3 = [] __host__ __device__ { };

  // not an extended lambda: explicitly annotated with only '__host__'
  auto lam4 = [] __host__ { };
}

__host__ __device__ void foo_host_device(void) {
  // not an extended lambda: no explicit execution space annotations
  auto lam1 = [] { };

  // extended __device__ lambda
  auto lam2 = [] __device__ { };

  // extended __host__ __device__ lambda
  auto lam3 = [] __host__ __device__ { };

  // not an extended lambda: explicitly annotated with only '__host__'
  auto lam4 = [] __host__ { };
}

__device__ void foo_device(void) {
  // none of the lambdas within this function are extended lambdas,
  // because the enclosing function is not a __host__ or __host__ __device__
  // function.
  auto lam1 = [] { };
  auto lam2 = [] __device__ { };
  auto lam3 = [] __host__ __device__ { };
  auto lam4 = [] __host__ { };
}

// lam1 and lam2 are not extended lambdas because they are not defined
// within a __host__ or __host__ __device__ function.
auto lam1 = [] { };
auto lam2 = [] __host__ __device__ { };
```

### 18.7.1. Extended Lambda Type Traits[](#extended-lambda-type-traits "Permalink to this headline")

The compiler provides type traits to detect closure types for extended lambdas at compile time:

`__nv_is_extended_device_lambda_closure_type(type)`: If ‘type’ is the closure class created for an extended `__device__` lambda, then the trait is true, otherwise it is false.

`__nv_is_extended_device_lambda_with_preserved_return_type(type)`: If ‘type’ is the closure class created for an extended `__device__` lambda and the lambda is defined with trailing return type (with restriction), then the trait is true, otherwise it is false. If the trailing return type definition refers to any lambda parameter name, the return type is not preserved.

`__nv_is_extended_host_device_lambda_closure_type(type)`: If ‘type’ is the closure class created for an extended `__host__ __device__` lambda, then the trait is true, otherwise it is false.

These traits can be used in all compilation modes, irrespective of whether lambdas or extended lambdas are enabled[24](#fn31).

Example:

```
#define IS_D_LAMBDA(X) __nv_is_extended_device_lambda_closure_type(X)
#define IS_DPRT_LAMBDA(X) __nv_is_extended_device_lambda_with_preserved_return_type(X)
#define IS_HD_LAMBDA(X) __nv_is_extended_host_device_lambda_closure_type(X)

auto lam0 = [] __host__ __device__ { };

void foo(void) {
  auto lam1 = [] { };
  auto lam2 = [] __device__ { };
  auto lam3 = [] __host__ __device__ { };
  auto lam4 = [] __device__ () --> double { return 3.14; }
  auto lam5 = [] __device__ (int x) --> decltype(&x) { return 0; }

  // lam0 is not an extended lambda (since defined outside function scope)
  static_assert(!IS_D_LAMBDA(decltype(lam0)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam0)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam0)), "");

  // lam1 is not an extended lambda (since no execution space annotations)
  static_assert(!IS_D_LAMBDA(decltype(lam1)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam1)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam1)), "");

  // lam2 is an extended __device__ lambda
  static_assert(IS_D_LAMBDA(decltype(lam2)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam2)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam2)), "");

  // lam3 is an extended __host__ __device__ lambda
  static_assert(!IS_D_LAMBDA(decltype(lam3)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam3)), "");
  static_assert(IS_HD_LAMBDA(decltype(lam3)), "");

  // lam4 is an extended __device__ lambda with preserved return type
  static_assert(IS_D_LAMBDA(decltype(lam4)), "");
  static_assert(IS_DPRT_LAMBDA(decltype(lam4)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam4)), "");

  // lam5 is not an extended __device__ lambda with preserved return type
  // because it references the operator()'s parameter types in the trailing return type.
  static_assert(IS_D_LAMBDA(decltype(lam5)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam5)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam5)), "");
}
```

### 18.7.2. Extended Lambda Restrictions[](#extended-lambda-restrictions "Permalink to this headline")

The CUDA compiler will replace an extended lambda expression with an instance of a placeholder type defined in namespace scope, before invoking the host compiler. The template argument of the placeholder type requires taking the address of a function enclosing the original extended lambda expression. This is required for the correct execution of any `__global__` function template whose template argument involves the closure type of an extended lambda. The *enclosing function* is computed as follows.

By definition, the extended lambda is present within the immediate or nested block scope of a `__host__` or `__host__ __device__` function. If this function is not the `operator()` of a lambda expression, then it is considered the enclosing function for the extended lambda. Otherwise, the extended lambda is defined within the immediate or nested block scope of the `operator()` of one or more enclosing lambda expressions. If the outermost such lambda expression is defined in the immediate or nested block scope of a function `F`, then `F` is the computed enclosing function, else the enclosing function does not exist.

Example:

```
void foo(void) {
  // enclosing function for lam1 is "foo"
  auto lam1 = [] __device__ { };

  auto lam2 = [] {
     auto lam3 = [] {
        // enclosing function for lam4 is "foo"
        auto lam4 = [] __host__ __device__ { };
     };
  };
}

auto lam6 = [] {
  // enclosing function for lam7 does not exist
  auto lam7 = [] __host__ __device__ { };
};
```

Here are the restrictions on extended lambdas:

1. An extended lambda cannot be defined inside another extended lambda expression.

   Example:

   ```
   void foo(void) {
     auto lam1 = [] __host__ __device__  {
       // error: extended lambda defined within another extended lambda
       auto lam2 = [] __host__ __device__ { };
     };
   }
   ```
2. An extended lambda cannot be defined inside a generic lambda expression.

   Example:

   ```
   void foo(void) {
     auto lam1 = [] (auto) {
       // error: extended lambda defined within a generic lambda
       auto lam2 = [] __host__ __device__ { };
     };
   }
   ```
3. If an extended lambda is defined within the immediate or nested block scope of one or more nested lambda expression, the outermost such lambda expression must be defined inside the immediate or nested block scope of a function.

   Example:

   ```
   auto lam1 = []  {
     // error: outer enclosing lambda is not defined within a
     // non-lambda-operator() function.
     auto lam2 = [] __host__ __device__ { };
   };
   ```
4. The enclosing function for the extended lambda must be named and its address can be taken. If the enclosing function is a class member, then the following conditions must be satisfied:

   * All classes enclosing the member function must have a name.
   * The member function must not have private or protected access within its parent class.
   * All enclosing classes must not have private or protected access within their respective parent classes.

   Example:

   ```
   void foo(void) {
     // OK
     auto lam1 = [] __device__ { return 0; };
     {
       // OK
       auto lam2 = [] __device__ { return 0; };
       // OK
       auto lam3 = [] __device__ __host__ { return 0; };
     }
   }

   struct S1_t {
     S1_t(void) {
       // Error: cannot take address of enclosing function
       auto lam4 = [] __device__ { return 0; };
     }
   };

   class C0_t {
     void foo(void) {
       // Error: enclosing function has private access in parent class
       auto temp1 = [] __device__ { return 10; };
     }
     struct S2_t {
       void foo(void) {
         // Error: enclosing class S2_t has private access in its
         // parent class
         auto temp1 = [] __device__ { return 10; };
       }
     };
   };
   ```
5. It must be possible to take the address of the enclosing routine unambiguously, at the point where the extended lambda has been defined. This may not be feasible in some cases e.g. when a class typedef shadows a template type argument of the same name.

   Example:

   ```
   template <typename> struct A {
     typedef void Bar;
     void test();
   };

   template<> struct A<void> { };

   template <typename Bar>
   void A<Bar>::test() {
     /* In code sent to host compiler, nvcc will inject an
        address expression here, of the form:
        (void (A< Bar> ::*)(void))(&A::test))

        However, the class typedef 'Bar' (to void) shadows the
        template argument 'Bar', causing the address
        expression in A<int>::test to actually refer to:
        (void (A< void> ::*)(void))(&A::test))

        ..which doesn't take the address of the enclosing
        routine 'A<int>::test' correctly.
     */
     auto lam1 = [] __host__ __device__ { return 4; };
   }

   int main() {
     A<int> xxx;
     xxx.test();
   }
   ```
6. An extended lambda cannot be defined in a class that is local to a function.

   Example:

   ```
   void foo(void) {
     struct S1_t {
       void bar(void) {
         // Error: bar is member of a class that is local to a function.
         auto lam4 = [] __host__ __device__ { return 0; };
       }
     };
   }
   ```
7. The enclosing function for an extended lambda cannot have deduced return type.

   Example:

   ```
   auto foo(void) {
     // Error: the return type of foo is deduced.
     auto lam1 = [] __host__ __device__ { return 0; };
   }
   ```
8. \_\_host\_\_ \_\_device\_\_ extended lambdas cannot be generic lambdas.

   Example:

   ```
   void foo(void) {
     // Error: __host__ __device__ extended lambdas cannot be
     // generic lambdas.
     auto lam1 = [] __host__ __device__ (auto i) { return i; };

     // Error: __host__ __device__ extended lambdas cannot be
     // generic lambdas.
     auto lam2 = [] __host__ __device__ (auto ...i) {
                  return sizeof...(i);
                 };
   }
   ```
9. If the enclosing function is an instantiation of a function template or a member function template, and/or the function is a member of a class template, the template(s) must satisfy the following constraints:

   * The template must have at most one variadic parameter, and it must be listed last in the template parameter list.
   * The template parameters must be named.
   * The template instantiation argument types cannot involve types that are either local to a function (except for closure types for extended lambdas), or are private or protected class members.

   Example:

   ```
   template <typename T>
   __global__ void kern(T in) { in(); }

   template <typename... T>
   struct foo {};

   template < template <typename...> class T, typename... P1,
             typename... P2>
   void bar1(const T<P1...>, const T<P2...>) {
     // Error: enclosing function has multiple parameter packs
     auto lam1 =  [] __device__ { return 10; };
   }

   template < template <typename...> class T, typename... P1,
             typename T2>
   void bar2(const T<P1...>, T2) {
     // Error: for enclosing function, the
     // parameter pack is not last in the template parameter list.
     auto lam1 =  [] __device__ { return 10; };
   }

   template <typename T, T>
   void bar3(void) {
     // Error: for enclosing function, the second template
     // parameter is not named.
     auto lam1 =  [] __device__ { return 10; };
   }

   int main() {
     foo<char, int, float> f1;
     foo<char, int> f2;
     bar1(f1, f2);
     bar2(f1, 10);
     bar3<int, 10>();
   }
   ```

   Example:

   ```
   template <typename T>
   __global__ void kern(T in) { in(); }

   template <typename T>
   void bar4(void) {
     auto lam1 =  [] __device__ { return 10; };
     kern<<<1,1>>>(lam1);
   }

   struct C1_t { struct S1_t { }; friend int main(void); };
   int main() {
     struct S1_t { };
     // Error: enclosing function for device lambda in bar4
     // is instantiated with a type local to main.
     bar4<S1_t>();

     // Error: enclosing function for device lambda in bar4
     // is instantiated with a type that is a private member
     // of a class.
     bar4<C1_t::S1_t>();
   }
   ```
10. With Visual Studio host compilers, the enclosing function must have external linkage. The restriction is present because this host compiler does not support using the address of non-extern linkage functions as template arguments, which is needed by the CUDA compiler transformations to support extended lambdas.
11. With Visual Studio host compilers, an extended lambda shall not be defined within the body of an ‘if-constexpr’ block.
12. An extended lambda has the following restrictions on captured variables:

    * In the code sent to the host compiler, the variable may be passed by value to a sequence of helper functions before being used to direct-initialize the field of the class type used to represent the closure type for the extended lambda[25](#fn32).
    * A variable can only be captured by value.
    * A variable of array type cannot be captured if the number of array dimensions is greater than 7.
    * For a variable of array type, in the code sent to the host compiler, the closure type’s array field is first default-initialized, and then each element of the array field is copy-assigned from the corresponding element of the captured array variable. Therefore, the array element type must be default-constructible and copy-assignable in host code.
    * A function parameter that is an element of a variadic argument pack cannot be captured.
    * The type of the captured variable cannot involve types that are either local to a function (except for closure types of extended lambdas), or are private or protected class members.
    * For a \_\_host\_\_ \_\_device\_\_ extended lambda, the types used in the return or parameter types of the lambda expression’s `operator()` cannot involve types that are either local to a function (except for closure types of extended lambdas), or are private or protected class members.
    * Init-capture is not supported for \_\_host\_\_ \_\_device\_\_ extended lambdas. Init-capture is supported for \_\_device\_\_ extended lambdas, except when the init-capture is of array type or of type `std::initializer_list`.
    * The function call operator for an extended lambda is not constexpr. The closure type for an extended lambda is not a literal type. The constexpr and consteval specifier cannot be used in the declaration of an extended lambda.
    * A variable cannot be implicitly captured inside an if-constexpr block lexically nested inside an extended lambda, unless it has already been implicitly captured earlier outside the if-constexpr block or appears in the explicit capture list for the extended lambda (see example below).

    Example

    ```
    void foo(void) {
      // OK: an init-capture is allowed for an
      // extended __device__ lambda.
      auto lam1 = [x = 1] __device__ () { return x; };

      // Error: an init-capture is not allowed for
      // an extended __host__ __device__ lambda.
      auto lam2 = [x = 1] __host__ __device__ () { return x; };

      int a = 1;
      // Error: an extended __device__ lambda cannot capture
      // variables by reference.
      auto lam3 = [&a] __device__ () { return a; };

      // Error: by-reference capture is not allowed
      // for an extended __device__ lambda.
      auto lam4 = [&x = a] __device__ () { return x; };

      struct S1_t { };
      S1_t s1;
      // Error: a type local to a function cannot be used in the type
      // of a captured variable.
      auto lam6 = [s1] __device__ () { };

      // Error: an init-capture cannot be of type std::initializer_list.
      auto lam7 = [x = {11}] __device__ () { };

      std::initializer_list<int> b = {11,22,33};
      // Error: an init-capture cannot be of type std::initializer_list.
      auto lam8 = [x = b] __device__ () { };

      // Error scenario (lam9) and supported scenarios (lam10, lam11)
      // for capture within 'if-constexpr' block
      int yyy = 4;
      auto lam9 = [=] __device__ {
        int result = 0;
        if constexpr(false) {
          //Error: An extended __device__ lambda cannot first-capture
          //      'yyy' in constexpr-if context
          result += yyy;
        }
        return result;
      };

      auto lam10 = [yyy] __device__ {
        int result = 0;
        if constexpr(false) {
          //OK: 'yyy' already listed in explicit capture list for the extended lambda
          result += yyy;
        }
        return result;
      };

      auto lam11 = [=] __device__ {
        int result = yyy;
        if constexpr(false) {
          //OK: 'yyy' already implicit captured outside the 'if-constexpr' block
          result += yyy;
        }
        return result;
      };
    }
    ```
13. When parsing a function, the CUDA compiler assigns a counter value to each extended lambda within that function. This counter value is used in the substituted named type passed to the host compiler. Hence, whether or not an extended lambda is defined within a function should not depend on a particular value of `__CUDA_ARCH__`, or on `__CUDA_ARCH__` being undefined.

    Example

    ```
    template <typename T>
    __global__ void kernel(T in) { in(); }

    __host__ __device__ void foo(void) {
      // Error: the number and relative declaration
      // order of extended lambdas depends on
      // __CUDA_ARCH__
    #if defined(__CUDA_ARCH__)
      auto lam1 = [] __device__ { return 0; };
      auto lam1b = [] __host___ __device__ { return 10; };
    #endif
      auto lam2 = [] __device__ { return 4; };
      kernel<<<1,1>>>(lam2);
    }
    ```
14. As described above, the CUDA compiler replaces a `__device__` extended lambda defined in a host function with a placeholder type defined in namespace scope. Unless the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns true for the closure type of the extended lambda, the placeholder type does not define a `operator()` function equivalent to the original lambda declaration. An attempt to determine the return type or parameter types of the `operator()` function of such a lambda may therefore work incorrectly in host code, as the code processed by the host compiler will be semantically different than the input code processed by the CUDA compiler. However, it is OK to introspect the return type or parameter types of the `operator()` function within device code. Note that this restriction does not apply to `__host__ __device__` extended lambdas, or to `__device__` extended lambdas for which the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns true.

    Example

    ```
    #include <type_traits>
    const char& getRef(const char* p) { return *p; }

    void foo(void) {
      auto lam1 = [] __device__ { return "10"; };

      // Error: attempt to extract the return type
      // of a __device__ lambda in host code
      std::result_of<decltype(lam1)()>::type xx1 = "abc";


      auto lam2 = [] __host__ __device__  { return "10"; };

      // OK : lam2 represents a __host__ __device__ extended lambda
      std::result_of<decltype(lam2)()>::type xx2 = "abc";

      auto lam3 = []  __device__ () -> const char * { return "10"; };

      // OK : lam3 represents a __device__ extended lambda with preserved return type
      std::result_of<decltype(lam3)()>::type xx2 = "abc";
      static_assert( std::is_same_v< std::result_of<decltype(lam3)()>::type, const char *>);

      auto lam4 = [] __device__ (char x) -> decltype(getRef(&x)) { return 0; };
      // lam4's return type is not preserved because it references the operator()'s
      // parameter types in the trailing return type.
      static_assert( ! __nv_is_extended_device_lambda_with_preserved_return_type(decltype(lam4)), "" );
    }
    ```
15. For an extended device lambda:
    - Introspecting the parameter type of operator() is only supported in device code.
    - Introspecting the return type of operator() is supported only in device code, unless the trait function \_\_nv\_is\_extended\_device\_lambda\_with\_preserved\_return\_type() returns true.
16. If the functor object represented by an extended lambda is passed from host to device code (e.g., as the argument of a `__global__` function), then any expression in the body of the lambda expression that captures variables must be remain unchanged irrespective of whether the `__CUDA_ARCH__` macro is defined, and whether the macro has a particular value. This restriction arises because the lambda’s closure class layout depends on the order in which captured variables are encountered when the compiler processes the lambda expression; the program may execute incorrectly if the closure class layout differs in device and host compilation.

    Example

    ```
    __device__ int result;

    template <typename T>
    __global__ void kernel(T in) { result = in(); }

    void foo(void) {
      int x1 = 1;
      auto lam1 = [=] __host__ __device__ {
        // Error: "x1" is only captured when __CUDA_ARCH__ is defined.
    #ifdef __CUDA_ARCH__
        return x1 + 1;
    #else
        return 10;
    #endif
      };
      kernel<<<1,1>>>(lam1);
    }
    ```
17. As described previously, the CUDA compiler replaces an extended `__device__` lambda expression with an instance of a placeholder type in the code sent to the host compiler. This placeholder type does not define a pointer-to-function conversion operator in host code, however the conversion operator is provided in device code. Note that this restriction does not apply to `__host__ __device__` extended lambdas.

    Example

    ```
    template <typename T>
    __global__ void kern(T in) {
      int (*fp)(double) = in;

      // OK: conversion in device code is supported
      fp(0);
      auto lam1 = [](double) { return 1; };

      // OK: conversion in device code is supported
      fp = lam1;
      fp(0);
    }

    void foo(void) {
      auto lam_d = [] __device__ (double) { return 1; };
      auto lam_hd = [] __host__ __device__ (double) { return 1; };
      kern<<<1,1>>>(lam_d);
      kern<<<1,1>>>(lam_hd);

      // OK : conversion for __host__ __device__ lambda is supported
      // in host code
      int (*fp)(double) = lam_hd;

      // Error: conversion for __device__ lambda is not supported in
      // host code.
      int (*fp2)(double) = lam_d;
    }
    ```
18. As described previously, the CUDA compiler replaces an extended `__device__` or `__host__ __device__` lambda expression with an instance of a placeholder type in the code sent to the host compiler. This placeholder type may define C++ special member functions (e.g. constructor, destructor). As a result, some standard C++ type traits may return different results for the closure type of the extended lambda, in the CUDA frontend compiler versus the host compiler. The following type traits are affected: `std::is_trivially_copyable`, `std::is_trivially_constructible`, `std::is_trivially_copy_constructible`, `std::is_trivially_move_constructible`, `std::is_trivially_destructible`.

    Care must be taken that the results of these type traits are not used in `__global__` function template instantiation or in `__device__ / __constant__ / __managed__` variable template instantiation.

    Example

    ```
    template <bool b>
    void __global__ foo() { printf("hi"); }

    template <typename T>
    void dolaunch() {

    // ERROR: this kernel launch may fail, because CUDA frontend compiler
    // and host compiler may disagree on the result of
    // std::is_trivially_copyable() trait on the closure type of the
    // extended lambda
    foo<std::is_trivially_copyable<T>::value><<<1,1>>>();
    cudaDeviceSynchronize();
    }

    int main() {
    int x = 0;
    auto lam1 = [=] __host__ __device__ () { return x; };
    dolaunch<decltype(lam1)>();
    }
    ```

The CUDA compiler will generate compiler diagnostics for a subset of cases described in 1-12; no diagnostic will be generated for cases 13-17, but the host compiler may fail to compile the generated code.

### 18.7.3. Notes on \_\_host\_\_ \_\_device\_\_ lambdas[](#notes-on-host-device-lambdas "Permalink to this headline")

Unlike `__device__` lambdas, `__host__ __device__` lambdas can be called from host code. As described earlier, the CUDA compiler replaces an extended lambda expression defined in host code with an instance of a named placeholder type. The placeholder type for an extended `__host__ __device__` lambda invokes the original lambda’s `operator()` with an indirect function call [24](#fn31).

The presence of the indirect function call may cause an extended `__host__ __device__` lambda to be less optimized by the host compiler than lambdas that are implicitly or explicitly `__host__` only. In the latter case, the host compiler can easily inline the body of the lambda into the calling context. But in case of an extended `__host__                                  __device__` lambda, the host compiler encounters the indirect function call and may not be able to easily inline the original `__host__ __device__` lambda body.

### 18.7.4. \*this Capture By Value[](#this-capture-by-value "Permalink to this headline")

When a lambda is defined within a non-static class member function, and the body of the lambda refers to a class member variable, C++11/C++14 rules require that the `this` pointer of the class is captured by value, instead of the referenced member variable. If the lambda is an extended `__device__` or `__host__``__device__` lambda defined in a host function, and the lambda is executed on the GPU, accessing the referenced member variable on the GPU will cause a run time error if the `this` pointer points to host memory.

Example:

```
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("\n value = %d", in()); }

struct S1_t {
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };

  void doit(void) {

    auto lam1 = [=] __device__ {
       // reference to "xxx" causes
       // the 'this' pointer (S1_t*) to be captured by value
       return xxx + 1;

    };

    // Kernel launch fails at run time because 'this->xxx'
    // is not accessible from the GPU
    foo<<<1,1>>>(lam1);
    cudaDeviceSynchronize();
  }
};

int main(void) {
  S1_t s1;
  s1.doit();
}
```

C++17 solves this problem by adding a new “\*this” capture mode. In this mode, the compiler makes a copy of the object denoted by “\*this” instead of capturing the pointer `this` by value. The “\*this” capture mode is described in more detail here: `http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html` .

The CUDA compiler supports the “\*this” capture mode for lambdas defined within `__device__` and `__global__` functions and for extended `__device__` lambdas defined in host code, when the `--extended-lambda` nvcc flag is used.

Here’s the above example modified to use “\*this” capture mode:

```
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("\n value = %d", in()); }

struct S1_t {
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };

  void doit(void) {

    // note the "*this" capture specification
    auto lam1 = [=, *this] __device__ {

       // reference to "xxx" causes
       // the object denoted by '*this' to be captured by
       // value, and the GPU code will access copy_of_star_this->xxx
       return xxx + 1;

    };

    // Kernel launch succeeds
    foo<<<1,1>>>(lam1);
    cudaDeviceSynchronize();
  }
};

int main(void) {
  S1_t s1;
  s1.doit();
}
```

“\*this” capture mode is not allowed for unannotated lambdas defined in host code, or for extended `__host__``__device__` lambdas, unless “\*this” capture is enabled by the selected language dialect. Examples of supported and unsupported usage:

```
struct S1_t {
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };

  void host_func(void) {

    // OK: use in an extended __device__ lambda
    auto lam1 = [=, *this] __device__ { return xxx; };

    // Use in an extended __host__ __device__ lambda
    // Error if *this capture not enabled by language dialect
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };

    // Use in an unannotated lambda in host function
    // Error if *this capture not enabled by language dialect
    auto lam3 = [=, *this]  { return xxx; };
  }

  __device__ void device_func(void) {

    // OK: use in a lambda defined in a __device__ function
    auto lam1 = [=, *this] __device__ { return xxx; };

    // OK: use in a lambda defined in a __device__ function
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };

    // OK: use in a lambda defined in a __device__ function
    auto lam3 = [=, *this]  { return xxx; };
  }

   __host__ __device__ void host_device_func(void) {

    // OK: use in an extended __device__ lambda
    auto lam1 = [=, *this] __device__ { return xxx; };

    // Use in an extended __host__ __device__ lambda
    // Error if *this capture not enabled by language dialect
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };

    // Use in an unannotated lambda in a __host__ __device__ function
    // Error if *this capture not enabled by language dialect
    auto lam3 = [=, *this]  { return xxx; };
  }
};
```

### 18.7.5. Additional Notes[](#additional-notes "Permalink to this headline")

1. `ADL Lookup`: As described earlier, the CUDA compiler will replace an extended lambda expression with an instance of a placeholder type, before invoking the host compiler. One template argument of the placeholder type uses the address of the function enclosing the original lambda expression. This may cause additional namespaces to participate in argument dependent lookup (ADL), for any host function call whose argument types involve the closure type of the extended lambda expression. This may cause an incorrect function to be selected by the host compiler.

   Example:

   ```
   namespace N1 {
     struct S1_t { };
     template <typename T>  void foo(T);
   };

   namespace N2 {
     template <typename T> int foo(T);

     template <typename T>  void doit(T in) {     foo(in);  }
   }

   void bar(N1::S1_t in) {
     /* extended __device__ lambda. In the code sent to the host compiler, this
        is replaced with the placeholder type instantiation expression
        ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::S1_t in),(&bar),1> > { }'

        As a result, the namespace 'N1' participates in ADL lookup of the
        call to "foo" in the body of N2::doit, causing ambiguity.
     */
     auto lam1 = [=] __device__ { };
     N2::doit(lam1);
   }
   ```

   In the example above, the CUDA compiler replaced the extended lambda with a placeholder type that involves the `N1` namespace. As a result, the namespace `N1` participates in the ADL lookup for `foo(in)` in the body of `N2::doit`, and host compilation fails because multiple overload candidates `N1::foo` and `N2::foo` are found.

## 18.8. Relaxed Constexpr (-expt-relaxed-constexpr)[](#relaxed-constexpr-expt-relaxed-constexpr "Permalink to this headline")

By default, the following cross-execution space calls are not supported:

1. Calling a `__device__`-only `constexpr` function from a `__host__` function during host code generation phase (i.e, when `__CUDA_ARCH__` macro is undefined). Example:

   > ```
   > constexpr __device__ int D() { return 0; }
   > int main() {
   >     int x = D();  //ERROR: calling a __device__-only constexpr function from host code
   > }
   > ```
2. Calling a `__host__`-only `constexpr` function from a `__device__` or `__global__` function, during device code generation phase (i.e. when `__CUDA_ARCH__` macro is defined). Example:

   > ```
   > constexpr  int H() { return 0; }
   > __device__ void dmain()
   > {
   >     int x = H();  //ERROR: calling a __host__-only constexpr function from device code
   > }
   > ```

The experimental flag `-expt-relaxed-constexpr` can be used to relax this constraint. When this flag is specified, the compiler will support cross execution space calls described above, as follows:

1. A cross-execution space call to a constexpr function is supported if it occurs in a context that requires constant evaluation, e.g., in the initializer of a constexpr variable. Example:

   > ```
   > constexpr __host__ int H(int x) { return x+1; };
   > __global__ void doit() {
   > constexpr int val = H(1); // OK: call is in a context that
   >                           // requires constant evaluation.
   > }
   >
   > constexpr __device__ int D(int x) { return x+1; }
   > int main() {
   > constexpr int val = D(1); // OK: call is in a context that
   >                           // requires constant evaluation.
   > }
   > ```
2. Otherwise:

   > 1. During device code generation, device code is generated for the body of a `__host__`-only constexpr function `H`, unless `H` is not used or is only called in a constexpr context. Example:
   >
   >    > ```
   >    > // NOTE: "H" is emitted in generated device code because it is
   >    > // called from device code in a non-constexpr context
   >    > constexpr __host__ int H(int x) { return x+1; }
   >    >
   >    > __device__ int doit(int in) {
   >    >   in = H(in);  // OK, even though argument is not a constant expression
   >    >   return in;
   >    > }
   >    > ```
   > 2. **All code restrictions applicable to a ``\_\_device\_\_`` function are also applicable to the ``constexpr host``-only function ``H`` that is called from device code. However, compiler may not emit any build time diagnostics for ``H`` for these restrictions** [8](#frelaxedconstexpr1) .
   >
   >    > For example, the following code patterns are unsupported in the body of `H` (as with any `__device__` function), but no compiler diagnostic may be generated:
   >    >
   >    > > * ODR-use of a host variable or `__host__`-only non-constexpr function. Example:
   >    > >
   >    > >   > ```
   >    > >   > int qqq, www;
   >    > >   >
   >    > >   > constexpr __host__ int* H(bool b) { return b ? &qqq : &www; };
   >    > >   >
   >    > >   > __device__ int doit(bool flag) {
   >    > >   >   int *ptr;
   >    > >   >   ptr = H(flag); // ERROR: H() attempts to refer to host variables 'qqq' and 'www'.
   >    > >   >                  // code will compile, but will NOT execute correctly.
   >    > >   >   return *ptr;
   >    > >   > }
   >    > >   > ```
   >    > > * Use of exceptions (`throw/catch`) and RTTI (`typeid, dynamic_cast`). Example:
   >    > >
   >    > >   > ```
   >    > >   > struct Base { };
   >    > >   > struct Derived : public Base { };
   >    > >   >
   >    > >   > // NOTE: "H" is emitted in generated device code
   >    > >   > constexpr int H(bool b, Base *ptr) {
   >    > >   >   if (b) {
   >    > >   >     return 1;
   >    > >   >   } else if (typeid(ptr) == typeid(Derived)) { // ERROR: use of typeid in code executing on the GPU
   >    > >   >     return 2;
   >    > >   >   } else {
   >    > >   >     throw int{4}; // ERROR: use of throw in code executing on the GPU
   >    > >   >   }
   >    > >   > }
   >    > >   > __device__ void doit(bool flag) {
   >    > >   >   int val;
   >    > >   >   Derived d;
   >    > >   >   val = H(flag, &d); //ERROR: H() attempts use typeid and throw(), which are not allowed in code that executes on the GPU
   >    > >   > }
   >    > >   > ```
   > 3. During host code generation, the body of a `__device__`-only constexpr function `D` is preserved in the code sent to the host compiler. If the body of `D` attempts to ODR-use a namespace scope device variable or a `__device__`-only non-constexpr function, then the call to `D` from host code is not supported (code may build without compiler diagnostics, but may behave incorrectly at run time). Example:
   >
   >    > ```
   >    > __device__ int qqq, www;
   >    > constexpr __device__ int* D(bool b) { return b ? &qqq : &www; };
   >    >
   >    > int doit(bool flag) {
   >    >   int *ptr;
   >    >   ptr = D(flag); // ERROR: D() attempts to refer to device variables 'qqq' and 'www'
   >    >                  // code will compile, but will NOT execute correctly.
   >    >   return *ptr;
   >    > }
   >    > ```
   > 4. **Note: Given above restrictions and lack of compiler diagnostics for incorrect usage, be careful when calling a constexpr \_\_host\_\_ function in the standard C++ headers from device code**, since the implementation of the function will vary depending on the host platform, e.g., based on the `libstdc++` version for gcc host compiler. Such code may break silently when being ported to a different platform or host compiler version (if the target C++ library implementation odr-uses a host code variable or function, as described earlier).
   >
   >    > Example:
   >    >
   >    > ```
   >    > __device__ int get(int in) {
   >    >  int val = std::foo(in); // "std::foo" is constexpr function defined in the host compiler's standard library header
   >    >                          // WARNING: if std::foo implementation ODR-uses host variables or functions,
   >    >                          // code will not work correctly
   >    > }
   >    > ```

[8](#id380)
:   Diagnostics are usually generated during parsing, but the host-only function `H` may already have been parsed before the call to `H` from device code is encountered later in the translation unit.

## 18.9. Code Samples[](#code-samples "Permalink to this headline")

### 18.9.1. Data Aggregation Class[](#data-aggregation-class "Permalink to this headline")

```
class PixelRGBA {
public:
    __device__ PixelRGBA(): r_(0), g_(0), b_(0), a_(0) { }

    __device__ PixelRGBA(unsigned char r, unsigned char g,
                         unsigned char b, unsigned char a = 255):
                         r_(r), g_(g), b_(b), a_(a) { }

private:
    unsigned char r_, g_, b_, a_;

    friend PixelRGBA operator+(const PixelRGBA&, const PixelRGBA&);
};

__device__
PixelRGBA operator+(const PixelRGBA& p1, const PixelRGBA& p2)
{
    return PixelRGBA(p1.r_ + p2.r_, p1.g_ + p2.g_,
                     p1.b_ + p2.b_, p1.a_ + p2.a_);
}

__device__ void func(void)
{
    PixelRGBA p1, p2;
    // ...      // Initialization of p1 and p2 here
    PixelRGBA p3 = p1 + p2;
}
```

### 18.9.2. Derived Class[](#derived-class "Permalink to this headline")

```
__device__ void* operator new(size_t bytes, MemoryPool& p);
__device__ void operator delete(void*, MemoryPool& p);
class Shape {
public:
    __device__ Shape(void) { }
    __device__ void putThis(PrintBuffer *p) const;
    __device__ virtual void Draw(PrintBuffer *p) const {
         p->put("Shapeless");
    }
    __device__ virtual ~Shape() {}
};
class Point : public Shape {
public:
    __device__ Point() : x(0), y(0) {}
    __device__ Point(int ix, int iy) : x(ix), y(iy) { }
    __device__ void PutCoord(PrintBuffer *p) const;
    __device__ void Draw(PrintBuffer *p) const;
    __device__ ~Point() {}
private:
    int x, y;
};
__device__ Shape* GetPointObj(MemoryPool& pool)
{
    Shape* shape = new(pool) Point(rand(-20,10), rand(-100,-20));
    return shape;
}
```

### 18.9.3. Class Template[](#class-template "Permalink to this headline")

```
template <class T>
class myValues {
    T values[MAX_VALUES];
public:
    __device__ myValues(T clear) { ... }
    __device__ void setValue(int Idx, T value) { ... }
    __device__ void putToMemory(T* valueLocation) { ... }
};

template <class T>
void __global__ useValues(T* memoryBuffer) {
    myValues<T> myLocation(0);
    ...
}

__device__ void* buffer;

int main()
{
    ...
    useValues<int><<<blocks, threads>>>(buffer);
    ...
}
```

### 18.9.4. Function Template[](#function-template "Permalink to this headline")

```
template <typename T>
__device__ bool func(T x)
{
   ...
   return (...);
}

template <>
__device__ bool func<int>(T x) // Specialization
{
   return true;
}

// Explicit argument specification
bool result = func<double>(0.5);

// Implicit argument deduction
int x = 1;
bool result = func(x);
```

### 18.9.5. Functor Class[](#functor-class "Permalink to this headline")

```
class Add {
public:
    __device__  float operator() (float a, float b) const
    {
        return a + b;
    }
};

class Sub {
public:
    __device__  float operator() (float a, float b) const
    {
        return a - b;
    }
};

// Device code
template<class O> __global__
void VectorOperation(const float * A, const float * B, float * C,
                     unsigned int N, O op)
{
    unsigned int iElement = blockDim.x * blockIdx.x + threadIdx.x;
    if (iElement < N)
        C[iElement] = op(A[iElement], B[iElement]);
}

// Host code
int main()
{
    ...
    VectorOperation<<<blocks, threads>>>(v1, v2, v3, N, Add());
    ...
}
```

9
:   e.g., the `<<<...>>>` syntax for launching kernels.

10
:   This does not apply to entities that may be defined in more than one translation unit, such as compiler generated template instantiations.

[11](#id339)
:   The intent is to allow variable memory space specifiers for static variables in a `__host__ __device__` function during device compilation, but disallow it during host compilation

[12](#id349)
:   One way to debug suspected layout mismatch of a type `C` is to use `printf` to output the values of `sizeof(C)` and `offsetof(C, field)` in host and device code.

[13](#id355)
:   Note that this may negatively impact compile time due to presence of extra declarations.

[14](#id356)
:   At present, the `-std=c++11` flag is supported only for the following host compilers : gcc version >= 4.7, clang, icc >= 15, and xlc >= 13.1

[15](#id358)
:   including `operator()`

[16](#id360)
:   The restrictions are the same as with a non-constexpr callee function.

[17](#id361)
:   Note that the behavior of experimental flags may change in future compiler releases.

[18](#id363)
:   C++ Standard Section `[basic.types]`

[19](#id364)
:   C++ Standard Section `[expr.const]`

[20](#id368)
:   At present, the `-std=c++14` flag is supported only for the following host compilers : gcc version >= 5.1, clang version >= 3.7 and icc version >= 17

[21](#id370)
:   At present, the `-std=c++17` flag is supported only for the following host compilers : gcc version >= 7.0, clang version >= 8.0, Visual Studio version >= 2017, pgi compiler version >= 19.0, icc compiler version >= 19.0

[22](#id373)
:   At present, the `-std=c++20` flag is supported only for the following host compilers : gcc version >= 10.0, clang version >= 10.0, Visual Studio Version >= 2022 and nvc++ version >= 20.7.

[23](#id375)
:   When using the icc host compiler, this flag is only supported for icc >= 1800.

24([1](#id376),[2](#id379))
:   The traits will always return false if extended lambda mode is not active.

[25](#id378)
:   In contrast, the C++ standard specifies that the captured variable is used to direct-initialize the field of the closure type.
