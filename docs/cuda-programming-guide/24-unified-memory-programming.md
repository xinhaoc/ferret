# 24. Unified Memory Programming[](#unified-memory-programming "Permalink to this headline")

Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.

Note

This chapter applies to devices with compute capability 5.0 or higher unless stated otherwise.
For devices with compute capability lower than 5.0, refer to the CUDA toolkit documentation for CUDA 11.8.

This documentation on Unified Memory is divided into 3 parts:

* [General description of unified memory](#um-introduction)
* [Unified Memory on devices with full CUDA Unified Memory support](#um-pageable-systems)
* [Unified Memory on devices without full CUDA Unified Memory support](#um-no-pageable-systems)

## 24.1. Unified Memory Introduction[](#unified-memory-introduction "Permalink to this headline")

CUDA Unified Memory provides all processors with:

* A single *unified* memory pool, that is,
  a single pointer value enables all processors in the system
  (all CPUs, all GPUs, etc.)
  to access this memory with all of their native memory operations
  (pointer dereferences, atomics, etc.).
* Concurrent access to the unified memory pool from all processors in the system.

Unified Memory improves GPU programming in several ways:

* **Productivity**: GPU programs may access Unified Memory from GPU and CPU threads
  concurrently without needing to create separate allocations (`cudaMalloc()`) and
  copy memory manually back and forth (`cudaMemcpy*()`).
* **Performance**:

  + Data access speed may be maximized by migrating data towards processors that access it most frequently.
    Applications can trigger manual migration of data and may use hints to control migration heuristics.
  + Total system memory usage may be reduced by avoiding duplicating memory on both CPUs and GPUs.
* **Functionality**: It enables GPU programs to work on data that exceeds the GPU memory’s capacity.

With CUDA Unified Memory, data movement still takes place, and hints may improve performance.
These hints are not required for correctness or functionality, that is, programmers may focus on parallelizing
their applications across GPUs and CPUs first, and worry about data-movement later in the development cycle as a performance optimization.
Note that the physical location of data is invisible to a program and may be changed at any time,
but accesses to the data’s virtual address will remain valid and coherent from any processor regardless of locality.

There are two main ways to obtain CUDA Unified Memory:

* [System-Allocated Memory](#um-implicit-allocation): memory allocated on the host with system APIs:
  stack variables, global-/file-scope variables, `malloc()` / `mmap()`
  (see [System-Allocated Memory: in-depth examples](#um-system-allocator) for in-depth examples), thread locals, etc.
* [CUDA APIs that explicitly allocate Unified Memory](#um-explicit-allocation): memory allocated with, for example, `cudaMallocManaged()`,
  are available on more systems and may perform better than System-Allocated Memory.

### 24.1.1. System Requirements for Unified Memory[](#system-requirements-for-unified-memory "Permalink to this headline")

The following table shows the different levels of support for CUDA Unified Memory,
the device properties required to detect these levels of support
and links to the documentation specific to each level of support:

Table 31 Overview of levels of unified memory support[](#table-unified-memory-levels "Permalink to this table")


| Unified Memory Support Level | System device properties | Further documentation |
| --- | --- | --- |
| Full CUDA Unified Memory: all memory has full support. This includes System-Allocated and CUDA Managed Memory. | Set to 1: `pageableMemoryAccess`  [Systems with hardware acceleration](#um-system-allocator) also have the following properties set to 1:  `hostNativeAtomicSupported`, `pageableMemoryAccessUsesHostPageTables`, `directManagedMemAccessFromHost` | [Unified Memory on devices with full CUDA Unified Memory support](#um-pageable-systems) |
| Only CUDA Managed Memory has full support. | Set to 1: `concurrentManagedAccess`  Set to 0: `pageableMemoryAccess` | [Unified Memory on devices with only CUDA Managed Memory support](#um-cc60) |
| CUDA Managed Memory without full support: unified addressing but no concurrent access. | Set to 1: `managedMemory`  Set to 0: `concurrentManagedAccess` | [Unified Memory on Windows or devices with compute capability 5.x](#um-legacy-devices)  [CUDA for Tegra Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)  [Unified Memory on Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#effective-usage-of-unified-memory-on-tegra) |
| No Unified Memory support. | Set to 0: `managedMemory` | [CUDA for Tegra Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management) |

The behavior of an application that attempts to use Unified Memory on a system that does not support it is undefined.
The following properties enable CUDA applications to check the level of system support for Unified Memory, and
to be portable between systems with different levels of support:

* `pageableMemoryAccess`: This property is set to 1 on systems with CUDA Unified Memory support where
  all threads may access System-Allocated Memory and CUDA Managed Memory.
  These systems include NVIDIA Grace Hopper, IBM Power9 + Volta, and modern Linux systems with HMM enabled (see next bullet), among others.

  + Linux HMM requires Linux kernel version 6.1.24+, 6.2.11+ or 6.3+,
    devices with compute capability 7.5 or higher and
    a CUDA driver version 535+ installed with
    [Open Kernel Modules](http://download.nvidia.com/XFree86/Linux-x86_64/515.43.04/README/kernel_open.html).
* `concurrentManagedAccess`: This property is set to 1 on systems with full CUDA Managed Memory support.
  When this property is set to 0, there is only partial support for Unified Memory in CUDA Managed Memory.
  For Tegra support of Unified Memory, see
  [CUDA for Tegra Memory Management](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management).

A program may query the level of GPU support for CUDA Unified Memory, by querying the attributes in
[Table 31](#table-unified-memory-levels) using `cudaGetDeviceProperties()`.

### 24.1.2. Programming Model[](#um-opt-in "Permalink to this headline")

With CUDA Unified Memory, separate allocations between host and device, and explicit memory transfers between them, are no longer required.
Programs may allocate Unified Memory in the following ways:

* [System-Allocation APIs](#um-implicit-allocation): on [systems with full CUDA Unified Memory support](#um-requirements) via
  any system allocation of the host process (C’s `malloc()`, C++’s `new` operator, POSIX’s `mmap` and so on).
* [CUDA Managed Memory Allocation APIs](#um-explicit-allocation): via the `cudaMallocManaged()` API which is syntactically similar to `cudaMalloc()`.
* [CUDA Managed Variables](#um-language-integration): variables declared with `__managed__`, which are semantically similar to a `__device__` variable.

Most examples in this chapter provide at least two versions, one using CUDA Managed Memory and one using System-Allocated Memory.
Tabs allow you to choose between them. The following samples illustrate how Unified Memory simplifies CUDA programs:

System (`malloc()`)

|  |  |
| --- | --- |
| ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* d_ptr = nullptr;   // Does not require any unified memory support   cudaMalloc(&d_ptr, sizeof(int));   write_value<<<1, 1>>>(d_ptr, 1);   int h_value;   // Copy memory back to the host and synchronize   cudaMemcpy(&h_value, d_ptr, sizeof(int),              cudaMemcpyDefault);   printf("value = %d\n", h_value);    cudaFree(d_ptr);    return 0; } ``` | ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   // Requires System-Allocated Memory support   int* ptr = (int*)malloc(sizeof(int));   write_value<<<1, 1>>>(ptr, 1);   // Synchronize required   // (before, cudaMemcpy was synchronizing)   cudaDeviceSynchronize();   printf("value = %d\n", *ptr);    free(ptr);    return 0; } ``` |


System (Stack)

|  |  |
| --- | --- |
| ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* d_ptr = nullptr;   // Does not require any unified memory support   cudaMalloc(&d_ptr, sizeof(int));   write_value<<<1, 1>>>(d_ptr, 1);   int h_value;   // Copy memory back to the host and synchronize   cudaMemcpy(&h_value, d_ptr, sizeof(int),              cudaMemcpyDefault);   printf("value = %d\n", h_value);    cudaFree(d_ptr);    return 0; } ``` | ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   // Requires System-Allocated Memory support   int value;   write_value<<<1, 1>>>(&value, 1);   // Synchronize required   // (before, cudaMemcpy was synchronizing)   cudaDeviceSynchronize();   printf("value = %d\n", value);   return 0; } ``` |


Managed (`cudaMallocManaged()`)

|  |  |
| --- | --- |
| ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* d_ptr = nullptr;   // Does not require any unified memory support   cudaMalloc(&d_ptr, sizeof(int));   write_value<<<1, 1>>>(d_ptr, 1);   int h_value;   // Copy memory back to the host and synchronize   cudaMemcpy(&h_value, d_ptr, sizeof(int),              cudaMemcpyDefault);   printf("value = %d\n", h_value);    cudaFree(d_ptr);    return 0; } ``` | ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* ptr = nullptr;   // Requires CUDA Managed Memory support   cudaMallocManaged(&ptr, sizeof(int));   write_value<<<1, 1>>>(ptr, 1);   // Synchronize required   // (before, cudaMemcpy was synchronizing)   cudaDeviceSynchronize();   printf("value = %d\n", *ptr);    cudaFree(ptr);    return 0; } ``` |


Managed (`__managed__`)

|  |  |
| --- | --- |
| ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  int main() {   int* d_ptr = nullptr;   // Does not require any unified memory support   cudaMalloc(&d_ptr, sizeof(int));   write_value<<<1, 1>>>(d_ptr, 1);   int h_value;   // Copy memory back to the host and synchronize   cudaMemcpy(&h_value, d_ptr, sizeof(int),              cudaMemcpyDefault);   printf("value = %d\n", h_value);    cudaFree(d_ptr);    return 0; } ``` | ``` __global__ void write_value(int* ptr, int v) {   *ptr = v; }  // Requires CUDA Managed Memory support __managed__ int value;  int main() {   write_value<<<1, 1>>>(&value, 1);   // Synchronize required   // (before, cudaMemcpy was synchronizing)   cudaDeviceSynchronize();   printf("value = %d\n", value);   return 0; } ``` |

In the example above, the device writes a value which is then read by the host:

* **Without Unified Memory**: both host- and device-side storage for the written value is required (`h_value` and `d_ptr` in the example),
  as is an explicit copy between the two using `cudaMemcpy()`.
* **With Unified Memory**: device accesses data directly from the host. `ptr` / `value` may be used without a separate `h_value` / `d_ptr`
  allocation and no copy routine is required, greatly simplifying and reducing the size of the program. With:

  + **System Allocated**: no other changes required.
  + **Managed Memory**: data allocation changed to use `cudaMallocManaged()`, which returns a pointer valid from both host and device code.

#### 24.1.2.1. Allocation APIs for System-Allocated Memory[](#allocation-apis-for-system-allocated-memory "Permalink to this headline")

On [systems with full CUDA Unified Memory support](#um-requirements), all memory is unified memory.
This includes memory allocated with system allocation APIs, such as `malloc()`, `mmap()`, C++ `new()` operator,
and also automatic variables on CPU thread stacks, thread locals, global variables, and so on.

System-Allocated Memory may be populated on first touch, depending on the API and system settings used.
First touch means that:

* The allocation APIs allocate virtual memory and return immediately, and
* physical memory is populated when a thread accesses the memory for the first time.

Usually, the physical memory will be chosen “close” to the processor that thread is running on. For example,

* GPU thread accesses it first: physical GPU memory of GPU that thread runs on is chosen.
* CPU thread accesses it first: physical CPU memory in the memory NUMA node of the CPU core that thread runs on is chosen.

CUDA Unified Memory Hint and Prefetch APIs, `cudaMemAdvise` and `cudaMemPreftchAsync`, may be used on System-Allocated Memory.
These APIs are covered below in the [Data Usage Hints](#um-tuning-usage) section.

```
__global__ void printme(char *str) {
  printf(str);
}

int main() {
  // Allocate 100 bytes of memory, accessible to both Host and Device code
  char *s = (char*)malloc(100);
  // Physical allocation placed in CPU memory because host accesses "s" first
  strncpy(s, "Hello Unified Memory\n", 99);
  // Here we pass "s" to a kernel without explicitly copying
  printme<<< 1, 1 >>>(s);
  cudaDeviceSynchronize();
  // Free as for normal CUDA allocations
  cudaFree(s); 
  return  0;
}
```

#### 24.1.2.2. Allocation API for CUDA Managed Memory: `cudaMallocManaged()`[](#allocation-api-for-cuda-managed-memory-cudamallocmanaged "Permalink to this headline")

On systems with CUDA Managed Memory support, unified memory may be allocated using:

```
__host__ cudaError_t cudaMallocManaged(void **devPtr, size_t size);
```

This API is syntactically identical to `cudaMalloc()`: it allocates `size` bytes of managed memory and
sets `devPtr` to refer to the allocation.
CUDA Managed Memory is also deallocated with `cudaFree()`.

On [systems with full CUDA Managed Memory support](#um-requirements), managed memory allocations
may be accessed concurrently by all CPUs and GPUs in the system.
Replacing host calls to `cudaMalloc()` with `cudaMallocManaged()` does not impact program semantics on these systems;
device code is not able to call `cudaMallocManaged()`.

The following example shows the use of `cudaMallocManaged()`:

```
__global__ void printme(char *str) {
  printf(str);
}

int main() {
  // Allocate 100 bytes of memory, accessible to both Host and Device code
  char *s;
  cudaMallocManaged(&s, 100);
  // Note direct Host-code use of "s"
  strncpy(s, "Hello Unified Memory\n", 99);
  // Here we pass "s" to a kernel without explicitly copying
  printme<<< 1, 1 >>>(s);
  cudaDeviceSynchronize();
  // Free as for normal CUDA allocations
  cudaFree(s); 
  return  0;
}
```

Note

For systems that support CUDA Managed Memory allocations, but do not provide full support,
see [Coherency and Concurrency](#um-coherency-hd).
Implementation details (may change any time):

* Devices of compute capability 5.x allocate CUDA Managed Memory on the GPU.
* Devices of compute capability 6.x and greater populate the memory on first touch, just like System-Allocated Memory APIs.

#### 24.1.2.3. Global-Scope Managed Variables Using `__managed__`[](#global-scope-managed-variables-using-managed "Permalink to this headline")

CUDA `__managed__` variables behave as if they were allocated via `cudaMallocManaged()`
(see [Allocation API for CUDA Managed Memory: cudaMallocManaged()](#um-explicit-allocation)).
They simplify programs with global variables, making it particularly easy to exchange data between host and device
without manual allocations or copying.

On [systems with full CUDA Unified Memory support](#um-requirements),
file-scope or global-scope variables cannot be directly accessed by device code.
But a pointer to these variables may be passed to the kernel as an argument,
see [System-Allocated Memory: in-depth examples](#um-system-allocator) for examples.

System Allocator

```
__global__ void write_value(int* ptr, int v) {
  *ptr = v;
}

int main() {
  // Requires System-Allocated Memory support
  int value;
  write_value<<<1, 1>>>(&value, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", value);
  return 0;
}
```


Managed

```
__global__ void write_value(int* ptr, int v) {
  *ptr = v;
}

// Requires CUDA Managed Memory support
__managed__ int value;

int main() {
  write_value<<<1, 1>>>(&value, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", value);
  return 0;
}
```

Note the absence of explicit `cudaMemcpy()` commands and the fact that the written value `value` is visible on both CPU and GPU.

CUDA `__managed__` variable implies `__device__` and is equivalent to `__managed__ __device__`, which is also allowed.
Variables marked `__constant__` may not be marked as `__managed__`.

A valid CUDA context is necessary for the correct operation of `__managed__` variables.
Accessing `__managed__` variables can trigger CUDA context creation if a context for the current device hasn’t already been created.
In the example above, accessing `value` before the kernel launch triggers context creation on the default device.
In the absence of that access, the kernel launch would have triggered context creation.

C++ objects declared as `__managed__` are subject to certain specific constraints, particularly where static initializers are concerned.
Please refer to [C++ Language Support](#c-cplusplus-language-support) for a list of these constraints.

Note

For [devices with CUDA Managed Memory without full support](#um-requirements),
visibility of `__managed__` variables for asynchronous operations executing in CUDA streams
is discussed in the section on
[Managing Data Visibility and Concurrent CPU + GPU Access with Streams](#um-managing-data).

#### 24.1.2.4. Difference between Unified Memory and Mapped Memory[](#difference-between-unified-memory-and-mapped-memory "Permalink to this headline")

The main difference between Unified Memory and [Mapped Memory](#mapped-memory) is that
CUDA Mapped Memory does not guarantee that all kinds of memory accesses (for example atomics) are supported on all systems,
while Unified Memory does. The limited set of memory operations that are guaranteed to be portably supported by CUDA Mapped Memory
is available on more systems than Unified Memory.

#### 24.1.2.5. Pointer Attributes[](#um-pointer-attributes "Permalink to this headline")

CUDA Programs may check whether a pointer addresses a CUDA Managed Memory allocation by calling `cudaPointerGetAttributes()` and
testing whether the pointer attribute `value` is `cudaMemoryTypeManaged`.

This API returns `cudaMemoryTypeHost` for System-Allocated Memory that has been registered with `cudaHostRegister()`
and `cudaMemoryTypeUnregistered` for System-Allocated Memory that CUDA is unaware of.

Pointer attributes do not state where the memory resides, they state how the memory was allocated or registered.

The following example shows how to detect the type of pointer at runtime:

```
char const* kind(cudaPointerAttributes a, bool pma, bool cma) {
    switch(a.type) {
    case cudaMemoryTypeHost: return pma?
      "Unified: CUDA Host or Registered Memory" :
      "Not Unified: CUDA Host or Registered Memory";
    case cudaMemoryTypeDevice: return "Not Unified: CUDA Device Memory";
    case cudaMemoryTypeManaged: return cma?
      "Unified: CUDA Managed Memory" : "Not Unified: CUDA Managed Memory";
    case cudaMemoryTypeUnregistered: return pma?
      "Unified: System-Allocated Memory" :
      "Not Unified: System-Allocated Memory";
    default: return "unknown";
    }
}

void check_pointer(int i, void* ptr) {
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, ptr);
  int pma = 0, cma = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, device);
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, device);
  printf("Pointer %d: memory is %s\n", i, kind(attr, pma, cma));
}

__managed__ int managed_var = 5;

int main() {
  int* ptr[5];
  ptr[0] = (int*)malloc(sizeof(int));
  cudaMallocManaged(&ptr[1], sizeof(int));
  cudaMallocHost(&ptr[2], sizeof(int));
  cudaMalloc(&ptr[3], sizeof(int));
  ptr[4] = &managed_var;

  for (int i = 0; i < 5; ++i) check_pointer(i, ptr[i]);
  
  cudaFree(ptr[3]);
  cudaFreeHost(ptr[2]);
  cudaFree(ptr[1]);
  free(ptr[0]);
  return 0;
}
```

#### 24.1.2.6. Runtime detection of Unified Memory Support Level[](#runtime-detection-of-unified-memory-support-level "Permalink to this headline")

The following example shows how to detect the Unified Memory support level at runtime:

```
int main() {
  int d;
  cudaGetDevice(&d);

  int pma = 0;
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, d);
  printf("Full Unified Memory Support: %s\n", pma == 1? "YES" : "NO");
  
  int cma = 0;
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, d);
  printf("CUDA Managed Memory with full support: %s\n", cma == 1? "YES" : "NO");

  return 0;
}
```

#### 24.1.2.7. GPU Memory Oversubscription[](#gpu-memory-oversubscription "Permalink to this headline")

Unified Memory enables applications to *oversubscribe* the memory of any individual processor:
in other words they can allocate and share arrays larger than
the memory capacity of any individual processor in the system,
enabling among others out-of-core processing of datasets that do not fit within
a single GPU, without adding significant complexity to the programming model.

#### 24.1.2.8. Performance Hints[](#performance-hints "Permalink to this headline")

The following sections describes the available unified memory performance hints,
which may be used on all Unified Memory, for example, CUDA Managed memory or,
on [systems with full CUDA Unified Memory support](#um-requirements),
also all System-Allocated Memory.
These APIs are hints, that is, they do not impact the semantics of applications, only their peformance.
That is, they can be added or removed anywhere on any application without impacting its results.

CUDA Unified Memory may not always have all the information necessary to make
the best performance decisions related to unified memory.
These performance hints enable the application to provide CUDA with more information.

Note that applications should only use these hints if they improve their performance.

##### 24.1.2.8.1. Data Prefetching[](#data-prefetching "Permalink to this headline")

The `cudaMemPrefetchAsync` API is an asynchronous stream-ordered API that may migrate data to reside closer to the specified processor.
The data may be accessed while it is being prefetched.
The migration does not begin until all prior operations in the stream have completed,
and completes before any subsequent operation in the stream.

```
cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                 size_t count,
                                 struct cudaMemLocation location,
                                 unsigned int flags,
                                 cudaStream_t stream);
```

A memory region containing `[devPtr, devPtr + count)` may be migrated to
the destination device `location.id` if `location.type` is `cudaMemLocationTypeDevice`
- or CPU if `location.type` is `cudaMemLocationTypeHost` -
when the prefetch task is executed in the given `stream`.
For details on `flags`, see the current
[CUDA Runtime API documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).

Consider a simple code example below:

System Allocator

```
void test_prefetch_sam(cudaStream_t s) {
  char *data = (char*)malloc(N);
  init_data(data, N);                                         // execute on CPU
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemPrefetchAsync(data, N, location, s, 0 /* flags */);  // prefetch to GPU
  mykernel<<<(N + TPB - 1) / TPB, TPB, 0, s>>>(data, N);      // execute on GPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, N, location, s, 0 /* flags */);  // prefetch to CPU
  cudaStreamSynchronize(s);
  use_data(data, N);
  free(data);
}
```


Managed

```
void test_prefetch_managed(cudaStream_t s) {
  char *data;
  cudaMallocManaged(&data, N);
  init_data(data, N);                                         // execute on CPU
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemPrefetchAsync(data, N, location, s, 0 /* flags */);  // prefetch to GPU
  mykernel<<<(N + TPB - 1) / TPB, TPB, 0, s>>>(data, N);      // execute on GPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, N, location, s, 0 /* flags */);  // prefetch to CPU
  cudaStreamSynchronize(s);
  use_data(data, N);
  cudaFree(data);
}
```

##### 24.1.2.8.2. Memory Discarding[](#memory-discarding "Permalink to this headline")

The `cudaMemDiscardBatchAsync` API allows applications to inform the CUDA runtime that the contents
of specified memory ranges are no longer useful. The Unified Memory driver performs automatic memory
transfers due to fault-based migration or memory evictions to support device memory oversubscription.
These automatic memory transfers can sometimes be redundant, which severely decreases performance.
Marking an address range as ‘discard’ will inform the Unified Memory driver that the application has
consumed the contents in the range and there is no need to migrate this data on prefetches or page evictions
in order to make room for other allocations. Reading a discarded page without a subsequent write access
or prefetch will yield an indeterminate value. Whereas any new writes after the discard operation is guaranteed
to be seen by a subsequent read access. Concurrent accesses or prefetches to address ranges being discarded
will result in undefined behavior.

```
cudaError_t cudaMemDiscardBatchAsync(void **dptrs,
                                    size_t *sizes,
                                    size_t count,
                                    unsigned long long flags,
                                    cudaStream_t stream);
```

The function performs a batch of memory discards on address ranges specified in `dptrs` and `sizes` arrays.
Both arrays must be of the same length as specified by `count`. Each memory range must refer to
managed memory allocated via `cudaMallocManaged` or declared via `__managed__` variables.

The `cudaMemDiscardAndPrefetchBatchAsync` API combines both discard and prefetch operations. Calling
`cudaMemDiscardAndPrefetchBatchAsync` is semantically equivalent to calling `cudaMemDiscardBatchAsync`
followed by `cudaMemPrefetchBatchAsync`, but is more optimal. This is useful when the application needs
the memory to be on the target location but does not need the contents of the memory.

```
cudaError_t cudaMemDiscardAndPrefetchBatchAsync(void **dptrs,
                                               size_t *sizes,
                                               size_t count,
                                               struct cudaMemLocation *prefetchLocs,
                                               size_t *prefetchLocIdxs,
                                               size_t numPrefetchLocs,
                                               unsigned long long flags,
                                               cudaStream_t stream);
```

The `prefetchLocs` array specifies the destinations for prefetching, while `prefetchLocIdxs`
indicates which operations each prefetch location applies to. For example, if a batch has 10 operations
and the first 6 should be prefetched to one location while the remaining 4 to another, then
`numPrefetchLocs` would be 2, `prefetchLocIdxs` would be {0, 6}, and `prefetchLocs` would
contain the two destination locations.

**Important considerations:**

* Reading from a discarded range without a subsequent write or prefetch will return an indeterminate value
* The discard operation can be undone by writing to the range or prefetching it via `cudaMemPrefetchAsync`
* Any reads, writes, or prefetches that occur simultaneously with the discard operation result in undefined behavior
* All devices must have a non-zero value for `cudaDevAttrConcurrentManagedAccess`

##### 24.1.2.8.3. Data Usage Hints[](#data-usage-hints "Permalink to this headline")

When multiple processors simultaneously access the same data,
`cudaMemAdvise` may be used to hint how the data at
`[devPtr, devPtr + count)` will be accessed:

```
cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
```

Where `advice` may take the following values:

* `cudaMemAdviseSetReadMostly`:
  This implies that the data is mostly going to be read from and only occasionally written to.
  In general, it allows trading off read bandwidth for write bandwidth on this region.
  Example:

```
void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * TPB;  // 16 KiB
  // Allocate memory using cudaMallocManaged
  // (malloc may be used on systems with full CUDA Unified memory support)
  cudaMallocManaged(&dataPtr, dataSize);
  // Set the advice on the memory region
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);
  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written to in the outer loop on the CPU
    init_data(dataPtr, dataSize);
    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, 0 /* flags */, stream);
    }
    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, TPB, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  cudaFree(dataPtr);
}
```

* `cudaMemAdviseSetPreferredLocation`:
  In general, any memory may be migrated at any time to any location, for example,
  when a given processor is running out of physical memory.
  This hint tells the system that migrating this memory region away from
  its preferred location is undesired, by setting the preferred location for
  the data to be the physical memory belonging to device.
  Passing in a value of `cudaMemLocationTypeHost` for location.type sets the preferred location as CPU memory.
  Other hints, like `cudaMemPrefetchAsync`, may override this hint,
  leading the memory to be migrated away from its preferred location.

* `cudaMemAdviseSetAccessedBy`:
  In some systems, it may be beneficial for performance to establish a
  mapping into memory before accessing the data from a given processor.
  This hint tells the system that the data will be frequently accessed by `location.id`
  when `location.type` is `cudaMemLocationTypeDevice`,
  enabling the system to assume that creating these mappings pays off.
  This hint does not imply where the data should reside,
  but it can be combined with `cudaMemAdviseSetPreferredLocation` to specify that.

Each advice can be also unset by using one of the following values:
`cudaMemAdviseUnsetReadMostly`, `cudaMemAdviseUnsetPreferredLocation` and
`cudaMemAdviseUnsetAccessedBy`.

##### 24.1.2.8.4. Querying Data Usage Attributes on Managed Memory[](#querying-data-usage-attributes-on-managed-memory "Permalink to this headline")

A program can query memory range attributes assigned through `cudaMemAdvise`
or `cudaMemPrefetchAsync` on CUDA Managed Memory by using the following API:

```
cudaMemRangeGetAttribute(void *data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void *devPtr,
                         size_t count);
```

This function queries an attribute of the memory range starting at `devPtr` with a size of `count` bytes.
The memory range must refer to managed memory allocated via `cudaMallocManaged` or
declared via `__managed__` variables. It is possible to query the following attributes:

* `cudaMemRangeAttributeReadMostly`:
  the result returned will be 1 if the entire memory range has the `cudaMemAdviseSetReadMostly` attribute set, or 0 otherwise.
* `cudaMemRangeAttributePreferredLocation`:
  the result returned will be a GPU device id or `cudaCpuDeviceId` if the entire
  memory range has the corresponding processor as preferred location,
  otherwise `cudaInvalidDeviceId` will be returned.
  An application can use this query API to make decision about staging data through
  CPU or GPU depending on the preferred location attribute of the managed pointer.
  Note that the actual location of the memory range at the time
  of the query may be different from the preferred location.
* `cudaMemRangeAttributeAccessedBy`:
  will return the list of devices that have that advise set for that memory range.
* `cudaMemRangeAttributeLastPrefetchLocation`:
  will return the last location to which the memory range was prefetched
  explicitly using `cudaMemPrefetchAsync`.
  Note that this simply returns the last location that the application requested to prefetch the memory range to.
  It gives no indication as to whether the prefetch operation to that location has completed or even begun.
* `cudaMemRangeAttributePreferredLocationType`:
  :   will return the location type of the preferred location which will be `cudaMemLocationTypeDevice` if all pages
      in the memory range have the same GPU as their preferred location, or will be `cudaMemLocationTypeHost` if all
      pages in the memory range have the CPU as their preferred location, or it will be `cudaMemLocationTypeHostNuma`
      if all the pages in the memory range have the same host NUMA node ID as their preferred location or it will be
      `cudaMemLocationTypeInvalid` if either all the pages don’t have the same preferred location or some of the pages
      don’t have a preferred location at all.
* `cudaMemRangeAttributePreferredLocationId`:
  :   If the `cudaMemRangeAttributePreferredLocationType` query for the same address range returns `cudaMemLocationTypeDevice`,
      it will be a valid device ordinal or if it returns `cudaMemLocationTypeHostNuma`, it will be a valid host NUMA node ID
      or if it returns any other location type, the id should be ignored.
* `cudaMemRangeAttributeLastPrefetchLocationType`:
  :   will be the last location type to which all pages in the memory range were prefetched explicitly via `cudaMemPrefetchAsync`
      which will be `cudaMemLocationTypeDevice` if all pages in the memory range were prefetched to the same GPU,
      or will be `cudaMemLocationTypeHost` if all pages in the memory range were prefetched to the CPU or it will be
      `cudaMemLocationTypeHostNuma` if all the pages in the memory range were prefetched to the same host NUMA node ID
      or it will be `cudaMemLocationTypeInvalid` if either all the pages were not prefetched to the same location or some
      of the pages were never prefetched at all.
* `cudaMemRangeAttributeLastPrefetchLocationId`:
  :   If the `cudaMemRangeAttributeLastPrefetchLocationType` query for the same address range returns `cudaMemLocationTypeDevice`,
      it will be a valid device ordinal or if it returns `cudaMemLocationTypeHostNuma`, it will be a valid host NUMA node ID
      or if it returns any other location type, the id should be ignored.

Additionally, multiple attributes can be queried by using corresponding `cudaMemRangeGetAttributes` function.

## 24.2. Unified memory on devices with full CUDA Unified Memory support[](#unified-memory-on-devices-with-full-cuda-unified-memory-support "Permalink to this headline")

### 24.2.1. System-Allocated Memory: in-depth examples[](#system-allocated-memory-in-depth-examples "Permalink to this headline")

[Systems with full CUDA Unified Memory support](#um-requirements)
allow the device to access any memory owned by the host process interacting with the device.
This section shows a few advanced use-cases, using a kernel that simply prints
the first 8 characters of an input character array to the standard output stream:

```
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

The following tabs show various ways of how this kernel may be called:

Malloc

```
void test_malloc() {
  const char test_string[] = "Hello World";
  char* heap_data = (char*)malloc(sizeof(test_string));
  strncpy(heap_data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("malloc", heap_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  free(heap_data);
}
```


Managed

```
void test_managed() {
  const char test_string[] = "Hello World";
  char* data;
  cudaMallocManaged(&data, sizeof(test_string));
  strncpy(data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("managed", data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  cudaFree(data);
}
```


Stack variable

```
void test_stack() {
  const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("stack", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```


File-scope static variable

```
void test_static() {
  static const char test_string[] = "Hello World";
  kernel<<<1, 1>>>("static", test_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```


Global-scope variable

```
const char global_string[] = "Hello World";

void test_global() {
  kernel<<<1, 1>>>("global", global_string);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```


Global-scope extern variable

```
// declared in separate file, see below
extern char* ext_data;

void test_extern() {
  kernel<<<1, 1>>>("extern", ext_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
}
```

```
/** This may be a non-CUDA file */
char* ext_data;
static const char global_string[] = "Hello World";

void __attribute__ ((constructor)) setup(void) {
  ext_data = (char*)malloc(sizeof(global_string));
  strncpy(ext_data, global_string, sizeof(global_string));
}

void __attribute__ ((destructor)) tear_down(void) {
  free(ext_data);
}
```

The first three tabs above show the example as already detailed in the
[Programming Model section](#um-programming-model).
The next three tabs show various ways a file-scope or global-scope variable can
be accessed from the device.

Note that for the extern variable, it could be declared and its memory
owned and managed by a third-party library, which does not interact with CUDA at all.

Also note that stack variables as well as file-scope and global-scope variables can
only be accessed through a pointer by the GPU. In this specific example, this is
convenient because the character array is already declared as a pointer: `const char*`.
However, consider the following example with a global-scope integer:

```
// this variable is declared at global scope
int global_variable;

__global__ void kernel_uncompilable() {
  // this causes a compilation error: global (__host__) variables must not
  // be accessed from __device__ / __global__ code
  printf("%d\n", global_variable);
}

// On systems with pageableMemoryAccess set to 1, we can access the address
// of a global variable. The below kernel takes that address as an argument
__global__ void kernel(int* global_variable_addr) {
  printf("%d\n", *global_variable_addr);
}
int main() {
  kernel<<<1, 1>>>(&global_variable);
  ...
  return 0;
}
```

In the example above, we need to ensure to pass a *pointer* to the global variable
to the kernel instead of directly accessing the global variable in the kernel.
This is because global variables without the `__managed__` specifier are declared
as `__host__`-only by default, thus most compilers won’t allow using these
variables directly in device code as of now.

#### 24.2.1.1. File-backed Unified Memory[](#file-backed-unified-memory "Permalink to this headline")

Since [systems with full CUDA Unified Memory support](#um-requirements)
allow the device to access any memory owned by the host process,
they can directly access file-backed memory.

Here, we show a modified version of the initial example shown in the previous section to use
file-backed memory in order to print a string from the GPU, read directly from an input file.
In the following example, the memory is backed by a physical file, but the example
applies to memory-backed files, too, as detailed in the section on
[Inter-Process Communication (IPC) with Unified Memory](#um-sam-ipc).

```
__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}
```

```
void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "Invalid file handle");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "Invalid file stats");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
  kernel<<<1, 1>>>("file-backed", mapped);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
  ASSERT(close(fd) == 0, "Cannot close file");
}
```

Note that on systems without the `hostNativeAtomicSupported` property, including
[systems with Linux HMM enabled](#um-requirements),
atomic accesses to file-backed memory are not supported.

#### 24.2.1.2. Inter-Process Communication (IPC) with Unified Memory[](#inter-process-communication-ipc-with-unified-memory "Permalink to this headline")

Note

As of now, using IPC with Unified Memory can have significant performance implications.

Many applications prefer to manage one GPU per process, but still need to use Unified Memory,
for example for over-subscription, and access it from multiple GPUs.

CUDA IPC (see [Interprocess Communication](#interprocess-communication))
does not support Managed Memory: handles to this type of memory may not be shared through
any of the mechanisms discussed in this section.
On [systems with full CUDA Unified Memory support](#um-requirements),
System-Allocated Memory is Inter-Process Communication (IPC) capable.
Once access to System-Allocated Memory has been shared with other processes,
the same [Programming Model](#um-programming-model) applies,
similar to [File-backed Unified Memory](#um-sam-file-backed).

See the following references for more information on various ways of creating
IPC-capable System-Allocated Memory under Linux:

* [mmap with MAP\_SHARED](https://man7.org/linux/man-pages/man2/mmap.2.html)
* [POSIX IPC APIs](https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html)
* [Linux memfd\_create](https://man7.org/linux/man-pages/man2/memfd_create.2.html)

Note that it is not possible to share memory between different hosts and their devices using this technique.

### 24.2.2. Performance Tuning[](#performance-tuning "Permalink to this headline")

In order to achieve good performance with Unified Memory, it is important to:

* Understand how paging works on your system, and how to avoid unnecessary page faults.
* Understand the various mechanisms allowing you to keep data local to the accessing processor.
* Consider tuning your application for the granularity of memory transfers of your system.

As general advice, [Performance Hints](#um-perf-hints)
might provide improved performance, but using them incorrectly might degrade performance
compared to the default behavior.
Also note that any hint has a performance cost associated with it on the host,
thus useful hints must at the very least improve performance enough to overcome this cost.

#### 24.2.2.1. Memory Paging and Page Sizes[](#memory-paging-and-page-sizes "Permalink to this headline")

Many of the sections for unified memory performance tuning assume prior knowledge on virtual addressing,
memory pages and page sizes.
This section attempts to define all necessary terms and explain why paging matters for performance.

All currently supported systems for Unified Memory use a virtual address space:
this means that memory addresses used by an application represent a *virtual* location
which might be *mapped* to a physical location where the memory actually resides.

All currently supported processors, including both CPUs and GPUs, additionally use
memory *paging*. Because all systems use a virtual address space, there are two types
of memory pages:

* Virtual pages: this represents a fixed-size contiguous chunk of virtual memory
  per process tracked by the operating system, which can be *mapped* into physical memory.
  Note that the virtual page is linked to the *mapping*: for example, a single
  virtual address might be mapped into physical memory using different page sizes.
* Physical pages: this represents a fixed-size contiguous chunk of memory
  the processor’s main Memory Management Unit (MMU) supports and into which
  a virtual page can be mapped.

Currently, all x86\_64 CPUs use 4KiB physical pages.
Arm CPUs support multiple physical page sizes - 4KiB, 16KiB, 32KiB and 64KiB - depending on the exact CPU.
Finally, NVIDIA GPUs support multiple physical page sizes, but prefer 2MiB physical pages or larger.
Note that these sizes are subject to change in future hardware.

The default page size of virtual pages usually corresponds to the physical page size,
but an application may use different page sizes as long as they are supported by the
operating system and the hardware. Typically, supported virtual page sizes must be
powers of 2 and multiples of the physical page size.

The logical entity tracking the mapping of virtual pages into physical pages will be referred to as a *page table*,
and each mapping of a given virtual page with a given virtual size to physical pages is called a *page table entry (PTE)*.
All supported processors provide specific caches for the page table to speed up the translation of
virtual addresses to physical addresses. These caches are called *translation lookaside buffers (TLBs)*.

There are two important aspects for performance tuning of applications:

* the choice of virtual page size,
* whether the system offers a combined page table used by both CPUs and GPUs,
  or separate page tables for each CPU and GPU individually.

##### 24.2.2.1.1. Choosing the right page size[](#choosing-the-right-page-size "Permalink to this headline")

In general, small page sizes lead to less (virtual) memory fragmentation but more TLB misses,
whereas larger page sizes lead to more memory fragmentation but less TLB misses.
Additionally, memory migration is generally more expensive with larger page sizes compared to
smaller page sizes, because we typically migrate full memory pages. This can cause
larger latency spikes in an application using large page sizes. See also the next section
for more details on page faults.

One important aspect for performance tuning is that TLB misses are generally
significantly more expensive on the GPU compared to the CPU. This means that
if a GPU thread frequently accesses random locations of Unified Memory mapped
using a small enough page size, it might be significantly slower compared to
the same accesses to Unified Memory mapped using a large enough page size.
While a similar effect might occur for a CPU thread randomly accessing a large
area of memory mapped using a small page size, the slowdown is less pronounced,
meaning that the application might want to trade-off this slowdown with
having less memory fragmentation.

Note that in general, applications should not tune their performance to the
physical page size of a given processor, since physical page sizes are subject
to change depending on the hardware. The advice above only applies to virtual
page sizes.

##### 24.2.2.1.2. CPU and GPU page tables: hardware coherency vs. software coherency[](#cpu-and-gpu-page-tables-hardware-coherency-vs-software-coherency "Permalink to this headline")

Note

In the remainder of the performance tuning documentation, we will refer
to systems with a combined page table for both CPUs and GPUs as *hardware
coherent* systems. Systems with separate page tables for CPUs and GPUs are
referred to as *software coherent*.

Hardware coherent systems such as NVIDIA Grace Hopper offer a logically combined
page table for both CPUs and GPUs.
This is important because in order to access
[System-Allocated Memory from the GPU](#um-system-allocator),
the GPU uses whichever page table entry was created by the CPU for the requested memory.
If that page table entry uses the default CPU page size of 4KiB or 64KiB,
accesses to large virtual memory areas will cause significant TLB misses,
thus significant slowdowns.

See the section on configuring huge pages for examples on how to ensure
System-Allocated Memory uses large enough page sizes to avoid this type of issue.

On the other hand, on systems where the CPUs and GPUs each have their own logical
page table, different performance tuning aspects should be considered:
in order to [guarantee coherency](#um-introduction), these systems
usually use *page faults* in case a processor accesses a memory address mapped
into the physical memory of a different processor. Such a page fault means that:

* it needs to be ensured that the currently owning processor (where the physical page currently resides)
  cannot access this page anymore, either by deleting the page table entry or updating it.
* it needs to be ensured that the processor requesting access can access this page,
  either by creating a new page table entry or updating and existing entry, such that
  it becomes valid/active.
* the physical page backing this virtual page must be moved/migrated to the processor
  requesting access: this can be an expensive operation, and the amount of work
  is proportional to the page size.

Overall, hardware coherent systems provide significant performance benefits
compared to software coherent systems in cases where frequent concurrent accesses
to the same memory page are made by both CPU and GPU threads:

* less page-faults: these systems do not need to use page-faults for emulating coherency or migrating memory,
* less contention: these systems are coherent at cache-line granularity instead of page-size granularity, that is,
  when there is contention from multiple processors within a cache line, only the cache line is exchanged which is much smaller than the smallest page-size,
  and when the different processors access different cache-lines within a page, then there is no contention.

This impacts the performance of the following scenarios:

* Atomic updates to the same address concurrently from both CPUs and GPUs.
* Signaling a GPU thread from a CPU thread or vice-versa.

#### 24.2.2.2. Direct Unified Memory Access from host[](#direct-unified-memory-access-from-host "Permalink to this headline")

Some devices have hardware support for coherent reads, stores and atomic accesses
from the host on GPU-resident unified memory.
These devices have the attribute `cudaDevAttrDirectManagedMemAccessFromHost` set to 1.
Note that all [hardware coherent systems](#um-hw-coherency) have
this attribute set for NVLink-connected devices.
On these systems, the host has direct access to GPU-resident memory without page faults and
data migration (see [Data Usage Hints](#um-tuning-usage)
for more details on memory usage hints). Note that with CUDA Managed Memory,
the `cudaMemAdviseSetAccessedBy` hint with location type `cudaMemLocationTypeHost` is necessary
to enable this direct access without page faults.

Consider an example code below:

System Allocator

```
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}
void test_malloc() {
  int *ret = (int*)malloc(1000 * sizeof(int));
  // for shared page table systems, the following hint is not necessary
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  free(ret);
}
```


Managed

```
__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}

void test_managed() {
  int *ret;
  cudaMallocManaged(&ret, 1000 * sizeof(int));
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);  // set direct access hint

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  cudaFree(ret); 
}
```

After `write` kernel is completed, `ret` will be created and initialized in GPU memory.
Next, the CPU will access `ret` followed by `append` kernel using the same `ret` memory again.
This code will show different behavior depending on the system architecture and support of hardware coherency:

* On systems with `directManagedMemAccessFromHost=1`:
  CPU accesses to the managed buffer will not trigger any migrations;
  the data will remain resident in GPU memory and any subsequent GPU kernels
  can continue to access it directly without inflicting faults or migrations.
* On systems with `directManagedMemAccessFromHost=0`:
  CPU accesses to the managed buffer will page fault and initiate data migration;
  any GPU kernel trying to access the same data first time will page fault and
  migrate pages back to GPU memory.

#### 24.2.2.3. Host Native Atomics[](#host-native-atomics "Permalink to this headline")

Some devices, including NVLink-connected devices in
[hardware coherent systems](#um-hw-coherency), support hardware-accelerated
atomic accesses to CPU-resident memory. This implies that atomic accesses to host memory
do not have to be emulated with a page fault.
For these devices, the attribute `cudaDevAttrHostNativeAtomicSupported` is set to 1.

#### 24.2.2.4. Atomic accesses & synchronization primitives[](#atomic-accesses-synchronization-primitives "Permalink to this headline")

CUDA Unified Memory supports all atomic operations available to host and device threads,
enabling all threads to cooperate by concurrently accessing the same shared memory location.
The [CUDA C++ standard library](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html)
provides many heterogeneous synchronization primitives tuned for concurrent use between host and device threads,
including `cuda::atomic`, `cuda::atomic_ref`, `cuda::barrier`, `cuda::semaphore`, among many others.

On systems without [CPU and GPU page tables: hardware coherency vs. software coherency](#um-hw-coherency),
atomic accesses from the device to file-backed host memory are not supported.
The following example code is valid on systems with [CPU and GPU page tables: hardware coherency vs. software coherency](#um-hw-coherency)
but exhibits undefined behavior on other systems:

```
#include <cuda/atomic>

#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>

#define ERR(msg, ...) { fprintf(stderr, msg, ##__VA_ARGS__); return EXIT_FAILURE; }

__global__ void kernel(int* ptr) {
  cuda::atomic_ref{*ptr}.store(2);
}

int main() {
  // this will be closed/deleted by default on exit
  FILE* tmp_file = tmpfile64();
  // need to allocate space in the file, we do this with posix_fallocate here
  int status = posix_fallocate(fileno(tmp_file), 0, 4096);
  if (status != 0) ERR("Failed to allocate space in temp file\n");
  int* ptr = (int*)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(tmp_file), 0);
  if (ptr == MAP_FAILED) ERR("Failed to map temp file\n");

  // initialize the value in our file-backed memory
  *ptr = 1;
  printf("Atom value: %d\n", *ptr);

  // device and host thread access ptr concurrently, using cuda::atomic_ref
  kernel<<<1, 1>>>(ptr);
  while (cuda::atomic_ref{*ptr}.load() != 2);
  // this will always be 2
  printf("Atom value: %d\n", *ptr);

  return EXIT_SUCCESS;
}
```

On systems without [CPU and GPU page tables: hardware coherency vs. software coherency](#um-hw-coherency),
atomic accesses to unified memory may incur page faults which can lead to significant latencies.
Note that this is not the case for all GPU atomics to CPU memory on these systems:
operations listed by `nvidia-smi -q | grep "Atomic Caps Outbound"` may avoid page faults.

On systems with [CPU and GPU page tables: hardware coherency vs. software coherency](#um-hw-coherency),
atomics between host and device do not require page faults,
but may still fault for other reasons that any memory access can fault for.

#### 24.2.2.5. Memcpy()/Memset() Behavior With Unified Memory[](#memcpy-memset-behavior-with-unified-memory "Permalink to this headline")

`cudaMemcpy*()` and `cudaMemset*()` accept any unified memory pointer as arguments.

For `cudaMemcpy*()`, the direction specified as `cudaMemcpyKind` is a performance hint,
which can have a higher performance impact if any of the arguments is a unified memory pointer.

Thus, it is recommended to follow the following performance advice:

* When the physical location of unified memory is known, use an accurate `cudaMemcpyKind` hint.
* Prefer `cudaMemcpyDefault` over an inaccurate `cudaMemcpyKind` hint.
* Always use populated (initialized) buffers: avoid using these APIs to initialize memory.
* Avoid using `cudaMemcpy*()` if both pointers point to System-Allocated Memory:
  launch a kernel or use a CPU memory copy algorithm such as `std::memcpy` instead.

## 24.3. Unified memory on devices without full CUDA Unified Memory support[](#unified-memory-on-devices-without-full-cuda-unified-memory-support "Permalink to this headline")

### 24.3.1. Unified memory on devices with only CUDA Managed Memory support[](#unified-memory-on-devices-with-only-cuda-managed-memory-support "Permalink to this headline")

For devices with compute capability 6.x or higher but without [pageable memory access](#um-requirements),
CUDA Managed Memory is fully supported and coherent.
The programming model and performance tuning of unified memory is largely similar
to the model as described in
[Unified memory on devices with full CUDA Unified Memory support](#um-pageable-systems),
with the notable exception that system allocators cannot be used to allocate memory.
Thus, the following list of sub-sections do not apply:

* [System-Allocated Memory: in-depth examples](#um-system-allocator)
* [Hardware/Software Coherency](#um-hw-coherency)

### 24.3.2. Unified memory on Windows or devices with compute capability 5.x[](#unified-memory-on-windows-or-devices-with-compute-capability-5-x "Permalink to this headline")

Devices with compute capability lower than 6.0 or Windows platforms support CUDA Managed Memory v1.0 with limited support for data migration and coherency as well as memory oversubscription. The following sub-sections describe in more detail how to use and optimize Managed Memory on these platforms.

#### 24.3.2.1. Data Migration and Coherency[](#data-migration-and-coherency "Permalink to this headline")

GPU architectures of compute capability lower than 6.0 do not support fine-grained movement of the managed data to GPU on-demand. Whenever a GPU kernel is launched all managed memory generally has to be transferred to GPU memory to avoid faulting on memory access. With compute capability 6.x a new GPU page faulting mechanism is introduced that provides more seamless Unified Memory functionality. Combined with the system-wide virtual address space, page faulting provides several benefits. First, page faulting means that the CUDA system software doesn’t need to synchronize all managed memory allocations to the GPU before each kernel launch. If a kernel running on the GPU accesses a page that is not resident in its memory, it faults, allowing the page to be automatically migrated to the GPU memory on-demand. Alternatively, the page may be mapped into the GPU address space for access over the PCIe or NVLink interconnects (mapping on access can sometimes be faster than migration). Note that Unified Memory is system-wide: GPUs (and CPUs) can fault on and migrate memory pages either from CPU memory or from the memory of other GPUs in the system.

#### 24.3.2.2. GPU Memory Oversubscription[](#um-legacy-oversubscription "Permalink to this headline")

Devices of compute capability lower than 6.0 cannot allocate more managed memory than the physical size of GPU memory.

#### 24.3.2.3. Multi-GPU[](#multi-gpu "Permalink to this headline")

On systems with devices of compute capabilities lower than 6.0 managed allocations are automatically visible to all GPUs in a system via the peer-to-peer capabilities of the GPUs. Managed memory allocations behave similar to unmanaged memory allocated using `cudaMalloc()`: the current active device is the home for the physical allocation but other GPUs in the system will access the memory at reduced bandwidth over the PCIe bus.

On Linux the managed memory is allocated in GPU memory as long as all GPUs that are actively being used by a program have the peer-to-peer support. If at any time the application starts using a GPU that doesn’t have peer-to-peer support with any of the other GPUs that have managed allocations on them, then the driver will migrate all managed allocations to system memory. In this case, all GPUs experience PCIe bandwidth restrictions.

On Windows, if peer mappings are not available (for example, between GPUs of different architectures), then the system will automatically fall back to using zero-copy memory, regardless of whether both GPUs are actually used by a program. If only one GPU is actually going to be used, it is necessary to set the `CUDA_VISIBLE_DEVICES` environment variable before launching the program. This constrains which GPUs are visible and allows managed memory to be allocated in GPU memory.

Alternatively, on Windows users can also set `CUDA_MANAGED_FORCE_DEVICE_ALLOC` to a non-zero value to force the driver to always use device memory for physical storage. When this environment variable is set to a non-zero value, all devices used in that process that support managed memory have to be peer-to-peer compatible with each other. The error `::cudaErrorInvalidDevice` will be returned if a device that supports managed memory is used and it is not peer-to-peer compatible with any of the other managed memory supporting devices that were previously used in that process, even if `::cudaDeviceReset` has been called on those devices. These environment variables are described in [CUDA Environment Variables](#env-vars). Note that starting from CUDA 8.0 `CUDA_MANAGED_FORCE_DEVICE_ALLOC` has no effect on Linux operating systems.

#### 24.3.2.4. Coherency and Concurrency[](#coherency-and-concurrency "Permalink to this headline")

Simultaneous access to managed memory on devices of compute capability lower than 6.0 is not possible, because coherence could not be guaranteed if the CPU accessed a Unified Memory allocation while a GPU kernel was active.

##### 24.3.2.4.1. GPU Exclusive Access To Managed Memory[](#gpu-exclusive-access-to-managed-memory "Permalink to this headline")

To ensure coherency on pre-6.x GPU architectures, the Unified Memory programming model puts constraints on data accesses while both the CPU and GPU are executing concurrently. In effect, the GPU has exclusive access to all managed data while any kernel operation is executing, regardless of whether the specific kernel is actively using the data. When managed data is used with `cudaMemcpy*()` or `cudaMemset*()`, the system may choose to access the source or destination from the host or the device, which will put constraints on concurrent CPU access to that data while the `cudaMemcpy*()` or `cudaMemset*()` is executing. See [Memcpy()/Memset() Behavior With Unified Memory](#um-memcpy-memset) for further details.

It is not permitted for the CPU to access any managed allocations or variables while the GPU is active for devices with `concurrentManagedAccess` property set to 0. On these systems concurrent CPU/GPU accesses, even to different managed memory allocations, will cause a segmentation fault because the page is considered inaccessible to the CPU.

```
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    y = 20;            // Error on GPUs not supporting concurrent access

    cudaDeviceSynchronize();
    return  0;
}
```

In example above, the GPU program `kernel` is still active when the CPU touches `y`. (Note how it occurs before `cudaDeviceSynchronize()`.) The code runs successfully on devices of compute capability 6.x due to the GPU page faulting capability which lifts all restrictions on simultaneous access. However, such memory access is invalid on pre-6.x architectures even though the CPU is accessing different data than the GPU. The program must explicitly synchronize with the GPU before accessing `y`:

```
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    cudaDeviceSynchronize();
    y = 20;            //  Success on GPUs not supporing concurrent access
    return  0;
}
```

As this example shows, on systems with pre-6.x GPU architectures, a CPU thread may not access any managed data in between performing a kernel launch and a subsequent synchronization call, regardless of whether the GPU kernel actually touches that same data (or any managed data at all). The mere potential for concurrent CPU and GPU access is sufficient for a process-level exception to be raised.

Note that if memory is dynamically allocated with `cudaMallocManaged()` or `cuMemAllocManaged()` while the GPU is active, the behavior of the memory is unspecified until additional work is launched or the GPU is synchronized. Attempting to access the memory on the CPU during this time may or may not cause a segmentation fault. This does not apply to memory allocated using the flag `cudaMemAttachHost` or `CU_MEM_ATTACH_HOST`.

##### 24.3.2.4.2. Explicit Synchronization and Logical GPU Activity[](#explicit-synchronization-and-logical-gpu-activity "Permalink to this headline")

Note that explicit synchronization is required even if `kernel` runs quickly and finishes before the CPU touches `y` in the above example. Unified Memory uses logical activity to determine whether the GPU is idle. This aligns with the CUDA programming model, which specifies that a kernel can run at any time following a launch and is not guaranteed to have finished until the host issues a synchronization call.

Any function call that logically guarantees the GPU completes its work is valid. This includes `cudaDeviceSynchronize()`; `cudaStreamSynchronize()` and `cudaStreamQuery()` (provided it returns `cudaSuccess` and not `cudaErrorNotReady`) where the specified stream is the only stream still executing on the GPU; `cudaEventSynchronize()` and `cudaEventQuery()` in cases where the specified event is not followed by any device work; as well as uses of `cudaMemcpy()` and `cudaMemset()` that are documented as being fully synchronous with respect to the host.

Dependencies created between streams will be followed to infer completion of other streams by synchronizing on a stream or event. Dependencies can be created via `cudaStreamWaitEvent()` or implicitly when using the default (NULL) stream.

It is legal for the CPU to access managed data from within a stream callback, provided no other stream that could potentially be accessing managed data is active on the GPU. In addition, a callback that is not followed by any device work can be used for synchronization: for example, by signaling a condition variable from inside the callback; otherwise, CPU access is valid only for the duration of the callback(s).

There are several important points of note:

* It is always permitted for the CPU to access non-managed zero-copy data while the GPU is active.
* The GPU is considered active when it is running any kernel, even if that kernel does not make use of managed data. If a kernel might use data, then access is forbidden, unless device property `concurrentManagedAccess` is 1.
* There are no constraints on concurrent inter-GPU access of managed memory, other than those that apply to multi-GPU access of non-managed memory.
* There are no constraints on concurrent GPU kernels accessing managed data.

Note how the last point allows for races between GPU kernels, as is currently the case for non-managed GPU memory. As mentioned previously, managed memory functions identically to non-managed memory from the perspective of the GPU. The following code example illustrates these points:

```
int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    int *non_managed, *managed, *also_managed;
    cudaMallocHost(&non_managed, 4);    // Non-managed, CPU-accessible memory
    cudaMallocManaged(&managed, 4);
    cudaMallocManaged(&also_managed, 4);
    // Point 1: CPU can access non-managed data.
    kernel<<< 1, 1, 0, stream1 >>>(managed);
    *non_managed = 1;
    // Point 2: CPU cannot access any managed data while GPU is busy,
    //          unless concurrentManagedAccess = 1
    // Note we have not yet synchronized, so "kernel" is still active.
    *also_managed = 2;      // Will issue segmentation fault
    // Point 3: Concurrent GPU kernels can access the same data.
    kernel<<< 1, 1, 0, stream2 >>>(managed);
    // Point 4: Multi-GPU concurrent access is also permitted.
    cudaSetDevice(1);
    kernel<<< 1, 1 >>>(managed);
    return  0;
}
```

##### 24.3.2.4.3. Managing Data Visibility and Concurrent CPU + GPU Access with Streams[](#managing-data-visibility-and-concurrent-cpu-gpu-access-with-streams "Permalink to this headline")

Until now it was assumed that for SM architectures before 6.x: 1) any active kernel may use any managed memory, and 2) it was invalid to use managed memory from the CPU while a kernel is active. Here we present a system for finer-grained control of managed memory designed to work on all devices supporting managed memory, including older architectures with `concurrentManagedAccess` equal to 0.

The CUDA programming model provides streams as a mechanism for programs to indicate dependence and independence among kernel launches. Kernels launched into the same stream are guaranteed to execute consecutively, while kernels launched into different streams are permitted to execute concurrently. Streams describe independence between work items and hence allow potentially greater efficiency through concurrency.

Unified Memory builds upon the stream-independence model by allowing a CUDA program to explicitly associate managed allocations with a CUDA stream. In this way, the programmer indicates the use of data by kernels based on whether they are launched into a specified stream or not. This enables opportunities for concurrency based on program-specific data access patterns. The function to control this behavior is:

```
cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream,
                                     void *ptr,
                                     size_t length=0,
                                     unsigned int flags=0);
```

The `cudaStreamAttachMemAsync()` function associates `length` bytes of memory starting from `ptr` with the specified `stream`. (Currently, `length` must always be 0 to indicate that the entire region should be attached.) Because of this association, the Unified Memory system allows CPU access to this memory region so long as all operations in `stream` have completed, regardless of whether other streams are active. In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity.

Most importantly, if an allocation is not associated with a specific stream, it is visible to all running kernels regardless of their stream. This is the default visibility for a `cudaMallocManaged()` allocation or a `__managed__` variable; hence, the simple-case rule that the CPU may not touch the data while any kernel is running.

By associating an allocation with a specific stream, the program makes a guarantee that only kernels launched into that stream will touch that data. No error checking is performed by the Unified Memory system: it is the programmer’s responsibility to ensure that guarantee is honored.

In addition to allowing greater concurrency, the use of `cudaStreamAttachMemAsync()` can (and typically does) enable data transfer optimizations within the Unified Memory system that may affect latencies and other overhead.

##### 24.3.2.4.4. Stream Association Examples[](#stream-association-examples "Permalink to this headline")

Associating data with a stream allows fine-grained control over CPU + GPU concurrency, but what data is visible to which streams must be kept in mind when using devices of compute capability lower than 6.0. Looking at the earlier synchronization example:

```
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
    cudaDeviceSynchronize();          // Wait for Host attachment to occur.
    kernel<<< 1, 1, 0, stream1 >>>(); // Note: Launches into stream1.
    y = 20;                           // Success – a kernel is running but “y”
                                      // has been associated with no stream.
    return  0;
}
```

Here we explicitly associate `y` with host accessibility, thus enabling access at all times from the CPU. (As before, note the absence of `cudaDeviceSynchronize()` before the access.) Accesses to `y` by the GPU running `kernel` will now produce undefined results.

Note that associating a variable with a stream does not change the associating of any other variable. For example, associating `x` with `stream1` does not ensure that only `x` is accessed by kernels launched in `stream1`, thus an error is caused by this code:

```
__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &x);// Associate “x” with stream1.
    cudaDeviceSynchronize();              // Wait for “x” attachment to occur.
    kernel<<< 1, 1, 0, stream1 >>>();     // Note: Launches into stream1.
    y = 20;                               // ERROR: “y” is still associated globally
                                          // with all streams by default
    return  0;
}
```

Note how the access to `y` will cause an error because, even though `x` has been associated with a stream, we have told the system nothing about who can see `y`. The system therefore conservatively assumes that `kernel` might access it and prevents the CPU from doing so.

##### 24.3.2.4.5. Stream Attach With Multithreaded Host Programs[](#stream-attach-with-multithreaded-host-programs "Permalink to this headline")

The primary use for `cudaStreamAttachMemAsync()` is to enable independent task parallelism using CPU threads. Typically in such a program, a CPU thread creates its own stream for all work that it generates because using CUDA’s NULL stream would cause dependencies between threads.

The default global visibility of managed data to any GPU stream can make it difficult to avoid interactions between CPU threads in a multi-threaded program. Function `cudaStreamAttachMemAsync()` is therefore used to associate a thread’s managed allocations with that thread’s own stream, and the association is typically not changed for the life of the thread.

Such a program would simply add a single call to `cudaStreamAttachMemAsync()` to use unified memory for its data accesses:

```
// This function performs some task, in its own private stream.
void run_task(int *in, int *out, int length) {
    // Create a stream for us to use.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Allocate some managed data and associate with our stream.
    // Note the use of the host-attach flag to cudaMallocManaged();
    // we then associate the allocation with our stream so that
    // our GPU kernel launches can access it.
    int *data;
    cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, data);
    cudaStreamSynchronize(stream);
    // Iterate on the data in some way, using both Host & Device.
    for(int i=0; i<N; i++) {
        transform<<< 100, 256, 0, stream >>>(in, data, length);
        cudaStreamSynchronize(stream);
        host_process(data, length);    // CPU uses managed data.
        convert<<< 100, 256, 0, stream >>>(out, data, length);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(data);
}
```

In this example, the allocation-stream association is established just once, and then `data` is used repeatedly by both the host and device. The result is much simpler code than occurs with explicitly copying data between host and device, although the result is the same.

##### 24.3.2.4.6. Advanced Topic: Modular Programs and Data Access Constraints[](#advanced-topic-modular-programs-and-data-access-constraints "Permalink to this headline")

In the previous example `cudaMallocManaged()` specifies the `cudaMemAttachHost` flag, which creates an allocation that is initially invisible to device-side execution. (The default allocation would be visible to all GPU kernels on all streams.) This ensures that there is no accidental interaction with another thread’s execution in the interval between the data allocation and when the data is acquired for a specific stream.

Without this flag, a new allocation would be considered in-use on the GPU if a kernel launched by another thread happens to be running. This might impact the thread’s ability to access the newly allocated data from the CPU (for example, within a base-class constructor) before it is able to explicitly attach it to a private stream. To enable safe independence between threads, therefore, allocations should be made specifying this flag.

Note

An alternative would be to place a process-wide barrier across all threads after the allocation has been attached to the stream. This would ensure that all threads complete their data/stream associations before any kernels are launched, avoiding the hazard. A second barrier would be needed before the stream is destroyed because stream destruction causes allocations to revert to their default visibility. The `cudaMemAttachHost` flag exists both to simplify this process, and because it is not always possible to insert global barriers where required.

##### 24.3.2.4.7. Memcpy()/Memset() Behavior With Stream-associated Unified Memory[](#memcpy-memset-behavior-with-stream-associated-unified-memory "Permalink to this headline")

See [Memcpy()/Memset() Behavior With Unified Memory](#um-memcpy-memset) for a general overview of `cudaMemcpy*` / `cudaMemset*` behavior on devices with `concurrentManagedAccess` set. On devices where `concurrentManagedAccess` is not set, the following rules apply:

If `cudaMemcpyHostTo*` is specified and the source data is unified memory, then it will be accessed from the host if it is coherently accessible from the host in the copy stream [(1)](#um-legacy-memcpy-cit1); otherwise it will be accessed from the device. Similar rules apply to the destination when `cudaMemcpy*ToHost` is specified and the destination is unified memory.

If `cudaMemcpyDeviceTo*` is specified and the source data is unified memory, then it will be accessed from the device. The source must be coherently accessible from the device in the copy stream [(2)](#um-legacy-memcpy-cit2); otherwise, an error is returned. Similar rules apply to the destination when `cudaMemcpy*ToDevice` is specified and the destination is unified memory.

If `cudaMemcpyDefault` is specified, then unified memory will be accessed from the host either if it cannot be coherently accessed from the device in the copy stream [(2)](#um-legacy-memcpy-cit2) or if the preferred location for the data is `cudaCpuDeviceId` and it can be coherently accessed from the host in the copy stream [(1)](#um-legacy-memcpy-cit1); otherwise, it will be accessed from the device.

When using `cudaMemset*()` with unified memory, the data must be coherently accessible from the device in the stream being used for the `cudaMemset()` operation [(2)](#um-legacy-memcpy-cit2); otherwise, an error is returned.

When data is accessed from the device either by `cudaMemcpy*` or `cudaMemset*`, the stream of operation is considered to be active on the GPU. During this time, any CPU access of data that is associated with that stream or data that has global visibility, will result in a segmentation fault if the GPU has a zero value for the device attribute `concurrentManagedAccess`. The program must synchronize appropriately to ensure the operation has completed before accessing any associated data from the CPU.

> 1. Coherently accessible from the host in a given stream means that the memory neither has global visibility nor is it associated with the given stream.

> 2. Coherently accessible from the device in a given stream means that the memory either has global visibility or is associated with the given stream.
