# GPU Hardware Specifications

Peak theoretical numbers for roofline analysis and utilization calculations.

## NVIDIA Data Center GPUs

| GPU | Arch | CC | SMs | FP64 TFLOPS | FP32 TFLOPS | FP16 TFLOPS | BF16 TFLOPS | TF32 TFLOPS | FP8 TFLOPS | FP4 TFLOPS | INT8 TOPS | HBM BW (GB/s) | HBM Size | L2 Cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V100-SXM2 | Volta | 7.0 | 80 | 7.8 | 15.7 | 125 | — | — | — | — | 62.5 | 900 | 16/32 GB | 6 MB |
| V100-PCIe | Volta | 7.0 | 80 | 7.0 | 14.0 | 112 | — | — | — | — | 56 | 900 | 16/32 GB | 6 MB |
| A100-SXM | Ampere | 8.0 | 108 | 9.7 | 19.5 | 312 | 312 | 156 | — | — | 624 | 2039 | 40/80 GB | 40 MB |
| A100-PCIe-40GB | Ampere | 8.0 | 108 | 9.7 | 19.5 | 312 | 312 | 156 | — | — | 624 | 1555 | 40 GB | 40 MB |
| A100-PCIe-80GB | Ampere | 8.0 | 108 | 9.7 | 19.5 | 312 | 312 | 156 | — | — | 624 | 2039 | 80 GB | 40 MB |
| H100-SXM | Hopper | 9.0 | 132 | 33.5 | 66.9 | 989 | 989 | 495 | 1979 | — | 1979 | 3352 | 80 GB | 50 MB |
| H100-PCIe | Hopper | 9.0 | 114 | 25.6 | 51.2 | 756 | 756 | 378 | 1513 | — | 1513 | 2039 | 80 GB | 50 MB |
| H200 | Hopper | 9.0 | 132 | 33.5 | 66.9 | 989 | 989 | 495 | 1979 | — | 1979 | 4800 | 141 GB | 50 MB |
| **B200 (HGX)** | **Blackwell** | **10.0** | **148** | **37** | **75** | **2250** | **2250** | **1125** | **4500** | **9000** | **4500** | **7700** | **180 GB** | **96 MB** |
| **B200 (NVL72)** | **Blackwell** | **10.0** | **148** | **40** | **80** | **2500** | **2500** | **1250** | **5000** | **10000** | **5000** | **8000** | **186 GB** | **96 MB** |

## Consumer GPUs (commonly used for development)

| GPU | Arch | CC | SMs | FP32 TFLOPS | FP16 TFLOPS | FP8 TFLOPS | BW (GB/s) | VRAM | L2 Cache |
|---|---|---|---|---|---|---|---|---|---|
| RTX 3090 | Ampere | 8.6 | 82 | 35.6 | 71 | — | 936 | 24 GB GDDR6X | 6 MB |
| RTX 4090 | Ada | 8.9 | 128 | 82.6 | 330 | 661 | 1008 | 24 GB GDDR6X | 72 MB |
| L40S | Ada | 8.9 | 142 | 91.6 | 362 | 733 | 864 | 48 GB GDDR6 | 96 MB |
| **RTX 5090** | **Blackwell** | **12.0** | **170** | **105** | **419** | **838** | **1792** | **32 GB GDDR7** | **96 MB** |
| **RTX 5080** | **Blackwell** | **12.0** | **84** | **52** | **207** | **414** | **960** | **16 GB GDDR7** | **64 MB** |

## Notes

- **FP16/BF16 TFLOPS** include tensor core throughput (with sparsity: multiply by 2)
- **FP8 TFLOPS** only available on Hopper (SM90) and later
- **FP4 TFLOPS** only available on Blackwell (SM100) and later. FP4 dense = 2x FP8 dense (doubles both compute and memory efficiency).
- **B200 has two SKUs**: HGX B200 (standalone, 1000W TDP, 180GB, 7.7 TB/s) and GB200 NVL72 variant (higher clocks, 1200W, 186GB, 8 TB/s). Numbers differ ~10-15%.
- **NVIDIA datasheet footnote**: "All Tensor Core numbers except FP64 with sparsity." The table above shows **dense** numbers (datasheet values ÷ 2). Source: NVIDIA Blackwell Datasheet 3384703, DEC24.
- **TF32** = TensorFloat-32, a 19-bit format for tensor cores on Ampere+
- All TFLOPS numbers are **dense** (non-sparse). Structured sparsity doubles throughput on Ampere+.
- HBM bandwidth is **theoretical peak**. Achievable is typically 80-90% of peak.
- CC 10.0 (B200) and CC 12.0 (RTX 5090) are different SM designs despite both being "Blackwell".

## Shared Memory Limits Per SM

| Architecture | Max Shared Memory Per SM | Max Shared Memory Per Block (default) | Max Shared Memory Per Block (opt-in) |
|---|---|---|---|
| Volta (SM70) | 96 KB | 48 KB | 96 KB |
| Ampere (SM80) | 164 KB | 48 KB | 163 KB |
| Ada (SM89) | 100 KB | 48 KB | 100 KB |
| Hopper (SM90) | 228 KB | 48 KB | 228 KB |
| Blackwell DC (SM100) | 228 KB | 48 KB | 227 KB |
| Blackwell Consumer (SM120) | 100 KB | 48 KB | 99 KB |

To use >48KB shared memory per block, call `cudaFuncSetAttribute()` with `cudaFuncAttributeMaxDynamicSharedMemorySize`.

## Register Limits

| Architecture | Registers Per Thread (max) | Register File Per SM |
|---|---|---|
| All (SM70+) | 255 | 64K × 32-bit registers |

Occupancy is limited when `registers_per_thread × threads_per_block > register_file_size / max_blocks_per_sm`.

## Max Threads / Blocks Per SM

| Architecture | Max Threads Per SM | Max Blocks Per SM | Max Warps Per SM |
|---|---|---|---|
| Volta (SM70) | 2048 | 32 | 64 |
| Ampere (SM80) | 2048 | 32 | 64 |
| Ada (SM89) | 1536 | 24 | 48 |
| Hopper (SM90) | 2048 | 32 | 64 |
| Blackwell DC (SM100) | 2048 | 32 | 64 |
| Blackwell Consumer (SM120) | 1536 | 24 | 48 |

## Useful formulas

```
Theoretical Occupancy = active_warps_per_sm / max_warps_per_sm

Max Warps From Registers = floor(register_file_per_sm / (registers_per_thread × 32))
                           (32 threads per warp, each gets registers_per_thread registers)

Max Warps From Shared Memory = floor(shared_mem_per_sm / shared_mem_per_block) × (block_size / 32)

Peak Bandwidth (bytes/s) = 2 × memory_bus_width × memory_clock_rate
                           (factor of 2 for DDR)

Achieved Bandwidth (GB/s) = total_dram_bytes / kernel_duration_s / 1e9

Achieved TFLOPS = total_flops / kernel_duration_s / 1e12

Arithmetic Intensity = total_flops / total_dram_bytes
```
