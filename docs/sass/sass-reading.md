# SASS (Streaming ASSembler) Reading Guide

## What is SASS

SASS is the machine-level assembly for NVIDIA GPUs. Architecture-specific (not portable across major SM versions). Every instruction is 128 bits (16 bytes) on Volta+.

Pipeline: **CUDA C++ → PTX (virtual ISA) → SASS (real machine code)**

## Generating SASS

```bash
# Compile to cubin, then dump SASS
nvcc --cubin -arch=sm_90 -O3 -lineinfo kernel.cu
cuobjdump --dump-sass kernel.cubin

# Or directly from executable
cuobjdump --dump-sass ./my_app

# With source line correlation
nvdisasm -g kernel.cubin

# Resource usage (registers, shared memory)
cuobjdump -res-usage kernel.cubin
nvcc -arch=sm_90 -O3 -Xptxas -v kernel.cu 2>&1 | grep "ptxas info"

# Control flow graph (DOT format)
nvdisasm -cfg kernel.cubin > cfg.dot

# Register liveness
nvdisasm -plr kernel.cubin
```

### Key cuobjdump flags
`--dump-sass`, `--dump-ptx`, `--dump-resource-usage`, `--gpu-architecture <arch>`, `--sort-functions`

### Key nvdisasm flags
`-c` (code only), `-g` (source lines), `-gi` (with inlining), `-hex` (show encoding), `-cfg` (control flow graph), `-plr` (register liveness), `-json` (JSON output)

## Instruction Format

```
/*0080*/    @P0 FFMA.FTZ R4, |R2|.reuse, -RZ, R5 ;   /* 0x800000ff02047223 */
                                                        /* 0x001fc800000e0400 */
```

| Part | Meaning |
|---|---|
| `/*0080*/` | Instruction address (byte offset, always multiple of 16) |
| `@P0` | Guard predicate (execute only if P0 is true) |
| `FFMA` | Opcode |
| `.FTZ` | Modifier (flush-to-zero) |
| `R4` | Destination register |
| `\|R2\|.reuse` | Source 1: absolute value + operand reuse flag |
| `-RZ` | Source 2: negated zero register |
| `R5` | Source 3 |
| `/* hex */` | 128-bit binary encoding (two 64-bit words) |

### Operand Types

| Notation | Meaning |
|---|---|
| `R0`–`R254` | 32-bit general registers |
| `R255` / `RZ` | Zero register (always 0) |
| `R0R1` / `R2.64` | 64-bit register pair |
| `UR0`–`URx` | Uniform registers (same across warp) |
| `P0`–`P6` | Predicate registers (per-thread boolean) |
| `P7` / `PT` | Always-true predicate |
| `c[X][Y]` | Constant memory bank X, offset Y |
| `desc[URX][RY]` | Memory descriptor (Hopper+) |
| `0x1234` | Immediate constant |

### Predicate Syntax

- `@P0 INST` — execute if P0 true
- `@!P0 INST` — execute if P0 false
- `@PT INST` — always execute
- `@!PT INST` — never execute (scheduling slot / NOP)

### Common Modifiers

| Modifier | Meaning |
|---|---|
| `.FTZ` | Flush denormals to zero |
| `.SAT` | Saturate to [0,1] |
| `.RN/.RM/.RP/.RZ` | Rounding mode |
| `.E` | Extended (64-bit) address |
| `.128/.64/.32` | Data size |
| `.STRONG/.SYS/.CTA/.GPU` | Memory ordering/scope |
| `.reuse` | Keep in operand collector cache |
| `.WIDE` | Wide result (32×32→64) |
| `.HI/.LO` | High/low part of wide result |

## Control Codes (Scheduling)

On Volta+ (SM70+), each 128-bit instruction embeds its own scheduling metadata. **Not officially documented** — reverse-engineered by maxas, CuAssembler, academic papers.

### Fields Per Instruction

| Field | Bits | Description |
|---|---|---|
| **Stall count** | 4 (0-15) | Min cycles before next instruction issues |
| **Yield flag** | 1 | Hint: try switching to another warp |
| **Write barrier** | 3 (0-5) | Barrier armed when instruction completes writing |
| **Read barrier** | 3 (0-5) | Barrier armed when instruction finishes reading |
| **Wait barrier mask** | 6 | Which barriers must clear before this instruction issues |
| **Reuse flags** | 4 | Which operands to cache in operand collector |

### Stall Count

The primary mechanism for fixed-latency dependencies:
- **0**: Back-to-back issue (no dependency or pipeline forwarding)
- **1-4**: Common for arithmetic chains (FFMA has ~4-cycle latency)
- **5-15**: Longer dependencies or intentional spacing

### Barrier Mechanism (6 barriers)

Handles variable-latency instructions (memory loads, tensor core ops):

```
LDG.E R4, [R2]       ; write-barrier = 1 (arms barrier 1 when LDG completes)
IADD3 R6, R7, R8, RZ ; independent work (stall=4, no barrier wait)
FFMA R9, R10, R11, RZ ; more independent work
IMAD R12, R4, R5, RZ  ; wait-barrier-mask = 0b000010 (wait for barrier 1)
```

The compiler interleaves independent work between producer and consumer to hide latency.

### Yield Flag

When set: scheduler prefers switching warps (good after memory loads).
When clear: scheduler prefers continuing current warp (good in tight compute loops).
When yield is set, operand reuse flags become ineffective.

## Key Instruction Mnemonics

### Floating-Point

| Mnemonic | Description | Latency (SM80) |
|---|---|---|
| `FFMA` | FP32 fused multiply-add | 4 cycles |
| `FADD` / `FMUL` | FP32 add/multiply | 4 cycles |
| `HFMA2` | Packed FP16×2 FMA | 4 cycles |
| `HADD2` / `HMUL2` | Packed FP16×2 add/mul | 4 cycles |
| `DFMA` | FP64 FMA | 4 cycles |
| `MUFU` | Transcendental (sin, cos, rsqrt, exp2, lg2) | 4 cycles |

### Tensor Core

| Mnemonic | Description | Latency (SM80) |
|---|---|---|
| `HMMA.16816.F16` | FP16 tensor core MMA | 8 cycles |
| `HMMA.1684.F32.TF32` | TF32 tensor core MMA | 4 cycles |
| `IMMA.16816.U8` | INT8 tensor core MMA | 4 cycles |
| `DMMA.884` | FP64 tensor core MMA | 16 cycles |
| `HGMMA`/`IGMMA` | Hopper warpgroup MMA | varies |
| `OMMA`/`QMMA` | Blackwell MMA | varies |

### Global Memory

| Mnemonic | Description | Latency (SM80) |
|---|---|---|
| `LDG` | Load global | ~34 (L1 hit), ~200 (L2 hit), ~290 (HBM) |
| `STG` | Store global | variable |
| `LDG.E.128` | 128-bit global load (4×R32) | same |
| `LDGSTS` | Async global→shared (cp.async) | non-blocking |
| `LDGDEPBAR` | cp.async commit group | — |
| `ATOMG` / `RED` | Atomic / reduction on global | variable |

### Shared Memory

| Mnemonic | Description | Latency (SM80) |
|---|---|---|
| `LDS` | Load shared | ~23 cycles |
| `STS` | Store shared | ~19 cycles |
| `LDSM` | ldmatrix (warp cooperative matrix load) | ~23 cycles |
| `STSM` | stmatrix (Hopper) | similar |
| `ATOMS` | Atomic on shared | variable |

### Integer / Address

| Mnemonic | Description | Latency |
|---|---|---|
| `IMAD` | Integer multiply-add | 4 cycles |
| `IADD3` | Integer 3-operand add | 2 cycles |
| `LEA` | Load effective address | 2 cycles |
| `LOP3` | 3-input boolean (replaces AND/OR/XOR) | 2 cycles |
| `SHF` / `SHL` / `SHR` | Shift operations | 2 cycles |
| `PRMT` | Byte permutation | 2 cycles |
| `MOV` | Register move | 2 cycles |

### Control Flow

| Mnemonic | Description |
|---|---|
| `BRA` | Branch |
| `EXIT` | Thread exit |
| `BSSY` | Begin divergent section (set convergence point) |
| `BSYNC` | Synchronize at convergence point |
| `BAR.SYNC` | `__syncthreads()` |
| `DEPBAR` | cp.async wait (`DEPBAR.LE SB0, 0x0` = wait all) |
| `MEMBAR` | `__threadfence()` (`.CTA`, `.GPU`, `.SYS` scopes) |
| `WARPSYNC` | Warp synchronization |

### Special

| Mnemonic | Description |
|---|---|
| `S2R` | Read special register (`S2R R0, SR_TID.X` = threadIdx.x) |
| `VOTE` | Warp vote (ballot, all, any) |
| `SHFL` | Warp shuffle |
| `REDUX` | Warp reduction (Ampere+) |
| `NOP` | No operation (scheduling padding) |

## Identifying Performance Issues from SASS

### 1. Latency Not Hidden

Producer (LDG) and consumer (using that register) too close together:
```
LDG.E R4, [R2]         ; issues load, arms barrier
FFMA R6, R4, R5, R7    ; STALL: waiting for R4 from global memory
```
**Fix**: Interleave independent instructions between load and use.

### 2. Register Spilling

`LDL` / `STL` (local memory) instructions = register spilling:
```
STL [R1+0x0], R32   ; spill to local memory (slow!)
... many instructions ...
LDL R32, [R1+0x0]   ; reload from local memory
```
**Fix**: Reduce register pressure (`__launch_bounds__`, simplify kernel).

### 3. NOP Padding

```
NOP    ; stall=15
NOP    ; stall=15
```
Compiler couldn't find useful work. Indicates either a very short kernel or insufficient ILP.

### 4. Warp Divergence

`BSSY`/`BSYNC` pairs around conditional branches:
```
BSSY B0, target
@P0 BRA else_label
... then-branch ...
BRA endif
else_label:
... else-branch ...
endif:
BSYNC B0          ; reconverge
```

### 5. Barrier Usage

Count how many of the 6 barriers are used. All 6 = maximum variable-latency ops in flight. Fewer = room to overlap more loads/MMA.

### What to Look For (Summary)

1. **Instruction mix**: compute (FFMA/HMMA) vs memory (LDG/LDS) vs overhead (IMAD/NOP/BAR)?
2. **Latency hiding**: distance between memory loads and their consumers?
3. **Stall counts**: consistently high = poor ILP, low = good overlap
4. **Register spilling**: any LDL/STL?
5. **Tensor core pattern**: LDSM → HMMA ratio?
6. **Async copies**: LDGSTS + DEPBAR pipelining?
7. **Divergence**: BSSY/BSYNC complexity?
8. **Register count**: high → low occupancy

## Correlating with Nsight Compute

| ncu Stall Reason | SASS Cause |
|---|---|
| Long scoreboard | Waiting for LDG/LDL result (not ready) |
| Short scoreboard | Waiting for LDS/HMMA/MUFU result |
| Wait | Fixed-latency dependency (FFMA chain, stall count too low) |
| Math pipe throttle | FMA/tensor pipe FIFO full (compute saturated) |
| Barrier | At BAR.SYNC instruction |
| LG throttle | Too many outstanding LDG/STG |
| MIO throttle | Too many outstanding LDS/STS/MUFU |
| Not selected | Eligible but another warp chosen (occupancy may be reducible) |

Use `ncu --section SourceCounters` with `-lineinfo` to correlate stalls back to source lines, and `ncu --page source` to view SASS alongside source.

## Tools and Resources

| Tool | Purpose |
|---|---|
| `cuobjdump --dump-sass` | Extract SASS from binaries |
| `nvdisasm` | Rich disassembler with CFG, liveness, source correlation |
| `ncu --page source` | SASS with per-instruction profiling metrics |
| [CuAssembler](https://github.com/cloudcores/CuAssembler) | Unofficial assembler, control code decoding |
| [DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS) | Community instruction docs (SM89/SM90a) |
| [maxas wiki](https://github.com/NervanaSystems/maxas/wiki/Control-Codes) | Control code format reference (Maxwell, foundational) |
| [Kuter Dinel's Control Code Viewer](https://kuterdinel.com/nvidia-sass-control-code-viewer.html) | Interactive decoder |
