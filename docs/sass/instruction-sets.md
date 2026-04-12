# SASS Instruction Set Reference

Source: https://docs.nvidia.com/cuda/cuda-binary-utilities/#instruction-set-reference

Complete list of SASS instructions per GPU architecture.

# 4. Instruction Set Reference[](#instruction-set-reference "Permalink to this headline")

This section contains instruction set reference for NVIDIA® GPU architectures.

## 4.1. Turing Instruction Set[](#turing-instruction-set "Permalink to this headline")

> The Turing architecture (Compute Capability 7.5) have the following instruction set format:

```
(instruction) (destination) (source1), (source2) ...
```

Valid destination and source locations include:

* RX for registers
* URX for uniform registers
* SRX for special system-controlled registers
* PX for predicate registers
* c[X][Y] for constant memory

[Table 6](#turing-turing-instruction-set-table) lists valid instructions for the Turing GPUs.

Table 6. Turing Instruction Set[](#turing-turing-instruction-set-table "Permalink to this table")


| Opcode | Description |
| --- | --- |
| **Floating Point Instructions** | |
| FADD | FP32 Add |
| FADD32I | FP32 Add |
| FCHK | Floating-point Range Check |
| FFMA32I | FP32 Fused Multiply and Add |
| FFMA | FP32 Fused Multiply and Add |
| FMNMX | FP32 Minimum/Maximum |
| FMUL | FP32 Multiply |
| FMUL32I | FP32 Multiply |
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FSWZADD | FP32 Swizzle Add |
| MUFU | FP32 Multi Function Operation |
| HADD2 | FP16 Add |
| HADD2\_32I | FP16 Add |
| HFMA2 | FP16 Fused Mutiply Add |
| HFMA2\_32I | FP16 Fused Mutiply Add |
| HMMA | Matrix Multiply and Accumulate |
| HMUL2 | FP16 Multiply |
| HMUL2\_32I | FP16 Multiply |
| HSET2 | FP16 Compare And Set |
| HSETP2 | FP16 Compare And Set Predicate |
| DADD | FP64 Add |
| DFMA | FP64 Fused Mutiply Add |
| DMUL | FP64 Multiply |
| DSETP | FP64 Compare And Set Predicate |
| **Integer Instructions** | |
| BMMA | Bit Matrix Multiply and Accumulate |
| BMSK | Bitfield Mask |
| BREV | Bit Reverse |
| FLO | Find Leading One |
| IABS | Integer Absolute Value |
| IADD | Integer Addition |
| IADD3 | 3-input Integer Addition |
| IADD32I | Integer Addition |
| IDP | Integer Dot Product and Accumulate |
| IDP4A | Integer Dot Product and Accumulate |
| IMAD | Integer Multiply And Add |
| IMMA | Integer Matrix Multiply and Accumulate |
| IMNMX | Integer Minimum/Maximum |
| IMUL | Integer Multiply |
| IMUL32I | Integer Multiply |
| ISCADD | Scaled Integer Addition |
| ISCADD32I | Scaled Integer Addition |
| ISETP | Integer Compare And Set Predicate |
| LEA | LOAD Effective Address |
| LOP | Logic Operation |
| LOP3 | Logic Operation |
| LOP32I | Logic Operation |
| POPC | Population count |
| SHF | Funnel Shift |
| SHL | Shift Left |
| SHR | Shift Right |
| VABSDIFF | Absolute Difference |
| VABSDIFF4 | Absolute Difference |
| **Conversion Instructions** | |
| F2F | Floating Point To Floating Point Conversion |
| F2I | Floating Point To Integer Conversion |
| I2F | Integer To Floating Point Conversion |
| I2I | Integer To Integer Conversion |
| I2IP | Integer To Integer Conversion and Packing |
| FRND | Round To Integer |
| **Movement Instructions** | |
| MOV | Move |
| MOV32I | Move |
| MOVM | Move Matrix with Transposition or Expansion |
| PRMT | Permute Register Pair |
| SEL | Select Source with Predicate |
| SGXT | Sign Extend |
| SHFL | Warp Wide Register Shuffle |
| **Predicate Instructions** | |
| PLOP3 | Predicate Logic Operation |
| PSETP | Combine Predicates and Set Predicate |
| P2R | Move Predicate Register To Register |
| R2P | Move Register To Predicate Register |
| **Load/Store Instructions** | |
| LD | Load from generic Memory |
| LDC | Load Constant |
| LDG | Load from Global Memory |
| LDL | Load within Local Memory Window |
| LDS | Load within Shared Memory Window |
| LDSM | Load Matrix from Shared Memory with Element Size Expansion |
| ST | Store to Generic Memory |
| STG | Store to Global Memory |
| STL | Store to Local Memory |
| STS | Store to Shared Memory |
| MATCH | Match Register Values Across Thread Group |
| QSPC | Query Space |
| ATOM | Atomic Operation on Generic Memory |
| ATOMS | Atomic Operation on Shared Memory |
| ATOMG | Atomic Operation on Global Memory |
| RED | Reduction Operation on Generic Memory |
| CCTL | Cache Control |
| CCTLL | Cache Control |
| ERRBAR | Error Barrier |
| MEMBAR | Memory Barrier |
| CCTLT | Texture Cache Control |
| **Uniform Datapath Instructions** | |
| R2UR | Move from Vector Register to a Uniform Register |
| S2UR | Move Special Register to Uniform Register |
| UBMSK | Uniform Bitfield Mask |
| UBREV | Uniform Bit Reverse |
| UCLEA | Load Effective Address for a Constant |
| UFLO | Uniform Find Leading One |
| UIADD3 | Uniform Integer Addition |
| UIADD3.64 | Uniform Integer Addition |
| UIMAD | Uniform Integer Multiplication |
| UISETP | Integer Compare and Set Uniform Predicate |
| ULDC | Load from Constant Memory into a Uniform Register |
| ULEA | Uniform Load Effective Address |
| ULOP | Logic Operation |
| ULOP3 | Logic Operation |
| ULOP32I | Logic Operation |
| UMOV | Uniform Move |
| UP2UR | Uniform Predicate to Uniform Register |
| UPLOP3 | Uniform Predicate Logic Operation |
| UPOPC | Uniform Population Count |
| UPRMT | Uniform Byte Permute |
| UPSETP | Uniform Predicate Logic Operation |
| UR2UP | Uniform Register to Uniform Predicate |
| USEL | Uniform Select |
| USGXT | Uniform Sign Extend |
| USHF | Uniform Funnel Shift |
| USHL | Uniform Left Shift |
| USHR | Uniform Right Shift |
| VOTEU | Voting across SIMD Thread Group with Results in Uniform Destination |
| **Texture Instructions** | |
| TEX | Texture Fetch |
| TLD | Texture Load |
| TLD4 | Texture Load 4 |
| TMML | Texture MipMap Level |
| TXD | Texture Fetch With Derivatives |
| TXQ | Texture Query |
| **Surface Instructions** | |
| SUATOM | Atomic Op on Surface Memory |
| SULD | Surface Load |
| SURED | Reduction Op on Surface Memory |
| SUST | Surface Store |
| **Control Instructions** | |
| BMOV | Move Convergence Barrier State |
| BPT | BreakPoint/Trap |
| BRA | Relative Branch |
| BREAK | Break out of the Specified Convergence Barrier |
| BRX | Relative Branch Indirect |
| BRXU | Relative Branch with Uniform Register Based Offset |
| BSSY | Barrier Set Convergence Synchronization Point |
| BSYNC | Synchronize Threads on a Convergence Barrier |
| CALL | Call Function |
| EXIT | Exit Program |
| JMP | Absolute Jump |
| JMX | Absolute Jump Indirect |
| JMXU | Absolute Jump with Uniform Register Based Offset |
| KILL | Kill Thread |
| NANOSLEEP | Suspend Execution |
| RET | Return From Subroutine |
| RPCMOV | PC Register Move |
| RTT | Return From Trap |
| WARPSYNC | Synchronize Threads in Warp |
| YIELD | Yield Control |
| **Miscellaneous Instructions** | |
| B2R | Move Barrier To Register |
| BAR | Barrier Synchronization |
| CS2R | Move Special Register to Register |
| DEPBAR | Dependency Barrier |
| GETLMEMBASE | Get Local Memory Base Address |
| LEPC | Load Effective PC |
| NOP | No Operation |
| PMTRIG | Performance Monitor Trigger |
| R2B | Move Register to Barrier |
| S2R | Move Special Register to Register |
| SETCTAID | Set CTA ID |
| SETLMEMBASE | Set Local Memory Base Address |
| VOTE | Vote Across SIMD Thread Group |

## 4.2. NVIDIA Ampere GPU and Ada Instruction Set[](#nvidia-ampere-gpu-and-ada-instruction-set "Permalink to this headline")

The NVIDIA Ampere GPU and Ada architectures (Compute Capability 8.0, 8.6, and 8.9) have the following instruction set format:

```
(instruction) (destination) (source1), (source2) ...
```

Valid destination and source locations include:

* RX for registers
* URX for uniform registers
* SRX for special system-controlled registers
* PX for predicate registers
* UPX for uniform predicate registers
* c[X][Y] for constant memory

[Table 7](#ampere-ampere-instruction-set-table) lists valid instructions for the NVIDIA Ampere architecrture and Ada GPUs.

Table 7. NVIDIA Ampere GPU and Ada Instruction Set[](#ampere-ampere-instruction-set-table "Permalink to this table")


| Opcode | Description |
| --- | --- |
| **Floating Point Instructions** |  |
| FADD | FP32 Add |
| FADD32I | FP32 Add |
| FCHK | Floating-point Range Check |
| FFMA32I | FP32 Fused Multiply and Add |
| FFMA | FP32 Fused Multiply and Add |
| FMNMX | FP32 Minimum/Maximum |
| FMUL | FP32 Multiply |
| FMUL32I | FP32 Multiply |
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FSWZADD | FP32 Swizzle Add |
| MUFU | FP32 Multi Function Operation |
| HADD2 | FP16 Add |
| HADD2\_32I | FP16 Add |
| HFMA2 | FP16 Fused Mutiply Add |
| HFMA2\_32I | FP16 Fused Mutiply Add |
| HMMA | Matrix Multiply and Accumulate |
| HMNMX2 | FP16 Minimum / Maximum |
| HMUL2 | FP16 Multiply |
| HMUL2\_32I | FP16 Multiply |
| HSET2 | FP16 Compare And Set |
| HSETP2 | FP16 Compare And Set Predicate |
| DADD | FP64 Add |
| DFMA | FP64 Fused Mutiply Add |
| DMMA | Matrix Multiply and Accumulate |
| DMUL | FP64 Multiply |
| DSETP | FP64 Compare And Set Predicate |
| **Integer Instructions** |  |
| BMMA | Bit Matrix Multiply and Accumulate |
| BMSK | Bitfield Mask |
| BREV | Bit Reverse |
| FLO | Find Leading One |
| IABS | Integer Absolute Value |
| IADD | Integer Addition |
| IADD3 | 3-input Integer Addition |
| IADD32I | Integer Addition |
| IDP | Integer Dot Product and Accumulate |
| IDP4A | Integer Dot Product and Accumulate |
| IMAD | Integer Multiply And Add |
| IMMA | Integer Matrix Multiply and Accumulate |
| IMNMX | Integer Minimum/Maximum |
| IMUL | Integer Multiply |
| IMUL32I | Integer Multiply |
| ISCADD | Scaled Integer Addition |
| ISCADD32I | Scaled Integer Addition |
| ISETP | Integer Compare And Set Predicate |
| LEA | LOAD Effective Address |
| LOP | Logic Operation |
| LOP3 | Logic Operation |
| LOP32I | Logic Operation |
| POPC | Population count |
| SHF | Funnel Shift |
| SHL | Shift Left |
| SHR | Shift Right |
| VABSDIFF | Absolute Difference |
| VABSDIFF4 | Absolute Difference |
| **Conversion Instructions** |  |
| F2F | Floating Point To Floating Point Conversion |
| F2I | Floating Point To Integer Conversion |
| I2F | Integer To Floating Point Conversion |
| I2I | Integer To Integer Conversion |
| I2IP | Integer To Integer Conversion and Packing |
| I2FP | Integer to FP32 Convert and Pack |
| F2IP | FP32 Down-Convert to Integer and Pack |
| FRND | Round To Integer |
| **Movement Instructions** |  |
| MOV | Move |
| MOV32I | Move |
| MOVM | Move Matrix with Transposition or Expansion |
| PRMT | Permute Register Pair |
| SEL | Select Source with Predicate |
| SGXT | Sign Extend |
| SHFL | Warp Wide Register Shuffle |
| **Predicate Instructions** |  |
| PLOP3 | Predicate Logic Operation |
| PSETP | Combine Predicates and Set Predicate |
| P2R | Move Predicate Register To Register |
| R2P | Move Register To Predicate Register |
| **Load/Store Instructions** |  |
| LD | Load from generic Memory |
| LDC | Load Constant |
| LDG | Load from Global Memory |
| LDGDEPBAR | Global Load Dependency Barrier |
| LDGSTS | Asynchronous Global to Shared Memcopy |
| LDL | Load within Local Memory Window |
| LDS | Load within Shared Memory Window |
| LDSM | Load Matrix from Shared Memory with Element Size Expansion |
| ST | Store to Generic Memory |
| STG | Store to Global Memory |
| STL | Store to Local Memory |
| STS | Store to Shared Memory |
| MATCH | Match Register Values Across Thread Group |
| QSPC | Query Space |
| ATOM | Atomic Operation on Generic Memory |
| ATOMS | Atomic Operation on Shared Memory |
| ATOMG | Atomic Operation on Global Memory |
| RED | Reduction Operation on Generic Memory |
| CCTL | Cache Control |
| CCTLL | Cache Control |
| ERRBAR | Error Barrier |
| MEMBAR | Memory Barrier |
| CCTLT | Texture Cache Control |
| **Uniform Datapath Instructions** |  |
| R2UR | Move from Vector Register to a Uniform Register |
| REDUX | Reduction of a Vector Register into a Uniform Register |
| S2UR | Move Special Register to Uniform Register |
| UBMSK | Uniform Bitfield Mask |
| UBREV | Uniform Bit Reverse |
| UCLEA | Load Effective Address for a Constant |
| UF2FP | Uniform FP32 Down-convert and Pack |
| UFLO | Uniform Find Leading One |
| UIADD3 | Uniform Integer Addition |
| UIADD3.64 | Uniform Integer Addition |
| UIMAD | Uniform Integer Multiplication |
| UISETP | Integer Compare and Set Uniform Predicate |
| ULDC | Load from Constant Memory into a Uniform Register |
| ULEA | Uniform Load Effective Address |
| ULOP | Logic Operation |
| ULOP3 | Logic Operation |
| ULOP32I | Logic Operation |
| UMOV | Uniform Move |
| UP2UR | Uniform Predicate to Uniform Register |
| UPLOP3 | Uniform Predicate Logic Operation |
| UPOPC | Uniform Population Count |
| UPRMT | Uniform Byte Permute |
| UPSETP | Uniform Predicate Logic Operation |
| UR2UP | Uniform Register to Uniform Predicate |
| USEL | Uniform Select |
| USGXT | Uniform Sign Extend |
| USHF | Uniform Funnel Shift |
| USHL | Uniform Left Shift |
| USHR | Uniform Right Shift |
| VOTEU | Voting across SIMD Thread Group with Results in Uniform Destination |
| **Texture Instructions** |  |
| TEX | Texture Fetch |
| TLD | Texture Load |
| TLD4 | Texture Load 4 |
| TMML | Texture MipMap Level |
| TXD | Texture Fetch With Derivatives |
| TXQ | Texture Query |
| **Surface Instructions** |  |
| SUATOM | Atomic Op on Surface Memory |
| SULD | Surface Load |
| SURED | Reduction Op on Surface Memory |
| SUST | Surface Store |
| **Control Instructions** |  |
| BMOV | Move Convergence Barrier State |
| BPT | BreakPoint/Trap |
| BRA | Relative Branch |
| BREAK | Break out of the Specified Convergence Barrier |
| BRX | Relative Branch Indirect |
| BRXU | Relative Branch with Uniform Register Based Offset |
| BSSY | Barrier Set Convergence Synchronization Point |
| BSYNC | Synchronize Threads on a Convergence Barrier |
| CALL | Call Function |
| EXIT | Exit Program |
| JMP | Absolute Jump |
| JMX | Absolute Jump Indirect |
| JMXU | Absolute Jump with Uniform Register Based Offset |
| KILL | Kill Thread |
| NANOSLEEP | Suspend Execution |
| RET | Return From Subroutine |
| RPCMOV | PC Register Move |
| WARPSYNC | Synchronize Threads in Warp |
| YIELD | Yield Control |
| **Miscellaneous Instructions** |  |
| B2R | Move Barrier To Register |
| BAR | Barrier Synchronization |
| CS2R | Move Special Register to Register |
| DEPBAR | Dependency Barrier |
| GETLMEMBASE | Get Local Memory Base Address |
| LEPC | Load Effective PC |
| NOP | No Operation |
| PMTRIG | Performance Monitor Trigger |
| S2R | Move Special Register to Register |
| SETCTAID | Set CTA ID |
| SETLMEMBASE | Set Local Memory Base Address |
| VOTE | Vote Across SIMD Thread Group |

## 4.3. Hopper Instruction Set[](#hopper-instruction-set "Permalink to this headline")

The Hopper architecture (Compute Capability 9.0) has the following instruction set format:

```
(instruction) (destination) (source1), (source2) ...
```

Valid destination and source locations include:

* RX for registers
* URX for uniform registers
* SRX for special system-controlled registers
* PX for predicate registers
* UPX for uniform predicate registers
* c[X][Y] for constant memory
* desc[URX][RY] for memory descriptors
* gdesc[URX] for global memory descriptors

[Table 8](#hopper-hopper-instruction-set-table) lists valid instructions for the Hopper GPUs.

Table 8. Hopper Instruction Set[](#hopper-hopper-instruction-set-table "Permalink to this table")


| Opcode | Description |
| --- | --- |
| **Floating Point Instructions** | |
| FADD | FP32 Add |
| FADD32I | FP32 Add |
| FCHK | Floating-point Range Check |
| FFMA32I | FP32 Fused Multiply and Add |
| FFMA | FP32 Fused Multiply and Add |
| FMNMX | FP32 Minimum/Maximum |
| FMUL | FP32 Multiply |
| FMUL32I | FP32 Multiply |
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FSWZADD | FP32 Swizzle Add |
| MUFU | FP32 Multi Function Operation |
| HADD2 | FP16 Add |
| HADD2\_32I | FP16 Add |
| HFMA2 | FP16 Fused Mutiply Add |
| HFMA2\_32I | FP16 Fused Mutiply Add |
| HMMA | Matrix Multiply and Accumulate |
| HMNMX2 | FP16 Minimum / Maximum |
| HMUL2 | FP16 Multiply |
| HMUL2\_32I | FP16 Multiply |
| HSET2 | FP16 Compare And Set |
| HSETP2 | FP16 Compare And Set Predicate |
| DADD | FP64 Add |
| DFMA | FP64 Fused Mutiply Add |
| DMMA | Matrix Multiply and Accumulate |
| DMUL | FP64 Multiply |
| DSETP | FP64 Compare And Set Predicate |
| **Integer Instructions** | |
| BMMA | Bit Matrix Multiply and Accumulate |
| BMSK | Bitfield Mask |
| BREV | Bit Reverse |
| FLO | Find Leading One |
| IABS | Integer Absolute Value |
| IADD | Integer Addition |
| IADD3 | 3-input Integer Addition |
| IADD32I | Integer Addition |
| IDP | Integer Dot Product and Accumulate |
| IDP4A | Integer Dot Product and Accumulate |
| IMAD | Integer Multiply And Add |
| IMMA | Integer Matrix Multiply and Accumulate |
| IMNMX | Integer Minimum/Maximum |
| IMUL | Integer Multiply |
| IMUL32I | Integer Multiply |
| ISCADD | Scaled Integer Addition |
| ISCADD32I | Scaled Integer Addition |
| ISETP | Integer Compare And Set Predicate |
| LEA | LOAD Effective Address |
| LOP | Logic Operation |
| LOP3 | Logic Operation |
| LOP32I | Logic Operation |
| POPC | Population count |
| SHF | Funnel Shift |
| SHL | Shift Left |
| SHR | Shift Right |
| VABSDIFF | Absolute Difference |
| VABSDIFF4 | Absolute Difference |
| VHMNMX | SIMD FP16 3-Input Minimum / Maximum |
| VIADD | SIMD Integer Addition |
| VIADDMNMX | SIMD Integer Addition and Fused Min/Max Comparison |
| VIMNMX | SIMD Integer Minimum / Maximum |
| VIMNMX3 | SIMD Integer 3-Input Minimum / Maximum |
| **Conversion Instructions** | |
| F2F | Floating Point To Floating Point Conversion |
| F2I | Floating Point To Integer Conversion |
| I2F | Integer To Floating Point Conversion |
| I2I | Integer To Integer Conversion |
| I2IP | Integer To Integer Conversion and Packing |
| I2FP | Integer to FP32 Convert and Pack |
| F2IP | FP32 Down-Convert to Integer and Pack |
| FRND | Round To Integer |
| **Movement Instructions** | |
| MOV | Move |
| MOV32I | Move |
| MOVM | Move Matrix with Transposition or Expansion |
| PRMT | Permute Register Pair |
| SEL | Select Source with Predicate |
| SGXT | Sign Extend |
| SHFL | Warp Wide Register Shuffle |
| **Predicate Instructions** | |
| PLOP3 | Predicate Logic Operation |
| PSETP | Combine Predicates and Set Predicate |
| P2R | Move Predicate Register To Register |
| R2P | Move Register To Predicate Register |
| **Load/Store Instructions** | |
| FENCE | Memory Visibility Guarantee for Shared or Global Memory |
| LD | Load from generic Memory |
| LDC | Load Constant |
| LDG | Load from Global Memory |
| LDGDEPBAR | Global Load Dependency Barrier |
| LDGMC | Reducing Load |
| LDGSTS | Asynchronous Global to Shared Memcopy |
| LDL | Load within Local Memory Window |
| LDS | Load within Shared Memory Window |
| LDSM | Load Matrix from Shared Memory with Element Size Expansion |
| STSM | Store Matrix to Shared Memory |
| ST | Store to Generic Memory |
| STG | Store to Global Memory |
| STL | Store to Local Memory |
| STS | Store to Shared Memory |
| STAS | Asynchronous Store to Distributed Shared Memory With Explicit Synchronization |
| SYNCS | Sync Unit |
| MATCH | Match Register Values Across Thread Group |
| QSPC | Query Space |
| ATOM | Atomic Operation on Generic Memory |
| ATOMS | Atomic Operation on Shared Memory |
| ATOMG | Atomic Operation on Global Memory |
| REDAS | Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization |
| REDG | Reduction Operation on Generic Memory |
| CCTL | Cache Control |
| CCTLL | Cache Control |
| ERRBAR | Error Barrier |
| MEMBAR | Memory Barrier |
| CCTLT | Texture Cache Control |
| **Uniform Datapath Instructions** | |
| R2UR | Move from Vector Register to a Uniform Register |
| REDUX | Reduction of a Vector Register into a Uniform Register |
| S2UR | Move Special Register to Uniform Register |
| UBMSK | Uniform Bitfield Mask |
| UBREV | Uniform Bit Reverse |
| UCGABAR\_ARV | CGA Barrier Synchronization |
| UCGABAR\_WAIT | CGA Barrier Synchronization |
| UCLEA | Load Effective Address for a Constant |
| UF2FP | Uniform FP32 Down-convert and Pack |
| UFLO | Uniform Find Leading One |
| UIADD3 | Uniform Integer Addition |
| UIADD3.64 | Uniform Integer Addition |
| UIMAD | Uniform Integer Multiplication |
| UISETP | Integer Compare and Set Uniform Predicate |
| ULDC | Load from Constant Memory into a Uniform Register |
| ULEA | Uniform Load Effective Address |
| ULEPC | Uniform Load Effective PC |
| ULOP | Logic Operation |
| ULOP3 | Logic Operation |
| ULOP32I | Logic Operation |
| UMOV | Uniform Move |
| UP2UR | Uniform Predicate to Uniform Register |
| UPLOP3 | Uniform Predicate Logic Operation |
| UPOPC | Uniform Population Count |
| UPRMT | Uniform Byte Permute |
| UPSETP | Uniform Predicate Logic Operation |
| UR2UP | Uniform Register to Uniform Predicate |
| USEL | Uniform Select |
| USETMAXREG | Release, Deallocate and Allocate Registers |
| USGXT | Uniform Sign Extend |
| USHF | Uniform Funnel Shift |
| USHL | Uniform Left Shift |
| USHR | Uniform Right Shift |
| VOTEU | Voting across SIMD Thread Group with Results in Uniform Destination |
| **Warpgroup Instructions** | |
| BGMMA | Bit Matrix Multiply and Accumulate Across Warps |
| HGMMA | Matrix Multiply and Accumulate Across a Warpgroup |
| IGMMA | Integer Matrix Multiply and Accumulate Across a Warpgroup |
| QGMMA | FP8 Matrix Multiply and Accumulate Across a Warpgroup |
| WARPGROUP | Warpgroup Synchronization |
| WARPGROUPSET | Set Warpgroup Counters |
| **Tensor Memory Access Instructions** | |
| UBLKCP | Bulk Data Copy |
| UBLKPF | Bulk Data Prefetch |
| UBLKRED | Bulk Data Copy from Shared Memory with Reduction |
| UTMACCTL | TMA Cache Control |
| UTMACMDFLUSH | TMA Command Flush |
| UTMALDG | Tensor Load from Global to Shared Memory |
| UTMAPF | Tensor Prefetch |
| UTMAREDG | Tensor Store from Shared to Global Memory with Reduction |
| UTMASTG | Tensor Store from Shared to Global Memory |
| **Texture Instructions** | |
| TEX | Texture Fetch |
| TLD | Texture Load |
| TLD4 | Texture Load 4 |
| TMML | Texture MipMap Level |
| TXD | Texture Fetch With Derivatives |
| TXQ | Texture Query |
| **Surface Instructions** | |
| SUATOM | Atomic Op on Surface Memory |
| SULD | Surface Load |
| SURED | Reduction Op on Surface Memory |
| SUST | Surface Store |
| **Control Instructions** | |
| ACQBULK | Wait for Bulk Release Status Warp State |
| BMOV | Move Convergence Barrier State |
| BPT | BreakPoint/Trap |
| BRA | Relative Branch |
| BREAK | Break out of the Specified Convergence Barrier |
| BRX | Relative Branch Indirect |
| BRXU | Relative Branch with Uniform Register Based Offset |
| BSSY | Barrier Set Convergence Synchronization Point |
| BSYNC | Synchronize Threads on a Convergence Barrier |
| CALL | Call Function |
| CGAERRBAR | CGA Error Barrier |
| ELECT | Elect a Leader Thread |
| ENDCOLLECTIVE | Reset the MCOLLECTIVE mask |
| EXIT | Exit Program |
| JMP | Absolute Jump |
| JMX | Absolute Jump Indirect |
| JMXU | Absolute Jump with Uniform Register Based Offset |
| KILL | Kill Thread |
| NANOSLEEP | Suspend Execution |
| PREEXIT | Dependent Task Launch Hint |
| RET | Return From Subroutine |
| RPCMOV | PC Register Move |
| WARPSYNC | Synchronize Threads in Warp |
| YIELD | Yield Control |
| **Miscellaneous Instructions** | |
| B2R | Move Barrier To Register |
| BAR | Barrier Synchronization |
| CS2R | Move Special Register to Register |
| DEPBAR | Dependency Barrier |
| GETLMEMBASE | Get Local Memory Base Address |
| LEPC | Load Effective PC |
| NOP | No Operation |
| PMTRIG | Performance Monitor Trigger |
| S2R | Move Special Register to Register |
| SETCTAID | Set CTA ID |
| SETLMEMBASE | Set Local Memory Base Address |
| VOTE | Vote Across SIMT Thread Group |

## 4.4. Blackwell Instruction Set[](#blackwell-instruction-set "Permalink to this headline")

The Blackwell architecture (Compute Capability 10.0 and 12.0) has the following instruction set format:

```
(instruction) (destination) (source1), (source2) ...
```

Valid destination and source locations include:

* RX for registers
* URX for uniform registers
* SRX for special system-controlled registers
* PX for predicate registers
* UPX for uniform predicate registers
* c[X][Y] for constant memory
* desc[URX][RY] for memory descriptors
* gdesc[URX] for global memory descriptors
* tmem[URX] for tensor memory

[Table 8](index.html#blackwell-blackwell-instruction-set) lists valid instructions for the Blackwell GPUs.

Table 8. Blackwell Instruction Set[](#id13 "Permalink to this table")


| Opcode | Description |
| --- | --- |
| **Floating Point Instructions** |  |
| FADD | FP32 Add |
| FADD2 | FP32 Add |
| FADD32I | FP32 Add |
| FCHK | Floating-point Range Check |
| FFMA32I | FP32 Fused Multiply and Add |
| FFMA | FP32 Fused Multiply and Add |
| FFMA2 | FP32 Fused Multiply and Add |
| FHADD | FP32 Addition |
| FHFMA | FP32 Fused Multiply and Add |
| FMNMX | FP32 Minimum/Maximum |
| FMNMX3 | 3-Input Floating-point Minimum / Maximum |
| FMUL | FP32 Multiply |
| FMUL2 | FP32 Multiply |
| FMUL32I | FP32 Multiply |
| FSEL | Floating Point Select |
| FSET | FP32 Compare And Set |
| FSETP | FP32 Compare And Set Predicate |
| FSWZADD | FP32 Swizzle Add |
| MUFU | FP32 Multi Function Operation |
| HADD2 | FP16 Add |
| HADD2\_32I | FP16 Add |
| HFMA2 | FP16 Fused Mutiply Add |
| HFMA2\_32I | FP16 Fused Mutiply Add |
| HMMA | Matrix Multiply and Accumulate |
| HMNMX2 | FP16 Minimum / Maximum |
| HMUL2 | FP16 Multiply |
| HMUL2\_32I | FP16 Multiply |
| HSET2 | FP16 Compare And Set |
| HSETP2 | FP16 Compare And Set Predicate |
| DADD | FP64 Add |
| DFMA | FP64 Fused Mutiply Add |
| DMMA | Matrix Multiply and Accumulate |
| DMUL | FP64 Multiply |
| DSETP | FP64 Compare And Set Predicate |
| OMMA | FP4 Matrix Multiply and Accumulate Across a Warp |
| QMMA | FP8 Matrix Multiply and Accumulate Across a Warp |
| **Integer Instructions** |  |
| BMSK | Bitfield Mask |
| BREV | Bit Reverse |
| FLO | Find Leading One |
| IABS | Integer Absolute Value |
| IADD | Integer Addition |
| IADD3 | 3-input Integer Addition |
| IADD32I | Integer Addition |
| IDP | Integer Dot Product and Accumulate |
| IDP4A | Integer Dot Product and Accumulate |
| IMAD | Integer Multiply And Add |
| IMMA | Integer Matrix Multiply and Accumulate |
| IMNMX | Integer Minimum/Maximum |
| IMUL | Integer Multiply |
| IMUL32I | Integer Multiply |
| ISCADD | Scaled Integer Addition |
| ISCADD32I | Scaled Integer Addition |
| ISETP | Integer Compare And Set Predicate |
| LEA | LOAD Effective Address |
| LOP | Logic Operation |
| LOP3 | Logic Operation |
| LOP32I | Logic Operation |
| POPC | Population count |
| SHF | Funnel Shift |
| SHL | Shift Left |
| SHR | Shift Right |
| VABSDIFF | Absolute Difference |
| VABSDIFF4 | Absolute Difference |
| VHMNMX | SIMD FP16 3-Input Minimum / Maximum |
| VIADD | SIMD Integer Addition |
| VIADDMNMX | SIMD Integer Addition and Fused Min/Max Comparison |
| VIMNMX | SIMD Integer Minimum / Maximum |
| VIMNMX3 | SIMD Integer 3-Input Minimum / Maximum |
| **Conversion Instructions** |  |
| F2F | Floating Point To Floating Point Conversion |
| F2I | Floating Point To Integer Conversion |
| I2F | Integer To Floating Point Conversion |
| I2I | Integer To Integer Conversion |
| I2IP | Integer To Integer Conversion and Packing |
| I2FP | Integer to FP32 Convert and Pack |
| F2IP | FP32 Down-Convert to Integer and Pack |
| FRND | Round To Integer |
| **Movement Instructions** |  |
| MOV | Move |
| MOV32I | Move |
| MOVM | Move Matrix with Transposition or Expansion |
| PRMT | Permute Register Pair |
| SEL | Select Source with Predicate |
| SGXT | Sign Extend |
| SHFL | Warp Wide Register Shuffle |
| **Predicate Instructions** |  |
| PLOP3 | Predicate Logic Operation |
| PSETP | Combine Predicates and Set Predicate |
| P2R | Move Predicate Register To Register |
| R2P | Move Register To Predicate Register |
| **Load/Store Instructions** |  |
| FENCE | Memory Visibility Guarantee for Shared or Global Memory |
| LD | Load from generic Memory |
| LDC | Load Constant |
| LDG | Load from Global Memory |
| LDGDEPBAR | Global Load Dependency Barrier |
| LDGMC | Reducing Load |
| LDGSTS | Asynchronous Global to Shared Memcopy |
| LDL | Load within Local Memory Window |
| LDS | Load within Shared Memory Window |
| LDSM | Load Matrix from Shared Memory with Element Size Expansion |
| STSM | Store Matrix to Shared Memory |
| ST | Store to Generic Memory |
| STG | Store to Global Memory |
| STL | Store to Local Memory |
| STS | Store to Shared Memory |
| STAS | Asynchronous Store to Distributed Shared Memory With Explicit Synchronization |
| SYNCS | Sync Unit |
| MATCH | Match Register Values Across Thread Group |
| QSPC | Query Space |
| ATOM | Atomic Operation on Generic Memory |
| ATOMS | Atomic Operation on Shared Memory |
| ATOMG | Atomic Operation on Global Memory |
| REDAS | Asynchronous Reduction on Distributed Shared Memory With Explicit Synchronization |
|
| REDG | Reduction Operation on Generic Memory |
| CCTL | Cache Control |
| CCTLL | Cache Control |
| ERRBAR | Error Barrier |
| MEMBAR | Memory Barrier |
| CCTLT | Texture Cache Control |
| **Uniform Datapath Instructions** |  |
| CREDUX | Coupled Reduction of a Vector Register into a Uniform Register |
| CS2UR | Load a Value from Constant Memory into a Uniform Register |
| LDCU | Load a Value from Constant Memory into a Uniform Register |
| R2UR | Move from Vector Register to a Uniform Register |
| REDUX | Reduction of a Vector Register into a Uniform Register |
| S2UR | Move Special Register to Uniform Register |
| UBMSK | Uniform Bitfield Mask |
| UBREV | Uniform Bit Reverse |
| UCGABAR\_ARV | CGA Barrier Synchronization |
| UCGABAR\_WAIT | CGA Barrier Synchronization |
| UCLEA | Load Effective Address for a Constant |
| UFADD | Uniform Uniform FP32 Addition |
| UF2F | Uniform Float-to-Float Conversion |
| UF2FP | Uniform FP32 Down-convert and Pack |
| UF2I | Uniform Float-to-Integer Conversion |
| UF2IP | Uniform FP32 Down-Convert to Integer and Pack |
| UFFMA | Uniform FP32 Fused Multiply-Add |
| UFLO | Uniform Find Leading One |
| UFMNMX | Uniform Floating-point Minimum / Maximum |
| UFMUL | Uniform FP32 Multiply |
| UFRND | Uniform Round to Integer |
| UFSEL | Uniform Floating-Point Select |
| UFSET | Uniform Floating-Point Compare and Set |
| UFSETP | Uniform Floating-Point Compare and Set Predicate |
| UI2F | Uniform Integer to Float conversion |
| UI2FP | Uniform Integer to FP32 Convert and Pack |
| UI2I | Uniform Saturating Integer-to-Integer Conversion |
| UI2IP | Uniform Dual Saturating Integer-to-Integer Conversion and Packing |
| UIABS | Uniform Integer Absolute Value |
| UIMNMX | Uniform Integer Minimum / Maximum |
| UIADD3 | Uniform Integer Addition |
| UIADD3.64 | Uniform Integer Addition |
| UIMAD | Uniform Integer Multiplication |
| UISETP | Uniform Integer Compare and Set Uniform Predicate |
| ULEA | Uniform Load Effective Address |
| ULEPC | Uniform Load Effective PC |
| ULOP | Uniform Logic Operation |
| ULOP3 | Uniform Logic Operation |
| ULOP32I | Uniform Logic Operation |
| UMOV | Uniform Move |
| UP2UR | Uniform Predicate to Uniform Register |
| UPLOP3 | Uniform Predicate Logic Operation |
| UPOPC | Uniform Population Count |
| UPRMT | Uniform Byte Permute |
| UPSETP | Uniform Predicate Logic Operation |
| UR2UP | Uniform Register to Uniform Predicate |
| USEL | Uniform Select |
| USETMAXREG | Release, Deallocate and Allocate Registers |
| USGXT | Uniform Sign Extend |
| USHF | Uniform Funnel Shift |
| USHL | Uniform Left Shift |
| USHR | Uniform Right Shift |
| UGETNEXTWORKID | Uniform Get Next Work ID |
| UMEMSETS | Initialize Shared Memory |
| UREDGR | Uniform Reduction on Global Memory with Release |
| USTGR | Uniform Store to Global Memory with Release |
| UVIADD | Uniform SIMD Integer Addition |
| UVIMNMX | Uniform SIMD Integer Minimum / Maximum |
| UVIRTCOUNT | Virtual Resource Management |
| VOTEU | Voting across SIMD Thread Group with Results in Uniform Destination |
| **Tensor Memory Access Instructions** |  |
| UBLKCP | Bulk Data Copy |
| UBLKPF | Bulk Data Prefetch |
| UBLKRED | Bulk Data Copy from Shared Memory with Reduction |
| UTMACCTL | TMA Cache Control |
| UTMACMDFLUSH | TMA Command Flush |
| UTMALDG | Tensor Load from Global to Shared Memory |
| UTMAPF | Tensor Prefetch |
| UTMAREDG | Tensor Store from Shared to Global Memory with Reduction |
| UTMASTG | Tensor Store from Shared to Global Memory |
| **Tensor Core Memory Instructions** |  |
| LDT | Load Matrix from Tensor Memory to Register File |
| LDTM | Load Matrix from Tensor Memory to Register File |
| STT | Store Matrix to Tensor Memory from Register File |
| STTM | Store Matrix to Tensor Memory from Register File |
| UTCATOMSWS | Perform Atomic operation on SW State Register |
| UTCBAR | Tensor Core Barrier |
| UTCCP | Asynchonous data copy from Shared Memory to Tensor Memory |
| UTCHMMA | Uniform Matrix Multiply and Accumulate |
| UTCIMMA | Uniform Matrix Multiply and Accumulate |
| UTCOMMA | Uniform Matrix Multiply and Accumulate |
| UTCQMMA | Uniform Matrix Multiply and Accumulate |
| UTCSHIFT | Shift elements in Tensor Memory |
| **Texture Instructions** |  |
| TEX | Texture Fetch |
| TLD | Texture Load |
| TLD4 | Texture Load 4 |
| TMML | Texture MipMap Level |
| TXD | Texture Fetch With Derivatives |
| TXQ | Texture Query |
| **Surface Instructions** |  |
| SUATOM | Atomic Op on Surface Memory |
| SULD | Surface Load |
| SURED | Reduction Op on Surface Memory |
| SUST | Surface Store |
| **Control Instructions** |  |
| ACQBULK | Wait for Bulk Release Status Warp State |
| ACQSHMINIT | Wait for Shared Memory Initialization Release Status Warp State |
| BMOV | Move Convergence Barrier State |
| BPT | BreakPoint/Trap |
| BRA | Relative Branch |
| BREAK | Break out of the Specified Convergence Barrier |
| BRX | Relative Branch Indirect |
| BRXU | Relative Branch with Uniform Register Based Offset |
| BSSY | Barrier Set Convergence Synchronization Point |
| BSYNC | Synchronize Threads on a Convergence Barrier |
| CALL | Call Function |
| CGAERRBAR | CGA Error Barrier |
| ELECT | Elect a Leader Thread |
| ENDCOLLECTIVE | Reset the MCOLLECTIVE mask |
| EXIT | Exit Program |
| JMP | Absolute Jump |
| JMX | Absolute Jump Indirect |
| JMXU | Absolute Jump with Uniform Register Based Offset |
| KILL | Kill Thread |
| NANOSLEEP | Suspend Execution |
| PREEXIT | Dependent Task Launch Hint |
| RET | Return From Subroutine |
| RPCMOV | PC Register Move |
| WARPSYNC | Synchronize Threads in Warp |
| YIELD | Yield Control |
| **Miscellaneous Instructions** |  |
| B2R | Move Barrier To Register |
| BAR | Barrier Synchronization |
| CS2R | Move Special Register to Register |
| DEPBAR | Dependency Barrier |
| GETLMEMBASE | Get Local Memory Base Address |
| LEPC | Load Effective PC |
| NOP | No Operation |
| PMTRIG | Performance Monitor Trigger |
| S2R | Move Special Register to Register |
| SETCTAID | Set CTA ID |
| SETLMEMBASE | Set Local Memory Base Address |
| VOTE | Vote Across SIMT Thread Group |

