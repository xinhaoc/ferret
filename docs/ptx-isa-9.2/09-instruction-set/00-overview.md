# 9. [Instruction Set](#instruction-set)[](#instruction-set "Permalink to this headline")

## 9.1. [Format and Semantics of Instruction Descriptions](#format-and-semantics-of-instruction-descriptions)[](#format-and-semantics-of-instruction-descriptions "Permalink to this headline")

This section describes each PTX instruction. In addition to the name and the format of the
instruction, the semantics are described, followed by some examples that attempt to show several
possible instantiations of the instruction.

## 9.2. [PTX Instructions](#ptx-instructions)[](#ptx-instructions "Permalink to this headline")

PTX instructions generally have from zero to four operands, plus an optional guard predicate
appearing after an `@` symbol to the left of the `opcode`:

* `@p   opcode;`
* `@p   opcode a;`
* `@p   opcode d, a;`
* `@p   opcode d, a, b;`
* `@p   opcode d, a, b, c;`

For instructions that create a result value, the `d` operand is the destination operand, while
`a`, `b`, and `c` are source operands.

The `setp` instruction writes two destination registers. We use a `|` symbol to separate
multiple destination registers.

```
setp.lt.s32  p|q, a, b;  // p = (a < b); q = !(a < b);
```

For some instructions the destination operand is optional. A *bit bucket* operand denoted with an
underscore (`_`) may be used in place of a destination register.

## 9.3. [Predicated Execution](#predicated-execution)[](#predicated-execution "Permalink to this headline")

In PTX, predicate registers are virtual and have `.pred` as the type specifier. So, predicate
registers can be declared as

```
.reg .pred p, q, r;
```

All instructions have an optional *guard predicate* which controls conditional execution of the
instruction. The syntax to specify conditional execution is to prefix an instruction with `@{!}p`,
where `p` is a predicate variable, optionally negated. Instructions without a guard predicate are
executed unconditionally.

Predicates are most commonly set as the result of a comparison performed by the `setp`
instruction.

As an example, consider the high-level code

```
if (i < n)
    j = j + 1;
```

This can be written in PTX as

```
      setp.lt.s32  p, i, n;    // p = (i < n)
@p    add.s32      j, j, 1;    // if i < n, add 1 to j
```

To get a conditional branch or conditional function call, use a predicate to control the execution
of the branch or call instructions. To implement the above example as a true conditional branch, the
following PTX instruction sequence might be used:

```
      setp.lt.s32  p, i, n;    // compare i to n
@!p   bra  L1;                 // if False, branch over
      add.s32      j, j, 1;
L1:     ...
```

### 9.3.1. [Comparisons](#comparisons)[](#comparisons "Permalink to this headline")

#### 9.3.1.1. [Integer and Bit-Size Comparisons](#integer-and-bit-size-comparisons)[](#integer-and-bit-size-comparisons "Permalink to this headline")

The signed integer comparisons are the traditional `eq` (equal), `ne` (not-equal), `lt`
(less-than), `le` (less-than-or-equal), `gt` (greater-than), and `ge`
(greater-than-or-equal). The unsigned comparisons are `eq`, `ne`, `lo` (lower), `ls`
(lower-or-same), `hi` (higher), and `hs` (higher-or-same). The bit-size comparisons are `eq`
and `ne`; ordering comparisons are not defined for bit-size types.

[Table 22](#integer-and-bit-size-comparisons-operators-for-signed-integer-unsigned-integer-and-bit-size-types)
shows the operators for signed integer, unsigned integer, and bit-size types.

Table 22 Operators for Signed Integer, Unsigned Integer, and Bit-Size Types[](#integer-and-bit-size-comparisons-operators-for-signed-integer-unsigned-integer-and-bit-size-types "Permalink to this table")


| Meaning | Signed Operator | Unsigned Operator | Bit-Size Operator |
| --- | --- | --- | --- |
| `a == b` | `eq` | `eq` | `eq` |
| `a != b` | `ne` | `ne` | `ne` |
| `a < b` | `lt` | `lo` | n/a |
| `a <= b` | `le` | `ls` | n/a |
| `a > b` | `gt` | `hi` | n/a |
| `a >= b` | `ge` | `hs` | n/a |

#### 9.3.1.2. [Floating Point Comparisons](#floating-point-comparisons)[](#floating-point-comparisons "Permalink to this headline")

The ordered floating-point comparisons are `eq`, `ne`, `lt`, `le`, `gt`, and `ge`. If
either operand is `NaN`, the result is
`False`. [Table 23](#floating-point-comparisons-floating-point-operators) lists the floating-point
comparison operators.

Table 23 Floating-Point Comparison Operators[](#floating-point-comparisons-floating-point-operators "Permalink to this table")


| Meaning | Floating-Point Operator |
| --- | --- |
| `a == b && !isNaN(a) && !isNaN(b)` | `eq` |
| `a != b && !isNaN(a) && !isNaN(b)` | `ne` |
| `a < b && !isNaN(a) && !isNaN(b)` | `lt` |
| `a <= b && !isNaN(a) && !isNaN(b)` | `le` |
| `a > b && !isNaN(a) && !isNaN(b)` | `gt` |
| `a >= b && !isNaN(a) && !isNaN(b)` | `ge` |

To aid comparison operations in the presence of `NaN` values, unordered floating-point comparisons
are provided: `equ`, `neu`, `ltu`, `leu`, `gtu`, and `geu`. If both operands are numeric
values (not `NaN`), then the comparison has the same result as its ordered counterpart. If either
operand is `NaN`, then the result of the comparison is `True`.

[Table 24](#floating-point-comparisons-floating-point-operators-nan) lists the floating-point
comparison operators accepting `NaN` values.

Table 24 Floating-Point Comparison Operators Accepting NaN[](#floating-point-comparisons-floating-point-operators-nan "Permalink to this table")


| Meaning | Floating-Point Operator |
| --- | --- |
| `a == b || isNaN(a) || isNaN(b)` | `equ` |
| `a != b || isNaN(a) || isNaN(b)` | `neu` |
| `a < b || isNaN(a) || isNaN(b)` | `ltu` |
| `a <= b || isNaN(a) || isNaN(b)` | `leu` |
| `a > b || isNaN(a) || isNaN(b)` | `gtu` |
| `a >= b || isNaN(a) || isNaN(b)` | `geu` |

To test for `NaN` values, two operators `num` (`numeric`) and `nan` (`isNaN`) are
provided. `num` returns `True` if both operands are numeric values (not `NaN`), and `nan`
returns `True` if either operand is
`NaN`. [Table 25](#floating-point-comparisons-floating-point-operators-testing-nan) lists the
floating-point comparison operators testing for `NaN` values.

Table 25 Floating-Point Comparison Operators Testing for NaN[](#floating-point-comparisons-floating-point-operators-testing-nan "Permalink to this table")


| Meaning | Floating-Point Operator |
| --- | --- |
| `!isNaN(a) && !isNaN(b)` | `num` |
| `isNaN(a) || isNaN(b)` | `nan` |

### 9.3.2. [Manipulating Predicates](#manipulating-predicates)[](#manipulating-predicates "Permalink to this headline")

Predicate values may be computed and manipulated using the following instructions: `and`, `or`,
`xor`, `not`, and `mov`.

There is no direct conversion between predicates and integer values, and no direct way to load or
store predicate register values. However, `setp` can be used to generate a predicate from an
integer, and the predicate-based select (`selp`) instruction can be used to generate an integer
value based on the value of a predicate; for example:

```
selp.u32 %r1,1,0,%p;    // convert predicate to 32-bit value
```

## 9.4. [Type Information for Instructions and Operands](#type-information-for-instructions-and-operands)[](#type-information-for-instructions-and-operands "Permalink to this headline")

Typed instructions must have a type-size modifier. For example, the `add` instruction requires
type and size information to properly perform the addition operation (signed, unsigned, float,
different sizes), and this information must be specified as a suffix to the opcode.

Example

```
.reg .u16 d, a, b;

add.u16 d, a, b;    // perform a 16-bit unsigned add
```

Some instructions require multiple type-size modifiers, most notably the data conversion instruction
`cvt`. It requires separate type-size modifiers for the result and source, and these are placed in
the same order as the operands. For example:

```
.reg .u16 a;
.reg .f32 d;

cvt.f32.u16 d, a;   // convert 16-bit unsigned to 32-bit float
```

In general, an operand’s type must agree with the corresponding instruction-type modifier. The rules
for operand and instruction type conformance are as follows:

* Bit-size types agree with any type of the same size.
* Signed and unsigned integer types agree provided they have the same size, and integer operands are
  silently cast to the instruction type if needed. For example, an unsigned integer operand used in
  a signed integer instruction will be treated as a signed integer by the instruction.
* Floating-point types agree only if they have the same size; i.e., they must match exactly.

[Table 26](#type-information-for-instructions-and-operands-type-checking-rules) summarizes these type
checking rules.

Table 26 Type Checking Rules[](#type-information-for-instructions-and-operands-type-checking-rules "Permalink to this table")


|  | | **Operand Type** | | | |
| --- | --- | --- | --- | --- | --- |
|  | | **.bX** | **.sX** | **.uX** | **.fX** |
| **Instruction Type** | **.bX** | okay | okay | okay | okay |
| **.sX** | okay | okay | okay | invalid |
| **.uX** | okay | okay | okay | invalid |
| **.fX** | okay | invalid | invalid | okay |

Note

Some operands have their type and size defined independently from the instruction type-size. For
example, the shift amount operand for left and right shift instructions always has type `.u32`,
while the remaining operands have their type and size determined by the instruction type.

Example

```
// 64-bit arithmetic right shift; shift amount 'b' is .u32
    shr.s64 d,a,b;
```

### 9.4.1. [Operand Size Exceeding Instruction-Type Size](#operand-size-exceeding-instruction-type-size)[](#operand-size-exceeding-instruction-type-size "Permalink to this headline")

For convenience, `ld`, `st`, and `cvt` instructions permit source and destination data
operands to be wider than the instruction-type size, so that narrow values may be loaded, stored,
and converted using regular-width registers. For example, 8-bit or 16-bit values may be held
directly in 32-bit or 64-bit registers when being loaded, stored, or converted to other types and
sizes. The operand type checking rules are relaxed for bit-size and integer (signed and unsigned)
instruction types; floating-point instruction types still require that the operand type-size matches
exactly, unless the operand is of bit-size type.

When a source operand has a size that exceeds the instruction-type size, the source data is
truncated (chopped) to the appropriate number of bits specified by the instruction type-size.

[Table 27](#operand-size-exceeding-instruction-type-size-relaxed-type-checking-rules-source-operands)
summarizes the relaxed type-checking rules for source operands. Note that some combinations may
still be invalid for a particular instruction; for example, the `cvt` instruction does not support
`.bX` instruction types, so those rows are invalid for `cvt`.

Table 27 Relaxed Type-checking Rules for Source Operands[](#operand-size-exceeding-instruction-type-size-relaxed-type-checking-rules-source-operands "Permalink to this table")


|  | | **Source Operand Type** | | | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **b8** | **b16** | **b32** | **b64** | **b128** | **s8** | **s16** | **s32** | **s64** | **u8** | **u16** | **u32** | **u64** | **f16** | **f32** | **f64** |
| **Instruction Type** | **b8** | – | chop | chop | chop | chop | – | chop | chop | chop | – | chop | chop | chop | chop | chop | chop |
| **b16** | inv | – | chop | chop | chop | inv | – | chop | chop | inv | – | chop | chop | – | chop | chop |
| **b32** | inv | inv | – | chop | chop | inv | inv | – | chop | inv | inv | – | chop | inv | – | chop |
| **b64** | inv | inv | inv | – | chop | inv | inv | inv | – | inv | inv | inv | – | inv | inv | – |
| **b128** | inv | inv | inv | inv | – | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv |
| **s8** | – | chop | chop | chop | chop | – | chop | chop | chop | – | chop | chop | chop | inv | inv | inv |
| **s16** | inv | – | chop | chop | chop | inv | – | chop | chop | inv | – | chop | chop | inv | inv | inv |
| **s32** | inv | inv | – | chop | chop | inv | inv | – | chop | inv | inv | – | chop | inv | inv | inv |
| **s64** | inv | inv | inv | – | chop | inv | inv | inv | – | inv | inv | inv | – | inv | inv | inv |
| **u8** | – | chop | chop | chop | chop | – | chop | chop | chop | – | chop | chop | chop | inv | inv | inv |
| **u16** | inv | – | chop | chop | chop | inv | – | chop | chop | inv | – | chop | chop | inv | inv | inv |
| **u32** | inv | inv | – | chop | chop | inv | inv | – | chop | inv | inv | – | chop | inv | inv | inv |
| **u64** | inv | inv | inv | – | chop | inv | inv | inv | – | inv | inv | inv | – | inv | inv | inv |
| **f16** | inv | – | chop | chop | chop | inv | inv | inv | inv | inv | inv | inv | inv | – | inv | inv |
| **f32** | inv | inv | – | chop | chop | inv | inv | inv | inv | inv | inv | inv | inv | inv | – | inv |
| **f64** | inv | inv | inv | – | chop | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | – |
| **Notes** | | chop = keep only low bits that fit; “–” = allowed, but no conversion needed;  inv = invalid, parse error.   1. Source register size must be of equal or greater size than the instruction-type size. 2. Bit-size source registers may be used with any appropriately-sized instruction type. The data are    truncated (“chopped”) to the instruction-type size and interpreted according to the instruction    type. 3. Integer source registers may be used with any appropriately-sized bit-size or integer instruction    type. The data are truncated to the instruction-type size and interpreted according to the    instruction type. 4. Floating-point source registers can only be used with bit-size or floating-point instruction types.    When used with a narrower bit-size instruction type, the data are truncated. When used with a    floating-point instruction type, the size must match exactly. | | | | | | | | | | | | | | | |

When a destination operand has a size that exceeds the instruction-type size, the destination data
is zero- or sign-extended to the size of the destination register. If the corresponding instruction
type is signed integer, the data is sign-extended; otherwise, the data is zero-extended.

[Table 28](#operand-size-exceeding-instruction-type-size-relaxed-type-checking-rules-destination-operands)
summarizes the relaxed type-checking rules for destination operands.

Table 28 Relaxed Type-checking Rules for Destination Operands[](#operand-size-exceeding-instruction-type-size-relaxed-type-checking-rules-destination-operands "Permalink to this table")


|  | | **Destination Operand Type** | | | | | | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **b8** | **b16** | **b32** | **b64** | **b128** | **s8** | **s16** | **s32** | **s64** | **u8** | **u16** | **u32** | **u64** | **f16** | **f32** | **f64** |
| **Instruction Type** | **b8** | – | zext | zext | zext | zext | – | zext | zext | zext | – | zext | zext | zext | zext | zext | zext |
| **b16** | inv | – | zext | zext | zext | inv | – | zext | zext | inv | – | zext | zext | – | zext | zext |
| **b32** | inv | inv | – | zext | zext | inv | inv | – | zext | inv | inv | – | zext | inv | – | zext |
| **b64** | inv | inv | inv | – | zext | inv | inv | inv | – | inv | inv | inv | – | inv | inv | – |
| **b128** | inv | inv | inv | inv | – | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv |
| **s8** | – | sext | sext | sext | sext | – | sext | sext | sext | – | sext | sext | sext | inv | inv | inv |
| **s16** | inv | – | sext | sext | sext | inv | – | sext | sext | inv | – | sext | sext | inv | inv | inv |
| **s32** | inv | inv | – | sext | sext | inv | inv | – | sext | inv | inv | – | sext | inv | inv | inv |
| **s64** | inv | inv | inv | – | sext | inv | inv | inv | – | inv | inv | inv | – | inv | inv | inv |
| **u8** | – | zext | zext | zext | zext | – | zext | zext | zext | – | zext | zext | zext | inv | inv | inv |
| **u16** | inv | – | zext | zext | zext | inv | – | zext | zext | inv | – | zext | zext | inv | inv | inv |
| **u32** | inv | inv | – | zext | zext | inv | inv | – | zext | inv | inv | – | zext | inv | inv | inv |
| **u64** | inv | inv | inv | – | zext | inv | inv | inv | – | inv | inv | inv | – | inv | inv | inv |
| **f16** | inv | – | zext | zext | zext | inv | inv | inv | inv | inv | inv | inv | inv | – | inv | inv |
| **f32** | inv | inv | – | zext | zext | inv | inv | inv | inv | inv | inv | inv | inv | inv | – | inv |
| **f64** | inv | inv | inv | – | zext | inv | inv | inv | inv | inv | inv | inv | inv | inv | inv | – |
| **Notes** | | sext = sign-extend; zext = zero-extend; “–” = allowed, but no conversion needed;  inv = invalid, parse error.   1. Destination register size must be of equal or greater size than the instruction-type size. 2. Bit-size destination registers may be used with any appropriately-sized instruction type. The data    are sign-extended to the destination register width for signed integer instruction types, and are    zero-extended to the destination register width otherwise. 3. Integer destination registers may be used with any appropriately-sized bit-size or integer    instruction type. The data are sign-extended to the destination register width for signed integer    instruction types, and are zero-extended to the destination register width for bit-size an d    unsigned integer instruction types. 4. Floating-point destination registers can only be used with bit-size or floating-point instruction    types. When used with a narrower bit-size instruction type, the data are zero-extended. When used    with a floating-point instruction type, the size must match exactly. | | | | | | | | | | | | | | | |

## 9.5. [Divergence of Threads in Control Constructs](#divergence-of-threads-in-control-constructs)[](#divergence-of-threads-in-control-constructs "Permalink to this headline")

Threads in a CTA execute together, at least in appearance, until they come to a conditional control
construct such as a conditional branch, conditional function call, or conditional return. If threads
execute down different control flow paths, the threads are called *divergent*. If all of the threads
act in unison and follow a single control flow path, the threads are called *uniform*. Both
situations occur often in programs.

A CTA with divergent threads may have lower performance than a CTA with uniformly executing threads,
so it is important to have divergent threads re-converge as soon as possible. All control constructs
are assumed to be divergent points unless the control-flow instruction is marked as uniform, using
the `.uni` suffix. For divergent control flow, the optimizing code generator automatically
determines points of re-convergence. Therefore, a compiler or code author targeting PTX can ignore
the issue of divergent threads, but has the opportunity to improve performance by marking branch
points as uniform when the compiler or author can guarantee that the branch point is non-divergent.

## 9.6. [Semantics](#semantics)[](#semantics "Permalink to this headline")

The goal of the semantic description of an instruction is to describe the results in all cases in as
simple language as possible. The semantics are described using C, until C is not expressive enough.

### 9.6.1. [Machine-Specific Semantics of 16-bit Code](#machine-specific-semantics-of-16-bit-code)[](#machine-specific-semantics-of-16-bit-code "Permalink to this headline")

A PTX program may execute on a GPU with either a 16-bit or a 32-bit data path. When executing on a
32-bit data path, 16-bit registers in PTX are mapped to 32-bit physical registers, and 16-bit
computations are *promoted* to 32-bit computations. This can lead to computational differences
between code run on a 16-bit machine versus the same code run on a 32-bit machine, since the
promoted computation may have bits in the high-order half-word of registers that are not present in
16-bit physical registers. These extra precision bits can become visible at the application level,
for example, by a right-shift instruction.

At the PTX language level, one solution would be to define semantics for 16-bit code that is
consistent with execution on a 16-bit data path. This approach introduces a performance penalty for
16-bit code executing on a 32-bit data path, since the translated code would require many additional
masking instructions to suppress extra precision bits in the high-order half-word of 32-bit
registers.

Rather than introduce a performance penalty for 16-bit code running on 32-bit GPUs, the semantics of
16-bit instructions in PTX is machine-specific. A compiler or programmer may chose to enforce
portable, machine-independent 16-bit semantics by adding explicit conversions to 16-bit values at
appropriate points in the program to guarantee portability of the code. However, for many
performance-critical applications, this is not desirable, and for many applications the difference
in execution is preferable to limiting performance.

