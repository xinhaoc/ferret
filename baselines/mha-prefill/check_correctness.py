"""Correctness checker for MHA prefill kernel.
Runs the compiled kernel binary, captures output, compares vs CPU fp32 reference.

Usage (from ferret root):
    python3 baselines/mha-prefill/check_correctness.py workspace/kernel

The kernel binary must accept --check flag and print output values.
If no --check support, this script generates random inputs, computes
CPU reference, and reports what the correct output should be for s=0.
"""
import torch
import sys
import struct

S, H, D = 64, 32, 128
sm_scale = 1.0 / (D ** 0.5)

# Generate deterministic inputs (seed 42, same as kernel)
torch.manual_seed(42)
Q = torch.randn(S, H, D, dtype=torch.float32)
K = torch.randn(S, H, D, dtype=torch.float32)
V = torch.randn(S, H, D, dtype=torch.float32)

# CPU fp32 causal attention
O = torch.zeros(S, H, D)
for s in range(S):
    for h in range(H):
        scores = (Q[s, h, :] @ K[:s+1, h, :].T) * sm_scale
        p = torch.softmax(scores, dim=-1)
        O[s, h, :] = p @ V[:s+1, h, :]

print(f"=== CPU fp32 Reference (S={S}, H={H}, D={D}) ===")
print(f"O[0,0,:5] = {O[0,0,:5].tolist()} (should equal V[0,0,:5] = {V[0,0,:5].tolist()})")
print(f"O[0,0,:5] == V[0,0,:5]: {torch.allclose(O[0,0,:], V[0,0,:], atol=1e-6)}")
print(f"O[1,0,:5] = {O[1,0,:5].tolist()}")
print(f"O[63,0,:5] = {O[63,0,:5].tolist()}")
print()

# Save reference as binary files for kernel to load
Q.to(torch.bfloat16).contiguous().view(-1).to(torch.int16).numpy().tofile("/tmp/mha_check_q.bin")
K.to(torch.bfloat16).contiguous().view(-1).to(torch.int16).numpy().tofile("/tmp/mha_check_k.bin")
V.to(torch.bfloat16).contiguous().view(-1).to(torch.int16).numpy().tofile("/tmp/mha_check_v.bin")
O.to(torch.bfloat16).contiguous().view(-1).to(torch.int16).numpy().tofile("/tmp/mha_check_o_ref.bin")
O.contiguous().numpy().tofile("/tmp/mha_check_o_ref_f32.bin")

print("Saved to /tmp/mha_check_{q,k,v,o_ref,o_ref_f32}.bin")
print("Kernel should load these, compute output, save to /tmp/mha_check_o_kernel.bin")
print()
print("Quick check: if your kernel outputs O[0,0,:] and it doesn't match V[0,0,:],")
print("the QK or PV MMA or softmax or TMEM layout is fundamentally broken.")
print()

# If kernel output exists, compare
import os
if os.path.exists("/tmp/mha_check_o_kernel.bin"):
    raw = open("/tmp/mha_check_o_kernel.bin", "rb").read()
    if len(raw) == S * H * D * 2:
        kern = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float().view(S, H, D)
        ref = O
        cos = torch.nn.functional.cosine_similarity(kern.reshape(1,-1), ref.reshape(1,-1)).item()
        diff = (kern - ref).abs()
        print(f"Kernel vs CPU: cosine={cos:.6f} max_abs={diff.max():.6f} mean_abs={diff.mean():.6f}")
        print(f"O[0,0,:5] kernel={kern[0,0,:5].tolist()}")
        print(f"O[0,0,:5] ref   ={ref[0,0,:5].tolist()}")
        print("PASS" if cos > 0.99 else "FAIL")
    else:
        print(f"Wrong file size: {len(raw)} (expected {S*H*D*2})")
else:
    print("No kernel output found at /tmp/mha_check_o_kernel.bin")
    print("Add to kernel: save output to this path, then rerun this script.")
