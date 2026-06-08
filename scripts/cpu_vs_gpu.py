import time

import torch


torch.set_float32_matmul_precision('high')

size = 10000

print("Generating matrices...")
cpu_A = torch.randn(size, size, dtype=torch.float32)
cpu_B = torch.randn(size, size, dtype=torch.float32)

gpu_A = cpu_A.to('mps').half()
gpu_B = cpu_B.to('mps').half()

# OPTIMIZATION 1: Pre-allocate the output memory space
gpu_out = torch.empty(size, size, dtype=torch.half, device='mps')

# OPTIMIZATION 2: Define a compiled function for the GPU math
@torch.compile
def optimized_gpu_matmul(A, B, out_space) -> :
    return torch.matmul(A, B, out=out_space)

# --- Warmup ---
_ = optimized_gpu_matmul(gpu_A, gpu_B, gpu_out)
torch.mps.synchronize()

# --- TIME CPU (3 loops) ---
print("Running CPU Benchmark...")
start_time = time.time()
for _ in range(3):
    cpu_res = torch.matmul(cpu_A, cpu_B)
cpu_time = time.time() - start_time

# --- TIME GPU (3 loops with pre-allocated memory) ---
print("Running Optimized GPU Benchmark...")
start_time = time.time()
for _ in range(3):
    optimized_gpu_matmul(gpu_A, gpu_B, gpu_out)
torch.mps.synchronize()
gpu_time = time.time() - start_time

print("\n--- OPTIMIZED RESULTS ---")
print(f"CPU Time (Float32): {cpu_time:.6f} seconds")
print(f"GPU Time (Float16): {gpu_time:.6f} seconds")
print(f"Real Speedup: {cpu_time / gpu_time:.2f}x")