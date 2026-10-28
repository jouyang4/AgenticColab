#!/usr/bin/env -S colab run --gpu T4
"""Rent a GPU, report what you got, release it - one executable file.

    chmod +x 01-hello-shebang.py && ./01-hello-shebang.py

`colab run`: fresh VM, run this, release. `--keep` leaves it up; `-s NAME`
names it for later `colab exec -s NAME`.

`env -S` splits the shebang line into separate arguments, which is what lets a
shebang carry flags. Without it the kernel passes "colab run --gpu T4" as one
argv entry and the exec fails.
"""
import os
import subprocess
import sys

print("python    :", sys.version.split()[0])
print("cpus      :", os.cpu_count())

try:
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal"):
                print("ram       :", round(int(line.split()[1]) / 1024**2, 1), "GiB")
                break
except OSError:
    pass

smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
     "--format=csv,noheader"],
    capture_output=True, text=True)
print("gpu       :", smi.stdout.strip() or "(none)")

try:
    import torch
    print("torch     :", torch.__version__, "cuda", torch.version.cuda)
    if torch.cuda.is_available():
        import time
        n = 4096
        a = torch.randn(n, n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, n, device="cuda", dtype=torch.float16)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()
        tflops = 2 * n**3 * 20 / (time.perf_counter() - t) / 1e12
        print(f"fp16 gemm : {tflops:.1f} TFLOP/s")
except ImportError:
    print("torch     : not installed")

print("JOB_COMPLETE")
