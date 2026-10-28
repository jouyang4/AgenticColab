#!/usr/bin/env python3
"""Colab runtime inventory + micro-benchmarks. Runs ON the VM, emits JSON.

    AGC_ACCEL=a100 AGC_BENCH=1 python3 probe.py

Inventory half is stdlib-only, so it works on every runtime image (CPU, GPU,
TPU) with no install. Benchmark half uses numpy/torch/jax only if present and
degrades to null rather than failing.

  AGC_ACCEL         label recorded in the output (what you asked for)
  AGC_BENCH         "1" to run micro-benchmarks (+~60-120 s)
  AGC_CPU_SCALING   "1" to run the CPU scaling sweep (+~3-6 min)
  AGC_TARGET_S      seconds per scaling task (default 10)
  AGC_TRIALS        trials per scaling point, min taken (default 2)
  AGC_OUT           output path (default /content/agc_probe.json)

Scaling task size is CALIBRATED at runtime to ~AGC_TARGET_S: with sub-second
tasks process-spawn dominates and every machine appears to plateau near 4x.
"""
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time

# ----------------------------------------------------------------- helpers


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout.strip() or p.stderr.strip())[:4000]
    except Exception as exc:  # noqa: BLE001
        return f"ERR:{exc}"


def read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------- inventory


def cpuinfo():
    txt = read("/proc/cpuinfo") or ""
    model, sockets, cores = None, set(), set()
    for block in txt.split("\n\n"):
        phys = cid = None
        for line in block.splitlines():
            if line.startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            if line.startswith("physical id"):
                phys = line.split(":", 1)[1].strip()
                sockets.add(phys)
            if line.startswith("core id"):
                cid = line.split(":", 1)[1].strip()
        if phys is not None and cid is not None:
            cores.add((phys, cid))
    flags = ""
    m = re.search(r"^flags\s*:(.*)$", txt, re.M)
    if m:
        flags = f" {m.group(1)} "
    logical = os.cpu_count()
    physical = len(cores) or None
    return {
        "model": model,
        "logical_cpus": logical,
        "physical_cores": physical,
        "sockets": len(sockets) or None,
        "smt_ratio": round(logical / physical, 2) if (logical and physical) else None,
        "affinity_cpus": (len(os.sched_getaffinity(0))
                          if hasattr(os, "sched_getaffinity") else None),
        "isa": {f: (f" {f} " in flags) for f in
                ("avx2", "avx512f", "avx512bf16", "amx_bf16", "amx_int8")},
    }


def meminfo():
    txt = read("/proc/meminfo") or ""
    out = {}
    for key in ("MemTotal", "MemAvailable", "SwapTotal"):
        m = re.search(rf"^{key}:\s+(\d+)", txt, re.M)
        if m:
            out[key + "_GiB"] = round(int(m.group(1)) / 1024 / 1024, 2)
    return out


def cgroups():
    """cgroup v2 is how Colab would cap you; 'max' means no quota."""
    return {name: read(path) for name, path in (
        ("cpu_max", "/sys/fs/cgroup/cpu.max"),
        ("memory_max", "/sys/fs/cgroup/memory.max"),
        ("memory_high", "/sys/fs/cgroup/memory.high"),
        ("pids_max", "/sys/fs/cgroup/pids.max"))}


def gpu():
    if not shutil.which("nvidia-smi"):
        return None
    q = ("name,memory.total,driver_version,compute_cap,pcie.link.gen.max,"
         "pcie.link.width.max,clocks.max.sm,power.limit,persistence_mode")
    rows = []
    for line in sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 9:
            rows.append(dict(zip(
                ["name", "memory_total", "driver", "compute_cap", "pcie_gen_max",
                 "pcie_width_max", "sm_clock_max_mhz", "power_limit",
                 "persistence_mode"], parts)))
    return {"count": len(rows), "gpus": rows}


def accelerator_devices():
    """jax's view. NOTE on a GPU runtime this reports the GPU, not a TPU."""
    out = {}
    try:
        import jax
        out["jax_version"] = jax.__version__
        devs = jax.devices()
        out["device_count"] = len(devs)
        out["device_kinds"] = sorted({d.device_kind for d in devs})
        out["platform"] = devs[0].platform if devs else None
    except Exception as exc:  # noqa: BLE001
        out["jax"] = f"unavailable: {type(exc).__name__}"
    out["accel_devices"] = sh("ls -1 /dev/accel* 2>/dev/null || echo none")
    return out


def torchinfo():
    try:
        import torch
        info = {"version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda}
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(device_name=torch.cuda.get_device_name(0),
                        vram_GiB=round(p.total_memory / 1024**3, 2),
                        sm_count=p.multi_processor_count,
                        capability=f"{p.major}.{p.minor}")
        return info
    except Exception as exc:  # noqa: BLE001
        return {"torch": f"unavailable: {type(exc).__name__}"}


def disks():
    out = {}
    for mount in ("/", "/content", "/tmp", "/dev/shm"):
        if os.path.isdir(mount):
            try:
                st = shutil.disk_usage(mount)
                out[mount] = {"total_GiB": round(st.total / 1024**3, 1),
                              "free_GiB": round(st.free / 1024**3, 1)}
            except Exception:  # noqa: BLE001
                pass
    return out


def pkgs():
    raw = sh(f"{sys.executable} -m pip list --format=freeze 2>/dev/null", timeout=240)
    have = {}
    for line in raw.splitlines():
        if "==" in line:
            n, v = line.split("==", 1)
            have[n.strip().lower()] = v.strip()
    keys = ("torch", "torchvision", "jax", "jaxlib", "libtpu", "numpy", "scipy",
            "tensorflow", "pandas", "transformers", "uv", "flax")
    out = {k: have[k] for k in keys if k in have}
    out["_total_packages"] = len(have)
    return out


# ---------------------------------------------------------------- benchmarks


def bench_cpu_gemm():
    """Single- vs all-thread fp64 GEMM, GFLOP/s.

    Thread counts are pinned via env in a fresh subprocess, or BLAS silently
    uses every core and the '1 thread' number is not one.
    """
    try:
        import numpy  # noqa: F401
    except Exception:  # noqa: BLE001
        return None

    def gflops(n, reps, threads):
        env = os.environ.copy()
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(threads)
        code = (
            "import numpy as np,time\n"
            f"n={n}\n"
            "a=np.random.rand(n,n);b=np.random.rand(n,n)\n"
            "np.dot(a,b)\n"
            "t=time.perf_counter()\n"
            f"for _ in range({reps}): np.dot(a,b)\n"
            "d=time.perf_counter()-t\n"
            f"print(2*n**3*{reps}/d/1e9)\n"
        )
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, env=env, timeout=900)
            return round(float(p.stdout.strip().splitlines()[-1]), 1)
        except Exception:  # noqa: BLE001
            return None

    ncpu = os.cpu_count() or 1
    return {"fp64_gemm_2048_1thread_GFLOPs": gflops(2048, 3, 1),
            "fp64_gemm_4096_allthreads_GFLOPs": gflops(4096, 3, ncpu),
            "threads_used_for_all": ncpu}


_SPIN = "/tmp/_agc_spin.py"
_SPIN_SRC = """import sys, time
n = int(sys.argv[1])
t = time.perf_counter()
x = 0.0
i = 0
while i < n:
    x += i * 0.5
    i += 1
print(time.perf_counter() - t, x)
"""


def _spin_n(nworkers, iters):
    with open(_SPIN, "w") as fh:
        fh.write(_SPIN_SRC)
    t0 = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, _SPIN, str(iters)],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
             for _ in range(nworkers)]
    for p in procs:
        p.wait()
    return time.perf_counter() - t0


def bench_cpu_scaling():
    """How many workers does this box actually give you?

    Fresh subprocesses, task size calibrated to multi-second, best-of-N trials
    per point (a shared cloud host has noisy neighbours and the minimum is the
    closest thing to the machine's real capability).
    """
    target_s = float(os.environ.get("AGC_TARGET_S", "10"))
    trials = int(os.environ.get("AGC_TRIALS", "2"))
    ncpu = os.cpu_count() or 1

    iters, single = 2_000_000, None
    for _ in range(12):
        single = _spin_n(1, iters)
        work = max(single - 0.05, 1e-3)
        if work >= target_s * 0.8:
            break
        iters = int(iters * min(target_s / work, 8))

    points = sorted({1, 2, 4, 8, 12, max(1, ncpu // 4), max(1, ncpu // 2),
                     ncpu, int(ncpu * 1.5), ncpu * 2})
    points = [p for p in points if 1 <= p <= ncpu * 2]

    base = min(_spin_n(1, iters) for _ in range(max(trials, 2)))
    results = {}
    for n in points:
        walls = [_spin_n(n, iters) for _ in range(trials)]
        wall = min(walls)
        speedup = (n * base) / wall
        results[n] = {"wall_s": round(wall, 2),
                      "walls_s": [round(w, 2) for w in walls],
                      "speedup_vs_1": round(speedup, 2),
                      "efficiency_pct": round(100 * speedup / n, 1)}
        print(f"  {n:>4} workers: wall {wall:7.2f}s  speedup {speedup:6.2f}x  "
              f"eff {100 * speedup / n:5.1f}%", flush=True)

    best = max(results.items(), key=lambda kv: kv[1]["speedup_vs_1"])
    knee = max((n for n, r in results.items() if r["efficiency_pct"] >= 85), default=1)
    try:
        t = time.perf_counter()
        with mp.Pool(ncpu):
            pass
        pool_spawn = round(time.perf_counter() - t, 3)
    except Exception:  # noqa: BLE001
        pool_spawn = None

    return {"logical_cpus": ncpu, "task_iters": iters,
            "calibrated_single_task_s": round(single or 0, 2),
            "baseline_1worker_s": round(base, 2), "trials_per_point": trials,
            "points": results, "peak_speedup": best[1]["speedup_vs_1"],
            "peak_at_workers": best[0],
            "knee_85pct_efficiency_at_workers": knee,
            "mp_pool_spawn_s": pool_spawn}


def bench_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:  # noqa: BLE001
        return None
    import torch

    out = {}
    dev = torch.device("cuda")

    def matmul_tflops(dtype, n, tf32=False, reps=20):
        try:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn(n, n, device=dev, dtype=dtype)
            for _ in range(3):
                a @ b
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                a @ b
            torch.cuda.synchronize()
            d = time.perf_counter() - t
            del a, b
            torch.cuda.empty_cache()
            return round(2 * n**3 * reps / d / 1e12, 2)
        except Exception as exc:  # noqa: BLE001
            return f"ERR:{type(exc).__name__}"

    out["fp32_matmul_8192_TFLOPs"] = matmul_tflops(torch.float32, 8192, tf32=False)
    out["tf32_matmul_8192_TFLOPs"] = matmul_tflops(torch.float32, 8192, tf32=True)
    out["fp16_matmul_8192_TFLOPs"] = matmul_tflops(torch.float16, 8192)
    out["bf16_matmul_8192_TFLOPs"] = matmul_tflops(torch.bfloat16, 8192)

    try:  # HBM bandwidth
        n = 1 << 27
        a = torch.empty(n, device=dev, dtype=torch.float32).fill_(1.0)
        b = torch.empty(n, device=dev, dtype=torch.float32)
        torch.cuda.synchronize()
        t = time.perf_counter()
        reps = 20
        for _ in range(reps):
            b.copy_(a)
        torch.cuda.synchronize()
        out["hbm_copy_GBps"] = round(
            2 * a.numel() * 4 * reps / (time.perf_counter() - t) / 1e9, 1)
        del a, b
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        out["hbm_copy_GBps"] = f"ERR:{type(exc).__name__}"

    try:  # host<->device, pinned vs pageable: matters for data-loader design
        sz = 1 << 26
        for pin in (False, True):
            h = torch.empty(sz, dtype=torch.float32, pin_memory=pin)
            d_ = torch.empty(sz, dtype=torch.float32, device=dev)
            d_.copy_(h)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                d_.copy_(h)
            torch.cuda.synchronize()
            out[f"h2d_{'pinned' if pin else 'pageable'}_GBps"] = round(
                sz * 4 * 10 / (time.perf_counter() - t) / 1e9, 1)
            del h, d_
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        out["h2d"] = f"ERR:{type(exc).__name__}"

    try:
        free, total = torch.cuda.mem_get_info()
        out["vram_free_GiB"] = round(free / 1024**3, 2)
        out["vram_total_GiB"] = round(total / 1024**3, 2)
    except Exception:  # noqa: BLE001
        pass
    return out


def bench_tpu():
    try:
        import jax
        if jax.devices()[0].platform != "tpu":
            return None
    except Exception:  # noqa: BLE001
        return None
    import jax
    import jax.numpy as jnp

    out = {}
    try:
        n = 8192
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (n, n), dtype=jnp.bfloat16)
        b = jax.random.normal(key, (n, n), dtype=jnp.bfloat16)
        f = jax.jit(lambda x, y: x @ y)
        f(a, b).block_until_ready()
        t = time.perf_counter()
        reps = 20
        for _ in range(reps):
            r = f(a, b)
        r.block_until_ready()
        out["bf16_matmul_8192_TFLOPs"] = round(
            2 * n**3 * reps / (time.perf_counter() - t) / 1e12, 2)
    except Exception as exc:  # noqa: BLE001
        out["bf16_matmul_8192_TFLOPs"] = f"ERR:{type(exc).__name__}"
    return out


def bench_disk():
    """Sequential write on /content. The READ figure is cache-warm: you cannot
    drop caches without root on a Colab runtime, so do not read it as device
    read throughput."""
    out = {}
    try:
        out["write_1GiB"] = sh("dd if=/dev/zero of=/content/_agc_io.bin bs=1M "
                               "count=1024 conv=fdatasync 2>&1 | tail -1", timeout=600)
        out["read_1GiB_CACHE_WARM"] = sh("dd if=/content/_agc_io.bin of=/dev/null "
                                         "bs=1M 2>&1 | tail -1", timeout=600)
        sh("rm -f /content/_agc_io.bin")
    except Exception as exc:  # noqa: BLE001
        out["err"] = str(exc)
    return out


# ---------------------------------------------------------------------- main


_SECRET_HINTS = ("token", "secret", "key", "password", "passwd", "cred",
                 "auth", "session", "cookie", "signature")


def env_of_interest():
    """Colab-relevant env vars, with anything secret-shaped redacted.

    The output is meant to be publishable, and a Colab runtime carries vars
    such as COLAB_JUPYTER_TOKEN. Redact by name rather than by observed value:
    a var that is empty on one image may be populated on another.
    """
    out = {}
    for k, v in os.environ.items():
        if not k.startswith(("COLAB", "TPU_", "CUDA", "NVIDIA")):
            continue
        if any(h in k.lower() for h in _SECRET_HINTS):
            out[k] = f"<redacted: {len(v)} chars>" if v else "<redacted: empty>"
        else:
            out[k] = v
    return out


def collect():
    data = {
        "probe_version": 4,
        "probed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_accelerator": os.environ.get("AGC_ACCEL", "unknown"),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2").strip('"'),
        "python": sys.version.split()[0],
        "cpu": cpuinfo(),
        "memory": meminfo(),
        "cgroup_limits": cgroups(),
        "disk": disks(),
        "gpu": gpu(),
        "accelerator_devices": accelerator_devices(),
        "torch": torchinfo(),
        "packages": pkgs(),
        "uptime_s": float((read("/proc/uptime") or "0").split()[0]),
        "env_of_interest": env_of_interest(),
    }
    if os.environ.get("AGC_BENCH") == "1":
        data["bench"] = {"cpu_gemm": bench_cpu_gemm(), "gpu": bench_gpu(),
                         "tpu": bench_tpu(), "disk": bench_disk()}
    if os.environ.get("AGC_CPU_SCALING") == "1":
        data.setdefault("bench", {})["cpu_scaling"] = bench_cpu_scaling()
    return data


def main():
    data = collect()
    out = json.dumps(data, indent=2, sort_keys=True, default=str)
    dest = os.environ.get("AGC_OUT", "/content/agc_probe.json")
    try:
        with open(dest, "w") as fh:
            fh.write(out)
    except Exception:  # noqa: BLE001
        pass
    # markers a poller can grep for
    print("AGC_JSON_BEGIN")
    print(out)
    print("AGC_JSON_END")


if __name__ == "__main__":
    main()
