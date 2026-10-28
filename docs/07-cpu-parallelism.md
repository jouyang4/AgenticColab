# CPU parallelism

> **If your bottleneck is CPU cores, don't allocate a "CPU" runtime. Allocate a TPU and use
> its host.**

The CPU runtime is the smallest shape Colab offers (2 vCPU). A TPU runtime's *host* has an
order of magnitude more cores and RAM, allocates just as fast, and draws from a different
pool than GPU sessions. The TPU itself goes unused.

Measured numbers: [01](01-machine-shapes.md), [11](11-benchmarks.md),
[`spec/shapes.json`](../spec/shapes.json).

## 1. Core count is a concurrency ceiling, not a throughput multiplier

Pure-CPU work, fresh subprocesses, ~10 s calibrated tasks, best-of-2 per point:

| shape | vCPU | phys | at phys | at logical | peak | SMT gain | CPU of the measured instance |
|---|---|---|---|---|---|---|---|
| `--high-mem` | 8 | 4 | 3.39× (85%) | 3.55× (44%) | 3.6× | +5% | Intel Xeon @ 2.20GHz |
| `--gpu L4` | 12 | 6 | **5.53× (92%)** | 5.15× (43%) | 5.5× | **none** | Intel Xeon @ 2.20GHz |
| `--tpu v5e1` | 24 | 12 | 11.0× (92%) | 13.0× (54%) | 13.0× | +18% | AMD EPYC 7B13 |
| `--tpu v6e1` | 44 | 22 | 18.5× (84%) | 21.7× (49%) | 23.2× | +17% | AMD EPYC 9B14 |
| `--gpu G4` | 48 | 24 | 21.0× (88%) | 24.1× (50%) | 24.2× | +15% | AMD EPYC 9B45 |

- **Peak speedup ≈ physical core count** (ratio 0.90–1.08 on every shape). The vCPU number
  is not what you get.
- **SMT is vendor-split.** AMD EPYC: physical → logical buys 15–18%. Intel Xeon: nothing —
  L4 is *slower* at 12 workers (5.15×) than at 6 (5.53×).
- **The host CPU model is not fixed per shape.** `--high-mem` was an Intel Xeon @ 2.20GHz
  when the curve above was measured and an AMD EPYC 7B12 on a later allocation, with
  single-thread GEMM moving 27.3 → 40.8 GFLOP/s. Vendor-dependent conclusions apply to the
  instance in hand, so read `/proc/cpuinfo` instead of inferring the CPU from the flag.
- **Past the logical CPU count, nothing.** G4 moves from 24.06× at 48 workers to 24.19× at 96
  while wall-clock doubles. The processes do run (`mp.Pool(48)` spawns in 0.025 s) — they queue.
- **Cost:** G4 (24 cores, 24.2×) and v6e-1 (22 cores, 23.2×) deliver the same CPU
  throughput, at 8.90 vs 4.08 CU/hr. The TPU host is **~2.2× cheaper for the same result**.
  The big GPU shape is only worth its rate if its GPU is in use.

### Sizing

| situation | pool size |
|---|---|
| CPU-bound — the default | **physical cores.** 85–95% efficiency, best throughput per CU, least memory pressure |
| CPU-bound, AMD shape, wall-clock > cost | logical CPUs: ~15–18% more at ~50% efficiency |
| CPU-bound, Intel shape | physical cores, full stop. Wider gains nothing or loses |
| I/O-bound / latency-hiding | oversubscribe freely, 2× logical or more |
| unknown | start at physical cores and measure |

```python
import os
txt = open("/proc/cpuinfo").read()
cores = {(b.split("physical id")[1][:4], b.split("core id")[1][:4])
         for b in txt.split("\n\n") if "core id" in b and "physical id" in b}
print("logical", os.cpu_count(), "physical", len(cores))
```

The "85%-efficiency knee" in [11](11-benchmarks.md) is approximate: individual points vary a
few percent on a shared host, so it can land either side of the physical core count.

## 2. Benchmark it correctly

1. **Sub-second tasks measure process spawn.** Fork cost dominates and the curve plateaus
   near 4× regardless of machine. Use multi-second tasks.
2. **Don't fork from a kernel that imported a multithreaded library.** Forking after
   `import jax` warns about deadlock for good reason. Fresh subprocess.
3. **Pin BLAS threads**, or your "1 thread" number isn't one:

```python
env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
```

`agenticcolab/probe.py` implements the GEMM benchmark and the scaling sweep with these
precautions; `scripts/run-probe.sh` drives it.

## 3. There is no CPU cgroup quota

On every shape measured `/sys/fs/cgroup/cpu.max` is `max <period>` and `memory.max` is
`max`. The limits are the machine's real cores and RAM — you can use the whole box, and
nothing stops you oversubscribing into swap-death or OOM. Size pools against measured RAM,
especially for workers that each load a model.

## 4. Fan-out

```python
import multiprocessing as mp, os, subprocess, sys

JOBS = [...]
NPROC = int(os.environ.get("NPROC", os.cpu_count()))   # tune per §1

def run(item):
    out = f"/content/out/{item['id']}.json"
    if os.path.exists(out):                            # idempotent
        return ("skip", item["id"])
    r = subprocess.run([sys.executable, "worker.py", item["id"]],
                       capture_output=True, text=True, timeout=1800)
    return ("ok" if r.returncode == 0 else "fail", item["id"])

if __name__ == "__main__":
    os.makedirs("/content/out", exist_ok=True)
    with mp.Pool(NPROC) as pool:
        for status, jid in pool.imap_unordered(run, JOBS):
            print(status, jid, flush=True)              # streamed = warm websocket
    print("JOB_COMPLETE")
```

- one output file per item → idempotent and restartable, which is what makes a reclaimed VM
  survivable → [04](04-supervisor-patterns.md);
- a subprocess per item → one bad item can't take down the pool;
- streamed progress → keeps the control channel warm and gives the poller something to read;
- a final completion marker → what the poller greps for.

Run it detached and poll. See `examples/03-cpu-fanout/`.

## 5. When the CPU runtime is still right

A long-lived cheap orchestration box; work that must not touch an accelerator pool;
single-threaded work. For anything parallel the TPU host wins on cores, RAM and
cost-per-core.

## 6. GPU hosts have cores too

Large GPU shapes come with substantial hosts. On a shape already held for accelerator work,
the idle cores are free preprocessing capacity, and the largest GPU shape may also be the
largest CPU box the account can allocate. Which host is biggest depends on entitlements —
read the table.
