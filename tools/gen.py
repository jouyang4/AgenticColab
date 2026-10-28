#!/usr/bin/env python3
"""Generate spec/shapes.json and the generated docs. No hand edits to those files.

    python3 tools/gen.py [--data data/shapes] [--docs docs] [--spec spec]

Generated: spec/shapes.json, docs/01, docs/02, docs/04, docs/05, docs/10, docs/11.
Sources:   data/shapes/*.json, spec/{cli,errors,antipatterns,supervisor}.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

ORDER = ["cpu", "cpu-highmem", "t4", "t4-highmem", "l4", "a100", "a100-highmem",
         "g4", "v5e1", "v6e1"]

PRETTY = {
    "cpu": "(none)", "cpu-highmem": "--high-mem",
    "t4": "--gpu T4", "t4-highmem": "--gpu T4 --high-mem",
    "l4": "--gpu L4",
    "a100": "--gpu A100", "a100-highmem": "--gpu A100 --high-mem",
    "g4": "--gpu G4",
    "v5e1": "--tpu v5e1", "v6e1": "--tpu v6e1",
}

SHAPE_NOTES = [
    "The host CPU model is not fixed per shape: the same flags can allocate a different CPU on a later run, so treat the CPU column as the instance measured, not a guarantee -> [07](07-cpu-parallelism.md).",
    "`--high-mem` is a CPU upgrade too: CPU/T4 go 2 -> 8 vCPU and 12.7 -> 51 GiB. CU/hr rises 3.3x on the CPU shape (0.08 -> 0.26) but only 19% on T4 (1.07 -> 1.27). Ignored for L4, v5e1, v6e1 (single shape each).",
    "Alloc 3-5 s when capacity exists. A slow alloc (a100-highmem, 57 s) is a capacity signal; a client timeout during one can still leave a VM behind -> [03 §5](03-session-lifecycle.md).",
    "An entitlement rejection is a policy answer about your account, ~1-2 s. Not 503 (capacity), not 412 (concurrency) -> [05](05-error-taxonomy.md).",
    "Endpoint ids encode the shape: `gpu-t4-hm-...` high-RAM T4, `gpu-a100-s-...` standard A100, `tpu-v6e1-s-...` TPU.",
]

ACCEL_NOTES = [
    "tf32 is off by default and free on Ampere+. fp32 -> tf32 is multiples wherever the hardware supports it.",
    "bf16 on T4 (cc 7.5): no bf16 tensor cores, so bf16 matmul runs slower than fp32. Use fp16, ~6x faster.",
    "Pinned H2D beats pageable on every shape, ~2x on T4/L4/A100. `pin_memory=True`.",
    "Host link varies: most shapes ~12 GB/s pinned, G4 ~4.5x that. For feed-limited work this matters more than TFLOP/s.",
    "The two A100 shapes have identical compute (within 1%, same 108 SMs); the 80 GB part has more bandwidth. `--high-mem` buys capacity and bandwidth, not arithmetic.",
]

CPU_NOTES = [
    "The 2-vCPU shapes score ratio < 1.0: multi-threaded BLAS on one physical core contends. Pin `OMP_NUM_THREADS=1` and parallelise by process, or take a bigger shape -> [07](07-cpu-parallelism.md).",
]

IMAGE_NOTES = [
    "System python is 3.13. Many scientific packages cap lower, and pip then resolves an ancient version or builds from source (minutes of stall). Escape it: `pip install uv && uv venv --python 3.12 /content/venv` then `uv pip install --python /content/venv/bin/python <pkgs>`.",
    "GPU images: CUDA-13 torch. TPU images: CPU-only torch + jax.",
    "On a GPU runtime `jax.devices()` reports the GPU, not a TPU. Detect with `devices()[0].platform`.",
]

DISK_NOTES = [
    "Read figures are cache-warm, not device read throughput -- dropping caches needs root. Plan checkpointing against the write number (`conv=fdatasync`).",
    "No CPU or memory cgroup quota on any shape: `cpu.max` carries no quota, `memory.max` is `max`. Limits are the machine's real cores and RAM, so nothing stops you oversubscribing into an OOM.",
]

METHOD = [
    ["matmul", "dense square `a @ b`, n=8192, 3 warmup + 20 timed, `torch.cuda.synchronize()` around the timed region. tf32 toggled explicitly via `torch.backends.cuda.matmul.allow_tf32`, so fp32 vs tf32 is controlled, not default-dependent."],
    ["HBM", "512 MiB device-to-device `copy_`, 2 bytes per byte copied (read + write)."],
    ["H2D", "256 MiB host-to-device `copy_`, pageable and pinned measured separately. Naive single-stream copies, not tuned kernels: use the ratios, not the absolutes."],
    ["CPU GEMM", "numpy fp64 GEMM in a fresh subprocess with OMP/OPENBLAS/MKL/NUMEXPR thread counts pinned. Unpinned, BLAS uses every core and the single-thread figure is not single-threaded."],
    ["CPU scaling", "N independent processes run an identical pure-python CPU-bound loop; speedup = (N x baseline) / wall. Task size calibrated at runtime to ~10 s (at ~1 s the curve measures process spawn and plateaus near 4x on every machine). Fresh subprocesses, best-of-2 per point -- shared hosts have noisy neighbours, and forking a kernel that imported a multithreaded library is unsafe."],
    ["disk", "`dd` 1 GiB, `conv=fdatasync` for the write. Read is cache-warm and labelled as such."],
]


def esc(s):
    """A literal `|` inside a cell ends the column; markdown needs it escaped."""
    return str(s).replace("|", "\\|")


def g(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def load(res, label):
    p = os.path.join(res, f"{label}.json")
    m = os.path.join(res, f"{label}.meta.json")
    s = os.path.join(res, f"scaling-{label}.json")
    rd = json.load(open(p)) if os.path.exists(p) else None
    md = json.load(open(m)) if os.path.exists(m) else None
    sd = json.load(open(s)) if os.path.exists(s) else None
    return rd, md, sd


def f(v, suf="", dash="—"):
    return f"{v}{suf}" if v not in (None, "", "None") else dash


def mib_to_gb(s):
    if not s:
        return None
    m = re.match(r"(\d+)\s*MiB", str(s))
    return f"{int(m.group(1)) / 1024:.1f} GiB" if m else s


def dd(s):
    if not s:
        return None
    m = re.search(r",\s*([\d.]+\s*[kMG]B/s)\s*$", str(s).strip())
    return m.group(1) if m else None


def collect(res):
    rows = []
    for label in ORDER:
        d, m, sc = load(res, label)
        if d is None and m is None:
            continue
        gpus = g(d, "gpu", "gpus", default=[]) or []
        gpu0 = gpus[0] if gpus else {}
        # `colab usage` is an ACCOUNT-WIDE rate. A reading is this shape's rate
        # only if it was the sole live assignment; otherwise it is an aggregate
        # and must not be published. Older meta files predate cu_rate_clean.
        n_live = g(m, "active_assignments_while_live")
        clean = g(m, "cu_rate_clean")
        if clean is None:
            clean = (n_live == 1)
        cu_raw = g(m, "cu_per_hour")
        rows.append({
            "label": label, "flags": PRETTY.get(label, label), "ok": d is not None,
            "outcome": g(m, "outcome", default="ok" if d else "not_allocated"),
            "alloc_s": g(m, "alloc_seconds"),
            "cu": cu_raw if clean else None,
            "cu_raw": cu_raw, "cu_clean": bool(clean), "n_live": n_live,
            "sess": g(m, "session_line", default=""),
            "cpu_model": g(d, "cpu", "model"), "vcpu": g(d, "cpu", "logical_cpus"),
            "phys": g(d, "cpu", "physical_cores"), "smt": g(d, "cpu", "smt_ratio"),
            "ram": g(d, "memory", "MemTotal_GiB"), "ram_av": g(d, "memory", "MemAvailable_GiB"),
            "disk_root": g(d, "disk", "/", "total_GiB"),
            "disk_content": g(d, "disk", "/content", "total_GiB"),
            "shm": g(d, "disk", "/dev/shm", "total_GiB"),
            "gpu_name": gpu0.get("name"), "vram": gpu0.get("memory_total"),
            "driver": gpu0.get("driver"), "cc": gpu0.get("compute_cap"),
            "power": gpu0.get("power_limit"),
            "pcie": f"gen{gpu0.get('pcie_gen_max')} x{gpu0.get('pcie_width_max')}"
                    if gpu0.get("pcie_gen_max") else None,
            "sm": g(d, "torch", "sm_count"),
            "python": g(d, "python"), "kernel": g(d, "kernel"),
            "torch_v": g(d, "torch", "version"), "cuda": g(d, "torch", "cuda_version"),
            "jax": g(d, "accelerator_devices", "jax_version") or g(d, "tpu", "jax_version"),
            "tpu_kind": next((k for k in (g(d, "accelerator_devices", "device_kinds")
                                          or g(d, "tpu", "device_kinds") or [])
                              if "TPU" in str(k).upper()), None),
            "npkg": g(d, "packages", "_total_packages"),
            "cpu_max": g(d, "cgroup_limits", "cpu_max"),
            "mem_max": g(d, "cgroup_limits", "memory_max"),
            "isa": g(d, "cpu", "isa", default={}),
            "gemm1": g(d, "bench", "cpu_gemm", "fp64_gemm_2048_1thread_GFLOPs"),
            "gemmN": g(d, "bench", "cpu_gemm", "fp64_gemm_4096_allthreads_GFLOPs"),
            "fp32": g(d, "bench", "gpu", "fp32_matmul_8192_TFLOPs"),
            "tf32": g(d, "bench", "gpu", "tf32_matmul_8192_TFLOPs"),
            "fp16": g(d, "bench", "gpu", "fp16_matmul_8192_TFLOPs"),
            "bf16": g(d, "bench", "gpu", "bf16_matmul_8192_TFLOPs"),
            "hbm": g(d, "bench", "gpu", "hbm_copy_GBps"),
            "h2dp": g(d, "bench", "gpu", "h2d_pinned_GBps"),
            "h2dg": g(d, "bench", "gpu", "h2d_pageable_GBps"),
            "tpu_bf16": g(d, "bench", "tpu", "bf16_matmul_8192_TFLOPs"),
            "dd_w": g(d, "bench", "disk", "write_1GiB"),
            "dd_r": g(d, "bench", "disk", "read_1GiB_CACHE_WARM")
                    or g(d, "bench", "disk", "read_1GiB"),
            "scaling": sc,
        })
    return rows


def bullets(A, items):
    for it in items:
        A(f"- {it}")
    A("")


def code(A, lang, body, label=None):
    if label:
        A(f"**{label}**")
        A("")
    A(f"```{lang}")
    A(body)
    A("```")
    A("")


def write_shapes_json(rows, path, cli):
    out = {"measured": {"cli": cli, "account": "single paid Colab account"},
           "shapes": {}}
    for r in rows:
        e = {"flags": r["flags"], "allocated": r["ok"], "outcome": r["outcome"],
             "cu_per_hour": r["cu"], "alloc_seconds": r["alloc_s"]}
        if r["ok"] and not r["cu_clean"]:
            e["cu_per_hour_unreliable"] = {
                "reading": r["cu_raw"],
                "active_assignments_while_live": r["n_live"],
                "why": "account-wide reading taken with more than one live "
                       "assignment; not this shape's rate",
            }
        if r["ok"]:
            e.update({
                "vcpu": r["vcpu"], "physical_cores": r["phys"], "smt_ratio": r["smt"],
                "cpu_model": r["cpu_model"], "ram_gib": r["ram"],
                "accelerator": r["gpu_name"] or r["tpu_kind"],
                "vram": mib_to_gb(r["vram"]), "compute_cap": r["cc"], "sms": r["sm"],
                "python": r["python"], "torch": r["torch_v"], "cuda": r["cuda"],
                "jax": r["jax"], "driver": r["driver"],
                "disk_content_gib": r["disk_content"], "shm_gib": r["shm"],
                "cgroup": {"cpu_max": r["cpu_max"], "memory_max": r["mem_max"]},
                "bench": {
                    "fp32_tflops": r["fp32"], "tf32_tflops": r["tf32"],
                    "fp16_tflops": r["fp16"], "bf16_tflops": r["bf16"] or r["tpu_bf16"],
                    "hbm_gbps": r["hbm"], "h2d_pinned_gbps": r["h2dp"],
                    "h2d_pageable_gbps": r["h2dg"],
                    "gemm_1thread_gflops": r["gemm1"], "gemm_all_gflops": r["gemmN"],
                    "disk_write": dd(r["dd_w"]), "disk_read_cache_warm": dd(r["dd_r"]),
                },
            })
            if r["scaling"]:
                s = r["scaling"]
                e["cpu_scaling"] = {
                    "peak_speedup": s.get("peak_speedup"),
                    "peak_at_workers": s.get("peak_at_workers"),
                    "knee_85pct_at_workers": s.get("knee_85pct_efficiency_at_workers"),
                    "pool_spawn_s": s.get("mp_pool_spawn_s"),
                    "points": s.get("points"),
                }
        out["shapes"][r["label"]] = e
    json.dump(out, open(path, "w"), indent=1)
    return out


def doc_shapes(rows, cli):
    L, A = [], None
    A = L.append
    ok = [r for r in rows if r["ok"]]
    A("# Machine shapes")
    A("")
    A(f"GENERATED by `tools/gen.py` from `data/shapes/`. Machine-readable: "
      f"[`spec/shapes.json`](../spec/shapes.json).")
    A(f"Measured on `google-colab-cli {cli}`. Entitlements and images vary by subscription "
      f"tier and change over time. Re-measure: `scripts/run-probe.sh <label> [flags]`.")
    A("")
    A("## Shapes")
    A("")
    A("| `colab new` flags | vCPU | phys | RAM | accelerator | VRAM | CU/hr | alloc |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r["ok"]:
            A(f"| `{r['flags']}` | — | — | — | **rejected: no quota or entitlement** | — | — | — |")
            continue
        acc = r["gpu_name"] or r["tpu_kind"] or "—"
        A(f"| `{r['flags']}` | {f(r['vcpu'])} | {f(r['phys'])} | {f(r['ram'],' GiB')} | "
          f"{acc} | {f(mib_to_gb(r['vram']))} | {f(r['cu'])} | {f(r['alloc_s'],' s')} |")
    A("")
    notes = list(SHAPE_NOTES)
    unclean = [r for r in ok if not r["cu_clean"]]
    if unclean:
        notes.append(
            "CU/hr is **—** for "
            + ", ".join(f"`{r['flags']}`" for r in unclean)
            + ": `colab usage` reports an account-wide rate, and those readings were "
            "taken with more than one assignment live, so they are aggregates rather "
            "than per-shape rates. Re-probe with nothing else assigned "
            "(`scripts/run-probe.sh` now refuses to start otherwise). The raw readings "
            "are kept under `cu_per_hour_unreliable` in "
            "[`spec/shapes.json`](../spec/shapes.json).")
    bullets(A, notes)
    A("## Accelerator throughput")
    A("")
    A("TFLOP/s unless stated. Method: [11](11-benchmarks.md).")
    A("")
    A("| shape | fp32 | tf32 | fp16 | bf16 | HBM GB/s | H2D pinned | H2D pageable | SMs |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        if not r["gpu_name"]:
            continue
        A(f"| `{r['flags']}` | {f(r['fp32'])} | {f(r['tf32'])} | {f(r['fp16'])} | "
          f"{f(r['bf16'])} | {f(r['hbm'])} | {f(r['h2dp'])} | {f(r['h2dg'])} | {f(r['sm'])} |")
    A("")
    tpu_rows = [r for r in ok if r["tpu_bf16"]]
    if tpu_rows:
        A("TPU, same matmul through `jax.jit`:")
        A("")
        A("| shape | bf16 TFLOP/s | devices | jax |")
        A("|---|---|---|---|")
        for r in tpu_rows:
            A(f"| `{r['flags']}` | {f(r['tpu_bf16'])} | {f(r['tpu_kind'])} | {f(r['jax'])} |")
        A("")
    bullets(A, ACCEL_NOTES)
    A("## CPU throughput")
    A("")
    A("| shape | vCPU | CPU model | 1 thread GFLOP/s | all threads | ratio |")
    A("|---|---|---|---|---|---|")
    for r in ok:
        ratio = round(r["gemmN"] / r["gemm1"], 1) if (r["gemm1"] and r["gemmN"]) else None
        A(f"| `{r['flags']}` | {f(r['vcpu'])} | {f(r['cpu_model'])} | {f(r['gemm1'])} | "
          f"{f(r['gemmN'])} | {f(ratio,'x')} |")
    A("")
    bullets(A, CPU_NOTES)
    A("## Software image")
    A("")
    A("| shape | python | torch | CUDA | jax | driver | cc | pip pkgs |")
    A("|---|---|---|---|---|---|---|---|")
    for r in ok:
        A(f"| `{r['flags']}` | {f(r['python'])} | {f(r['torch_v'])} | {f(r['cuda'])} | "
          f"{f(r['jax'])} | {f(r['driver'])} | {f(r['cc'])} | {f(r['npkg'])} |")
    A("")
    bullets(A, IMAGE_NOTES)
    A("## cgroups and disk")
    A("")
    A("| shape | cpu.max | memory.max | `/` GiB | `/content` GiB | `/dev/shm` GiB | dd write | dd read (cache-warm) |")
    A("|---|---|---|---|---|---|---|---|")
    for r in ok:
        A(f"| `{r['flags']}` | `{f(r['cpu_max'])}` | `{f(r['mem_max'])}` | "
          f"{f(r['disk_root'])} | {f(r['disk_content'])} | {f(r['shm'])} | "
          f"{f(dd(r['dd_w']))} | {f(dd(r['dd_r']))} |")
    A("")
    bullets(A, DISK_NOTES)
    A("## Per shape")
    A("")
    for r in rows:
        A(f"### `{r['flags']}`")
        A("")
        if not r["ok"]:
            A("Rejected in ~1 s: *Backend rejected accelerator. You may not have quota or "
              "entitlement for this accelerator on your account.*")
            A("")
            continue
        A(f"- session: `{r['sess']}`")
        A(f"- CPU: {r['cpu_model']}, {r['vcpu']} logical / {r['phys']} physical (SMT {r['smt']}x)")
        isa = [k for k, v in (r["isa"] or {}).items() if v]
        if isa:
            A(f"- ISA: {', '.join(isa)}")
        A(f"- RAM: {r['ram']} GiB total, {r['ram_av']} GiB available at probe")
        if r["gpu_name"]:
            A(f"- GPU: {r['gpu_name']}, {mib_to_gb(r['vram'])}, cc {r['cc']}, {r['sm']} SMs, "
              f"{r['power']}, PCIe {r['pcie']}, driver {r['driver']}")
        if r["tpu_bf16"]:
            A(f"- TPU: {r['tpu_kind']}, {r['tpu_bf16']} TFLOP/s bf16, jax {r['jax']}")
        A(f"- disk: `/` {r['disk_root']} GiB, `/content` {r['disk_content']} GiB")
        A(f"- kernel {r['kernel']}, python {r['python']}, {r['npkg']} pip packages")
        A(f"- {r['alloc_s']} s to allocate, {r['cu']} CU/hr")
        A("")
    return "\n".join(L)


def doc_bench(rows, cli):
    L = []
    A = L.append
    ok = [r for r in rows if r["ok"]]
    sc_rows = [r for r in ok if r.get("scaling")]
    A("# Benchmarks")
    A("")
    A(f"GENERATED by `tools/gen.py`. Measured on `google-colab-cli {cli}`. "
      f"Raw: [`data/shapes/`](../data/shapes/). Machine-readable: "
      f"[`spec/shapes.json`](../spec/shapes.json).")
    A("")
    A("Micro-benchmarks: they characterise the machine, not your workload.")
    A("")
    A("## Method")
    A("")
    A("Payload [`agenticcolab/probe.py`](../agenticcolab/probe.py), driver "
      "[`scripts/run-probe.sh`](../scripts/run-probe.sh) (allocate -> probe detached -> "
      "download JSON -> release).")
    A("")
    A("| | |")
    A("|---|---|")
    for k, v in METHOD:
        A(f"| {k} | {v} |")
    A("")
    A("## CPU scaling")
    A("")
    if not sc_rows:
        A("*(none in this dataset: run `AGC_CPU_SCALING=1 scripts/run-probe.sh <label> [flags]`)*")
        A("")
    else:
        A("| shape | vCPU | phys | peak | at workers | 85%-eff knee | pool spawn |")
        A("|---|---|---|---|---|---|---|")
        for r in sc_rows:
            s = r["scaling"]
            A(f"| `{r['flags']}` | {f(s.get('logical_cpus'))} | {f(r['phys'])} | "
              f"{f(s.get('peak_speedup'),'x')} | {f(s.get('peak_at_workers'))} | "
              f"{f(s.get('knee_85pct_efficiency_at_workers'))} | "
              f"{f(s.get('mp_pool_spawn_s'),' s')} |")
        A("")
        bullets(A, [
            "Size pools at the PHYSICAL core count: peak speedup tracks physical cores to within ~10% on every shape measured.",
            "SMT gain beyond that is vendor-split: ~15-18% on AMD EPYC, ~0-5% on Intel Xeon. L4 is slower at 12 workers than at 6.",
            "The vendor applies to the instance measured, not to the shape: the same flags can allocate a different host CPU on a later run -> [07](07-cpu-parallelism.md).",
            "Past the logical CPU count there is no gain: G4 moves from 24.06x at 48 workers to 24.19x at 96 while wall-clock doubles.",
        ])
        for r in sc_rows:
            s = r["scaling"]
            A(f"### `{r['flags']}` — {s.get('logical_cpus')} vCPU / {r['phys']} physical")
            A("")
            A(f"Task calibrated to {s.get('calibrated_single_task_s')} s "
              f"({s.get('task_iters'):,} iters); baseline {s.get('baseline_1worker_s')} s; "
              f"best of {s.get('trials_per_point')}.")
            A("")
            A("| workers | wall s | speedup | efficiency |")
            A("|---|---|---|---|")
            for k in sorted(s.get("points", {}), key=lambda x: int(x)):
                p = s["points"][k]
                A(f"| {k} | {p['wall_s']} | {p['speedup_vs_1']}x | {p['efficiency_pct']}% |")
            A("")
    A("## Compute-unit rates")
    A("")
    A("From `colab usage` while each assignment was live, and only where it was the "
      "**sole** live assignment: the reading is account-wide, so a rate measured "
      "alongside another assignment is an aggregate. Colab publishes no such table.")
    A("")
    A("| shape | CU/hr | vs CPU | alloc |")
    A("|---|---|---|---|")
    base = next((r["cu"] for r in ok if r["label"] == "cpu" and r["cu"]), None)
    for r in ok:
        rel = f"{r['cu']/base:.0f}x" if (base and r["cu"]) else "—"
        A(f"| `{r['flags']}` | {f(r['cu'])} | {rel} | {f(r['alloc_s'],' s')} |")
    A("")
    clean_cu = [r["cu"] for r in ok if r["cu"]]
    spread = (f"{max(clean_cu)/min(clean_cu):.0f}x" if len(clean_cu) > 1 else "unknown")
    bullets(A, [
        f"Spread between the cheapest and dearest measured shape: **{spread}**. Budget on "
        f"cost per unit of WORK, not per hour: 3x the rate finishing 5x faster is cheaper "
        f"-> [09](09-cost-and-quota.md).",
        "Put analysis and post-processing on a cheap shape rather than leaving an expensive one idle.",
    ])
    A("## Reproduce")
    A("")
    code(A, "bash",
         "for s in cpu 'cpu-highmem --high-mem' 't4 --gpu T4' 'l4 --gpu L4' \\\n"
         "         'a100 --gpu A100' 'g4 --gpu G4' 'v6e1 --tpu v6e1'; do\n"
         "  scripts/run-probe.sh $s\n"
         "done\n"
         "AGC_CPU_SCALING=1 scripts/run-probe.sh v6e1 --tpu v6e1   # adds the scaling curve\n"
         "python3 tools/gen.py                                     # regenerate docs 01 + 11")
    A("Full 11-shape sweep: ~30 min, under 1 CU, because each machine is held for minutes.")
    A("")
    return "\n".join(L)


def doc_cli(c):
    L = []
    A = L.append
    m = c["measured"]
    A("# CLI reference")
    A("")
    A(f"GENERATED by `tools/gen.py` from [`spec/cli.json`](../spec/cli.json). "
      f"Read off `--help` and exercised live on `google-colab-cli {m['cli']}`, "
      f"with {m['compared_against']} kept alongside.")
    A("")
    code(A, "bash", c["install"][0] + "\n"
         + "\n".join(f"# or: {x}" for x in c["install"][1:]) + "\ncolab version")
    A("## Facts")
    A("")
    bullets(A, c["facts"])
    A("## Global flags")
    A("")
    A("Before the subcommand, always.")
    A("")
    A("| flag | meaning |")
    A("|---|---|")
    for x in c["global_flags"]:
        A(f"| `{esc(x['flag'])}` | {esc(x['meaning'])} |")
    A("")
    A("## Commands")
    A("")
    for group, items in c["commands"].items():
        A(f"### {group}")
        A("")
        A("| command | notes |")
        A("|---|---|")
        for x in items:
            A(f"| `{esc(x['cmd'])}` | {esc(x['note'])} |")
        A("")
    A("### `colab new` flags")
    A("")
    A("| flag | values |")
    A("|---|---|")
    for x in c["new_flags"]:
        A(f"| `{esc(x['flag'])}` | {esc(x['values'])} |")
    A("")
    A("### `colab exec` flags")
    A("")
    A("| flag | notes |")
    A("|---|---|")
    for x in c["exec_flags"]:
        A(f"| `{esc(x['flag'])}` | {esc(x['note'])} |")
    A("")
    A(f"### `colab run` flags")
    A("")
    A(c["run_flags"])
    A("")
    A("### Requestable enums")
    A("")
    A("| enum | values |")
    A("|---|---|")
    for k, v in c["enums"].items():
        A(f"| `{k}` | {', '.join(f'`{x}`' for x in v)} |")
    A("")
    A("")
    A("What you get is an entitlement question -> [01](01-machine-shapes.md), "
      "[05](05-error-taxonomy.md).")
    A("")
    A("## Version differences")
    A("")
    A("| | 0.6.0 | 0.7.5 | |")
    A("|---|---|---|---|")
    for x in c["version_diff"]:
        def mark(v):
            return "✓" if v is True else ("✗" if v is False else f"`{v}`")
        A(f"| {esc(x['feature'])} | {mark(x['v060'])} | {mark(x['v075'])} | "
          f"{esc(x.get('note',''))} |")
    A("")
    A("## Internals")
    A("")
    code(A, "python", c["internals"]["import"])
    A("| object | notes |")
    A("|---|---|")
    for o in c["internals"]["objects"]:
        A(f"| `{esc(o['name'])}` | {esc(o['note'])} |")
    A("")
    A(f"**Enum trap.** {c['internals']['trap']} -> "
      f"[03](03-session-lifecycle.md#sessionsjson-fails-closed)")
    A("")
    A("## Examples")
    A("")
    for ex in c["examples"]:
        code(A, ex["lang"], ex["code"], label=ex["title"])
    return "\n".join(L)


def doc_supervisor(s):
    L = []
    A = L.append
    A("# Supervisor patterns")
    A("")
    A("GENERATED by `tools/gen.py` from "
      "[`spec/supervisor.json`](../spec/supervisor.json).")
    A("")
    A(s["problem"])
    A("")
    A("## Checklist")
    A("")
    for c in s["checklist"]:
        A(f"- [ ] {c}")
    A("")
    for sec in s["sections"]:
        A(f"## {sec['id']}. {sec['title']}")
        A("")
        if sec.get("points"):
            bullets(A, sec["points"])
        if sec.get("note"):
            A(f"> {sec['note']}")
            A("")
        for cb in sec.get("code", []):
            code(A, cb["lang"], cb["body"], label=cb.get("label"))
    return "\n".join(L)


def doc_errors(e):
    L = []
    A = L.append
    m = e["measured"]
    A("# Error taxonomy")
    A("")
    A(f"GENERATED by `tools/gen.py` from [`spec/errors.json`](../spec/errors.json). "
      f"`google-colab-cli {m['cli']}`.")
    A("")
    A("## Classes")
    A("")
    A("| class | signal | latency | retry | action |")
    A("|---|---|---|---|---|")
    for c in e["classes"]:
        A(f"| **{esc(c['class'])}** | {esc(c['signal'])} | {esc(c['latency'])} | "
          f"{esc(c['retry'])} | {esc(c['action'])} |")
    A("")
    A("## Decision table")
    A("")
    A("| symptom | what it is | correct response |")
    A("|---|---|---|")
    for x in e["errors"]:
        ref = f" -> [{x['ref']}]({x['ref']})" if x.get("ref") else ""
        A(f"| {esc(x['symptom'])} | {esc(x['means'])} | {esc(x['do'])}{ref} |")
    A("")
    A("## Allocation triage")
    A("")
    code(A, e["allocation_triage"]["lang"], e["allocation_triage"]["code"])
    A("## Rules")
    A("")
    bullets(A, e["rules"])
    return "\n".join(L)


def doc_antipatterns(a):
    L = []
    A = L.append
    A("# Antipatterns")
    A("")
    A("GENERATED by `tools/gen.py` from "
      "[`spec/antipatterns.json`](../spec/antipatterns.json).")
    A("")
    A(a["note"])
    A("")
    for x in a["items"]:
        A(f"## {x['id']}. {x['name']}")
        A("")
        rows = [("symptom", x.get("symptom")), ("reality", x.get("reality")),
                ("cost", x.get("cost")), ("fix", x.get("fix"))]
        for k, v in rows:
            if v:
                A(f"- **{k}:** {v}")
        if x.get("ref"):
            A(f"- **ref:** [{x['ref']}]({x['ref']})")
        A("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shapes")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--spec", default="spec")
    ap.add_argument("--cli", default="0.7.5")
    args = ap.parse_args()

    rows = collect(args.data)
    write_shapes_json(rows, os.path.join(args.spec, "shapes.json"), args.cli)

    sj = lambda n: json.load(open(os.path.join(args.spec, n)))
    out = {
        "01-machine-shapes.md": doc_shapes(rows, args.cli),
        "02-cli-reference.md": doc_cli(sj("cli.json")),
        "04-supervisor-patterns.md": doc_supervisor(sj("supervisor.json")),
        "05-error-taxonomy.md": doc_errors(sj("errors.json")),
        "10-antipatterns.md": doc_antipatterns(sj("antipatterns.json")),
        "11-benchmarks.md": doc_bench(rows, args.cli),
    }
    for name, body in out.items():
        open(os.path.join(args.docs, name), "w").write(body.rstrip() + "\n")

    ok = sum(1 for r in rows if r["ok"])
    sc = sum(1 for r in rows if r.get("scaling"))
    print(json.dumps({"docs": sorted(out), "spec": ["shapes.json"],
                      "shapes_ok": ok, "shapes_rejected": len(rows) - ok,
                      "scaling_curves": sc,
                      "rates_withheld_as_aggregates":
                          [r["label"] for r in rows if r["ok"] and not r["cu_clean"]]},
                     indent=1))


if __name__ == "__main__":
    main()
