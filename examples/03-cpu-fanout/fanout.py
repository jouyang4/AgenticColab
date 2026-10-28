#!/usr/bin/env python3
"""Saturate a Colab host's CPUs with an idempotent, restartable worker pool.

    AGC_ACCEL=v6e1 NPROC=24 python3 fanout.py

For a TPU runtime's HOST (24-44 vCPU) rather than a "CPU" runtime (2 vCPU) -
docs/07-cpu-parallelism.md. You never touch the TPU.

Survives a reclaimed VM because:
  * one output file per item  -> restart skips completed work for free
  * a subprocess per item     -> one bad item cannot kill the pool
  * streamed progress         -> warm control channel + something to poll
  * a final JOB_COMPLETE line -> what the poller greps for

Pool sizing: set NPROC to the PHYSICAL core count for CPU-bound work. Wider
adds little (efficiency falls past physical cores) and only helps on I/O.
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

OUT = os.environ.get("AGC_OUT_DIR", "/content/work/out")
NITEMS = int(os.environ.get("NITEMS", "200"))
NPROC = int(os.environ.get("NPROC", str(os.cpu_count() or 1)))
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "item.py")


def run_item(item_id):
    dest = os.path.join(OUT, f"item_{item_id:05d}.json")
    if os.path.exists(dest):
        return ("skip", item_id, 0.0)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, WORKER, str(item_id), dest],
                           capture_output=True, text=True, timeout=1800)
        status = "ok" if p.returncode == 0 else f"fail:{p.returncode}"
        if p.returncode != 0:
            # keep the reason next to the failure instead of losing it
            with open(dest + ".err", "w") as fh:
                fh.write(p.stderr[-4000:])
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception as exc:  # noqa: BLE001
        status = f"error:{type(exc).__name__}"
    return (status, item_id, time.perf_counter() - t0)


def main():
    os.makedirs(OUT, exist_ok=True)
    items = list(range(NITEMS))
    already = sum(1 for i in items
                  if os.path.exists(os.path.join(OUT, f"item_{i:05d}.json")))
    print(f"host: {os.cpu_count()} logical CPUs | pool: {NPROC} | "
          f"items: {NITEMS} ({already} already done)", flush=True)

    t0 = time.perf_counter()
    counts = {}
    done = 0
    # imap_unordered so a slow item doesn't stall reporting
    with mp.Pool(NPROC) as pool:
        for status, item_id, secs in pool.imap_unordered(run_item, items):
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            done += 1
            if done % max(1, NITEMS // 20) == 0 or status.startswith(("fail", "err", "timeout")):
                el = time.perf_counter() - t0
                rate = done / el if el else 0
                print(f"  {done}/{NITEMS} ({rate:.1f} items/s) last={status} "
                      f"id={item_id} {secs:.1f}s", flush=True)

    el = time.perf_counter() - t0
    summary = {"items": NITEMS, "pool": NPROC, "logical_cpus": os.cpu_count(),
               "elapsed_s": round(el, 1),
               "items_per_s": round(NITEMS / el, 2) if el else None,
               "counts": counts,
               "accel": os.environ.get("AGC_ACCEL", "unknown")}
    with open(os.path.join(OUT, "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary), flush=True)

    # gate on real state, not on falling off the end of the loop
    produced = sum(1 for i in items
                   if os.path.exists(os.path.join(OUT, f"item_{i:05d}.json")))
    if produced == NITEMS:
        print("JOB_COMPLETE")
        return 0
    print(f"INCOMPLETE: {produced}/{NITEMS}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
