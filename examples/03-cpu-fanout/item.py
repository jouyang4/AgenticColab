#!/usr/bin/env python3
"""One work item for fanout.py. Writes its result atomically, then exits.

Replace the body with real work. Preserve two things:
  * temp path + os.replace, so a kill mid-write cannot leave a truncated file
    that a restart would mistake for completed work
  * single-threaded: the parent pool owns the parallelism. With multithreaded
    BLAS, N workers each spawn N threads and the box thrashes.
"""
import json
import os
import sys
import time

# single-threaded: the pool provides the parallelism
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def main():
    item_id = int(sys.argv[1])
    dest = sys.argv[2]

    t0 = time.perf_counter()
    # ---- stand-in for real work
    acc = 0.0
    for i in range(2_000_000):
        acc += (i % 97) * 0.5
    # ----

    result = {"item": item_id, "acc": acc,
              "seconds": round(time.perf_counter() - t0, 3),
              "pid": os.getpid(),
              "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    tmp = f"{dest}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(result, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
