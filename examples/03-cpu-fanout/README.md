# 03 — saturating a TPU host's CPUs

CPU runtime: 2 vCPU. A TPU runtime's *host*: 24 (v5e-1) or 44 (v6e-1), same allocation
speed, different pool from GPU sessions. The TPU itself goes unused.

```bash
colab new -s fanout --tpu v6e1
colab upload -s fanout examples/03-cpu-fanout/fanout.py /content/work/fanout.py
colab upload -s fanout examples/03-cpu-fanout/item.py   /content/work/item.py
echo 'import subprocess;print(subprocess.run(["bash","-lc","cd /content/work && NPROC=22 python3 fanout.py"],capture_output=True,text=True).stdout[-2000:])' \
  | colab exec -s fanout --timeout 1800
colab stop -s fanout

# or under the supervisor, with a run.sh wrapping fanout.py
scripts/colab-job.sh myfanout examples/03-cpu-fanout -- --tpu v6e1
```

## Pool sizing

Do not set the pool to 2× vCPU and expect 2× throughput
([full curves](../../docs/11-benchmarks.md)):

| | v5e-1 (24 vCPU / 12 cores) | v6e-1 (44 vCPU / 22 cores) |
|---|---|---|
| at physical cores | 11.0× (92% eff.) | 18.5× (84%) |
| at logical CPUs | 13.0× (54%) | 21.7× (49%) |
| at 2× logical | 13.0× (27%) | 23.2× (26%) |

- `NPROC` = physical cores → best throughput per compute unit;
- `NPROC` = logical CPUs → ~17% more, if wall-clock beats efficiency;
- beyond that you are paying for queueing.

`fanout.py` defaults `NPROC` to `os.cpu_count()` (logical). Set it explicitly.

## Why it is shaped this way

- **one output file per item** → idempotent; restart skips completed work, which is what
  makes a reclaimed VM survivable;
- **a subprocess per item** → one bad item cannot kill the pool, and a hung item is bounded
  by its own timeout;
- **`item.py` pins `OMP_NUM_THREADS=1`** and friends — otherwise N workers each spawn N BLAS
  threads and the box thrashes. The parent pool owns the parallelism;
- **progress streamed** via `imap_unordered` → a slow item does not stall reporting, the
  control channel stays warm, the poller has something to read;
- **completion gated on counting output files**, not on the loop finishing.

## Benchmark your own box first

```bash
AGC_CPU_SCALING=1 scripts/run-probe.sh v6e1 --tpu v6e1
```

Produces a scaling curve for the exact shape the account can allocate, which is more useful
than the table above. Costs a few minutes and a fraction of a compute unit.
