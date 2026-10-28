# 02 — long training job that survives a reclaimed VM

A payload for [`scripts/colab-job.sh`](../../scripts/colab-job.sh). The supervisor handles
the session; this directory is the work.

```bash
scripts/colab-job.sh train7 examples/02-long-training-job -- --gpu L4
```

## Contract

`run.sh` in the payload root must be:

| property | meaning |
|---|---|
| idempotent | safe to invoke again after a restart or VM swap |
| incremental | writes under `out/`, one file per completed unit |
| chatty | prints progress to stdout — keeps the control channel warm, gives the poller something to read |
| explicit | prints the literal line `JOB_COMPLETE`, and nothing else claims to |

Break any and the supervisor either redoes work or polls forever.

## Files

**`run.sh`** — loops over units, skips any whose output exists, gates completion on counting
output files rather than reaching the end of the loop. Exits non-zero on missing units so the
supervisor relaunches.

**`train_step.py`** — one unit:

- returns immediately if its output file exists;
- checkpoints every `CKPT_EVERY` steps, not only at the end — a job whose only save is its
  last statement loses everything if the VM goes away before it;
- writes via a temp path then `os.replace` — atomic on POSIX, so a kill mid-write can't leave
  a truncated file a restart would mistake for good state;
- resumes from the checkpoint when present, deletes it on success.

## Try the failure mode

```bash
scripts/colab-job.sh train7 examples/02-long-training-job -- --gpu L4
# ^C after a couple of units
scripts/colab-job.sh train7 examples/02-long-training-job -- --gpu L4
```

Expect: finished units skipped, the partial one resumed from its checkpoint, the existing
session reused. Stop the Colab session between runs instead and expect a fresh VM restored
from the last snapshot in `bank-train7/`.

## Tuning

| env | effect |
|---|---|
| `TOTAL_UNITS` | units `run.sh` iterates over (default 20) |
| `STEPS` | steps per unit (default 40) |
| `CKPT_EVERY` | checkpoint interval in steps (default 5) |
