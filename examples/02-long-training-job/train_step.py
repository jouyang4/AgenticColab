#!/usr/bin/env python3
"""One unit of work, so a reclaimed VM costs at most this unit.

Checkpointing discipline from docs/04-supervisor-patterns.md:
  * skip immediately if this unit's output exists (idempotent)
  * checkpoint on STATE-CHANGE, every N steps, not only at the end
  * temp path + os.replace (atomic: no reader sees a half-written checkpoint,
    and a kill mid-write cannot corrupt the good one)
  * resume from the checkpoint if present

Replace the arithmetic with real work; the structure is the point.
"""
import json
import os
import sys
import time

UNIT = sys.argv[1] if len(sys.argv) > 1 else "0"
OUT = f"out/unit_{UNIT}.json"
CKPT = f"state/unit_{UNIT}.ckpt.json"
STEPS = int(os.environ.get("STEPS", "40"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "5"))


def save_atomic(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)          # atomic on POSIX


def main():
    os.makedirs("out", exist_ok=True)
    os.makedirs("state", exist_ok=True)

    if os.path.exists(OUT):
        print(f"unit {UNIT}: output exists, nothing to do")
        return 0

    start_step, acc = 0, 0.0
    if os.path.exists(CKPT):
        try:
            with open(CKPT) as fh:
                c = json.load(fh)
            start_step, acc = c["step"], c["acc"]
            print(f"unit {UNIT}: resuming from step {start_step}")
        except Exception as exc:  # noqa: BLE001
            print(f"unit {UNIT}: unreadable checkpoint ({exc}), starting over")

    for step in range(start_step, STEPS):
        # ---- stand-in for real work
        time.sleep(0.2)
        acc += (step + 1) ** 0.5

        if (step + 1) % CKPT_EVERY == 0:
            save_atomic(CKPT, {"step": step + 1, "acc": acc})
            # stdout progress keeps the websocket warm and is what the
            # supervisor's poller reads
            print(f"unit {UNIT}: step {step + 1}/{STEPS} acc={acc:.3f} (checkpointed)",
                  flush=True)

    save_atomic(OUT, {"unit": UNIT, "steps": STEPS, "acc": acc,
                      "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime())})
    if os.path.exists(CKPT):
        os.remove(CKPT)
    print(f"unit {UNIT}: wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
