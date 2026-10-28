# Cost and quota

Colab bills accelerator time in **compute units** (CU) and publishes no per-shape rate. On
0.7.x you can measure it in one command.

## 1. `colab usage`

```bash
$ colab usage
Current balance: 84.85 compute units
Usage rate: 8.90/hr
Active assignments: 1
```

| field | meaning |
|---|---|
| balance | CU remaining |
| usage rate | current aggregate burn across all live assignments, CU/hr |
| active assignments | authoritative live count, **without touching the local session store** |

Unlike `colab sessions` it cannot prune, which makes it the safe health check in
multi-account setups and the way to detect an orphan. 0.7.x only.

## 2. Measuring a shape's rate

`Usage rate` is the **account-wide** burn across every live assignment, so it is a per-shape
rate only when the probed shape is the **sole** live assignment. That is a precondition to
assert, not an assumption to make: two live assignments report their sum, and the sum still
reads as a plausible single-shape rate.

```bash
colab usage            # REQUIRE: Active assignments: 0
colab new -s probe --gpu A100
sleep 8                # the rate takes a moment to reflect a new assignment
colab usage            # REQUIRE: Active assignments: 1  -> this is A100's rate
colab stop -s probe
```

- Read the rate **while the assignment is live**; it returns to `0.00/hr` on release.
- Record the assignment count alongside the number. Without it, an aggregate is
  indistinguishable from a rate once written down.

`scripts/run-probe.sh` does this as part of a full inventory pass: it refuses to start when
anything else is assigned, and records `cu_rate_clean` into `<label>.meta.json`.
`tools/gen.py` publishes a rate only when that flag is true, and prints `—` otherwise.
Measured rates: [01](01-machine-shapes.md), [11](11-benchmarks.md),
[`spec/shapes.json`](../spec/shapes.json).

## 3. Cost per unit of work, not per hour

A rate table gives the cost of an hour, not the cost of a job. Measure throughput per shape
and divide. The cheapest shape per hour is frequently not the cheapest per unit of work: 3×
the hourly rate finishing 5× faster is cheaper and sooner. A large accelerator is waste for a
job that cannot fill it.

Three ways the number comes out wrong by more than 2×:

1. **Treating fixed cost as negligible.** Fit `seconds_per_item = a + b × n` instead of
   quoting one per-item rate. The fixed term can be most of a small job and still a large
   fraction of a big one, so a rate derived at a single size does not transfer.
2. **Deriving a rate from a resumed run.** Its wall-clock excludes the setup it skipped.
3. **Deriving a rate from a run that shared the device.** Check contention (`nvidia-smi`)
   before trusting a timing, and record a contention flag with the measurement so a polluted
   number stays attributable instead of entering the cost model silently.

A figure with no machine-readable file behind it is not a cost. Write the JSON, then read it
back when quoting the number.

## 4. Count box-hours, not work units

Occupied-machine-hours is the scarce quantity. An arm made of many short jobs can hold most
of the contended capacity for a small fraction of the nominal work — invisible when counting
work units, obvious in box-hours.

Many short jobs are also launch-bound rather than throughput-bound, so a scaling law fitted
on long jobs underestimates them. Quote measured wall-clock for those, not an extrapolation.

## 5. Nearly-free lanes

- **CPU runtimes:** a small fraction of a CU per hour, **separate pool** from GPUs, so a CPU
  session never costs a GPU slot. Analysis, post-processing, data munging.
- **TPU hosts:** the cheap way to buy many CPU cores → [07](07-cpu-parallelism.md).

## 6. Budget-gate the supervisor

```bash
BAL=$(colab usage | sed -n 's/.*balance: \([0-9.]*\).*/\1/p')
if (( $(echo "$BAL < 10" | bc -l) )); then
  echo "balance $BAL below floor - stopping cleanly"
  bank_results && colab stop -s "$SESS" && exit 0
fi
```

Check at unit boundaries, bank, stop deliberately rather than being cut off mid-unit.

## 7. Idle machines are pure loss

An allocated session burns its rate whether or not it computes.

- **Stop at phase boundaries**, not at the end of the day.
- An orphan is the worst case: it burns rate *and* holds a concurrency slot so your next
  allocation 412s. Finish every job with an explicit stop and a verification that `usage`
  shows `0.00/hr` and `Active assignments: 0`.

## 8. Quota is five separate limits

| limit | how it manifests |
|---|---|
| CU balance | the account stops granting paid shapes |
| concurrent assignments | 412 `TooManyAssignmentsError` (orphans count) |
| per-class entitlement | instant "Backend rejected accelerator 'X'" |
| per-class capacity | HTTP 503 |
| usage throttle | "no quota or entitlement" with balance remaining; self-heals in ~10–26 h |

→ [05](05-error-taxonomy.md)

## 9. Which budget does the hour come from?

Across providers, compare **which budget** the hour is drawn from, not the headline $/hour.
Platform-locked credits that expire unused have an effective marginal cost near zero; a
metered account is cash against a hard limit. Ignoring that leads to spending cash to
conserve credit that cannot be spent on anything else. When wall-clock is the binding
constraint, run the lanes in parallel rather than serialising onto the cheapest one.
