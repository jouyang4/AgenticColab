# AgenticColab

A measured reference for resilient, long-running, agent-driven Google Colab sessions:
machine-shape matrix, compute-unit rates, session-lifecycle internals, and resilient
supervisor patterns.

Colab allocates large machines in seconds and bills by the hour, but its tooling assumes a
human in a browser. Driving it from code runs into undocumented behaviour:

- session handles that stop working while the VM keeps running
- websocket drops that are indistinguishable from a dead machine
- orphaned assignments that block the next allocation
- an unpublished machine-shape matrix

Provenance:

- measured on `google-colab-cli 0.7.5`
- raw probe output in [`data/`](data/); machine-readable summaries in [`spec/`](spec/)
- `scripts/run-probe.sh` regenerates any row; the full sweep is ~30 min and under 1 CU

## Findings

| # | finding |
|---|---|
| 1 | The largest allocatable shape is a **96 GB-VRAM Blackwell: 48 vCPU, 177 GB RAM, 427 TFLOP/s bf16** measured. |
| 2 | **`--high-mem` is a CPU upgrade**, not just RAM: CPU and T4 shapes go 2 → 8 vCPU and 12.7 → 51 GB. CU/hr rises 3.3× on the CPU shape (0.08 → 0.26) but only 19% on T4 (1.07 → 1.27). |
| 3 | **For CPU cores, allocate a TPU.** Its *host* has 24–44 vCPU against the CPU runtime's 2, and draws from a different pool. The TPU itself goes unused. |
| 4 | **vCPU is a concurrency ceiling, not a throughput multiplier.** Across 5 shapes, peak CPU-bound speedup ≈ *physical* cores. SMT is vendor-split: AMD EPYC +15–18%, Intel Xeon none — L4 is slower at 12 workers than at 6. |
| 5 | **For CPU-bound work the TPU host is ~2.2× cheaper than the big GPU** at equal throughput: G4 24.2× at 8.90 CU/hr, v6e-1 23.2× at 4.08. |
| 6 | **Session handles do not expire on 0.7.x.** ≤0.6.0 cached a 3600 s proxy token and never refreshed it, so handles stopped working about an hour in while the VM stayed healthy. 0.7.x refreshes with a 5-minute margin. |
| 7 | **`colab usage` (0.7.x)** returns balance, live burn rate and active-assignment count without touching the local session store — the safe way to detect orphans, and what makes per-shape cost measurable. |
| 8 | **An empty poll means the websocket dropped, not that the VM died.** A supervisor that stops and re-allocates on a failed poll destroys a healthy machine and all in-flight work → [04](docs/04-supervisor-patterns.md). |
| 9 | **The host CPU model is not fixed per shape.** `--high-mem` allocated an Intel Xeon on one run and an AMD EPYC 7B12 on the next. Read `/proc/cpuinfo`; do not infer the vendor from the flag. |

## Shapes

| `colab new` flags | vCPU | phys | RAM | accelerator | VRAM | CU/hr |
|---|---|---|---|---|---|---|
| *(none)* | 2 | 1 | 12.7 GiB | — | — | 0.08 |
| `--high-mem` | 8 | 4 | 51.0 GiB | — | — | 0.26 |
| `--gpu T4` | 2 | 1 | 12.7 GiB | Tesla T4 | 15.0 GiB | 1.07 |
| `--gpu T4 --high-mem` | 8 | 4 | 51.0 GiB | Tesla T4 | 15.0 GiB | 1.27 |
| `--gpu L4` | 12 | 6 | 53.0 GiB | NVIDIA L4 | 22.5 GiB | 1.54 |
| `--gpu A100` | 12 | 6 | 83.5 GiB | A100-SXM4-40GB | 40.0 GiB | 5.30 |
| `--gpu A100 --high-mem` | 12 | 6 | 167.1 GiB | A100-SXM4-**80GB** | 80.0 GiB | 6.77 |
| `--gpu G4` | 48 | 24 | 176.9 GiB | RTX PRO 6000 Blackwell SE | 95.6 GiB | 8.90 |
| `--tpu v5e1` | 24 | 12 | 47.1 GiB | TPU v5 lite | — | 2.92 |
| `--tpu v6e1` | 44 | 22 | 173.0 GiB | TPU v6 lite | — | 4.08 |

- allocation 3–5 s when capacity exists; the 80 GB A100 took 57 s
- `--high-mem` is ignored for L4 and both TPUs
- bf16 matmul: 427 TFLOP/s on G4, 274 on the 80 GB A100, 58 on L4, 2.1 on T4 — T4 has no
  bf16 tensor cores, so use fp16 there, where it reaches 22
- CU/hr is read while the shape is the **sole** live assignment; `colab usage` is an
  account-wide counter → [09](docs/09-cost-and-quota.md)
- full tables: [01](docs/01-machine-shapes.md), [11](docs/11-benchmarks.md),
  [`spec/shapes.json`](spec/shapes.json)

## Quickstart

```bash
pip install 'google-colab-cli>=0.7.5'
git clone https://github.com/<you>/AgenticColab && cd AgenticColab
```

`colab run` as a shebang interpreter — allocate, run, release:

```python
#!/usr/bin/env -S colab run --gpu T4
import torch
print(torch.cuda.get_device_name(0))
```

```bash
chmod +x gpucheck.py && ./gpucheck.py
```

In a session — `exec` never takes code positionally, stdin or `-f` only:

```bash
colab new -s s1 --gpu L4
echo 'import torch; print(torch.cuda.get_device_name(0))' | colab exec -s s1
colab exec -s s1 -f ./job.py --env SEED=7
colab stop -s s1
```

Long job under the supervisor:

```bash
scripts/colab-job.sh train7 ./my_payload -- --gpu L4
```

`colab-job.sh` provides:

- external timeouts on every CLI call
- chunked, MD5-verified upload
- detached execution; bounded polls that never stop a live VM
- state-change checkpointing and cumulative banking
- verified release

Its payload contract: a `run.sh` that is idempotent, writes into `out/`, and prints
`JOB_COMPLETE`.

```bash
python -m agenticcolab.audit            # prune-safe audit; `colab sessions` prunes
scripts/run-probe.sh a100 --gpu A100    # allocate, inventory, benchmark, release
python3 tools/gen.py                    # regenerate generated docs + spec/shapes.json
```

## Layout

| path | |
|---|---|
| [docs/01](docs/01-machine-shapes.md) | every shape: cores, RAM, VRAM, TFLOP/s, bandwidth, disk, images, cgroups |
| [docs/02](docs/02-cli-reference.md) | command surface, flag semantics, 0.6.0↔0.7.5 diffs, package internals |
| [docs/03](docs/03-session-lifecycle.md) | handle vs VM, token refresh, pruning, orphans, adoption, reaping |
| [docs/04](docs/04-supervisor-patterns.md) | supervisor rules for hours of unattended work |
| [docs/05](docs/05-error-taxonomy.md) | every error, what it means, the correct response |
| [docs/06](docs/06-data-transfer.md) | chunked uploads, atomic snapshots, cumulative banking |
| [docs/07](docs/07-cpu-parallelism.md) | the TPU-host trick, sizing worker pools |
| [docs/08](docs/08-multi-account.md) | isolated CLI homes, the pruning hazard, shared-account rules |
| [docs/09](docs/09-cost-and-quota.md) | compute units, measuring rates, budgeting, the five limits |
| [docs/10](docs/10-antipatterns.md) | catalog of silent failures |
| [docs/11](docs/11-benchmarks.md) | full dataset, methodology, provenance |
| [spec/shapes.json](spec/shapes.json) | measured shape matrix + benchmarks + CU rates (generated) |
| [spec/cli.json](spec/cli.json) | command/flag surface, version diffs, internals |
| [spec/errors.json](spec/errors.json) | error taxonomy as records |
| [spec/antipatterns.json](spec/antipatterns.json) | antipattern records |
| [spec/supervisor.json](spec/supervisor.json) | supervisor rules, code, checklist |
| `scripts/colab-job.sh` | resilient supervisor for a long unattended job |
| `scripts/colab-home.sh` | per-account/per-campaign isolated CLI home wrapper |
| `scripts/upload-big.sh` | chunked, MD5-verified large upload |
| `scripts/run-probe.sh` | allocate → inventory → benchmark → release |
| `tools/gen.py` | regenerates docs 01/02/04/05/10/11 and `spec/shapes.json` |
| `agenticcolab/audit.py` | prune-safe audit of assignments and session stores |
| `agenticcolab/adopt_orphan.py` | rebuild a lost handle for a live VM |
| `agenticcolab/reap_orphans.py` | release assignments the CLI cannot target, with safety rails |
| `agenticcolab/refresh_tokens.py` | legacy manual token refresh for ≤0.6.0 |
| `agenticcolab/probe.py` | the inventory + benchmark payload |
| [examples/01](examples/01-hello-shebang.py) | rent a GPU from a shebang |
| [examples/02](examples/02-long-training-job/) | checkpoint-on-state-change payload |
| [examples/03](examples/03-cpu-fanout/) | saturate a TPU host's CPUs with an idempotent pool |

Docs 01, 02, 04, 05, 10 and 11 are **generated** from `spec/` and `data/` by `tools/gen.py`
— edit the JSON, not the markdown.

## Supervisor checklist

- [ ] every CLI call wrapped in coreutils `timeout` (`--timeout` does **not** bound a lost socket)
- [ ] transfers given their own, larger timeout budget
- [ ] work detached on the VM with `setsid`; polls short and bounded
- [ ] an empty poll triggers a **retried liveness check**, never a stop
- [ ] no `stop` in any failure path
- [ ] kill/restart logic in script files with bracketed `pgrep` patterns
- [ ] `flock` so a second supervisor cannot race the first
- [ ] every stage idempotent and skippable
- [ ] checkpoint on **state-change**, atomically, with the *full* restart state
- [ ] results merged cumulatively, never overwritten
- [ ] success gated on exit status with `pipefail`
- [ ] verification reads changed **state**, never a filtered log
- [ ] `colab new` output never suppressed
- [ ] explicit stop **plus** `colab usage` showing 0 assignments

Rationale per item: [docs/04](docs/04-supervisor-patterns.md).

## Scope

- **Entitlements are per-account and change over time.** `Backend rejected accelerator` is a
  policy answer about the calling account, not a statement about Colab. A shape with no row
  here is one this account could not allocate, not one Colab does not offer. Re-probe.
- **One account.** Rates and shapes may differ by subscription tier and region. Re-measure
  with `scripts/run-probe.sh`.
- **Micro-benchmarks characterise the machine, not a workload.**
- Colab suits bursty, parallel, interruptible work; not anything that must not be reclaimed.

## Contributing

- most useful: measurements from other tiers and entitlements
- include the raw probe JSON and `colab version`
- see [CONTRIBUTING.md](CONTRIBUTING.md)

MIT — see [LICENSE](LICENSE). Not affiliated with Google. "Colab" is Google's.
