# Contributing

The most useful contribution is a measurement from an account unlike the one measured here.
Entitlements, shapes and CU rates vary by subscription tier, account age and region, and
change over time.

## A measurement

```bash
scripts/run-probe.sh <label> [colab new flags...]
scripts/run-probe.sh l4 --gpu L4
AGC_CPU_SCALING=1 scripts/run-probe.sh v6e1 --tpu v6e1
```

Writes `data/shapes/<label>.json` (inventory + benchmarks) and `<label>.meta.json`
(allocation latency, CU/hr, raw `colab usage` lines). Open a PR with both files plus:

- `colab version`
- the subscription tier, if you are willing to state it
- anything anomalous

Do not hand-edit the JSON; its value is that it is machine-generated. If a number looks
wrong, say so in the PR. Run `python3 tools/gen.py` so the generated docs and
`spec/shapes.json` match the data.

Wanted specifically:

- **Shapes with no row here**, from any account whose entitlements cover them.
- **Multi-GPU shapes**, if any tier offers them.
- **Free-tier rows** for every shape.
- **Scaling curves** (`AGC_CPU_SCALING=1`) for shapes probed only once.
- **Anything that contradicts a claim in the docs.**

## Documentation

A factual claim must be reproducible by someone else.

- Docs 01, 02, 04, 05, 10 and 11 are generated. Edit `spec/*.json`, then run
  `python3 tools/gen.py`.
- State how the claim was established: "measured with `<command>`", not "Colab does X".
- Record the CLI version with capability claims. Entitlements move, and a bare claim goes
  stale silently.
- No account-specific content: no credential paths, account identifiers, project names or
  internal infrastructure. Paths in examples must be obvious placeholders.
- Prose is minimised deliberately. Tables, records and runnable commands over paragraphs; no
  narrative, no anecdote, no history of how a conclusion was reached.

## Code

- Bash: `set -uo pipefail`, clean under `bash -n`. Wrap **every** CLI call in coreutils
  `timeout`, with a larger budget for transfers.
- Python: stdlib-only in anything that runs on a Colab VM, so the probe works on every
  runtime image without installing anything. Degrade to `None` rather than raising.
- No tool may stop or unassign a session it cannot prove belongs to the caller. Shared
  accounts are normal. **Refuse and explain** when ownership is ambiguous.
- Do not filter the output of a step being verified. Verify changed state.

## Checks

```bash
bash -n scripts/*.sh
python3 -m py_compile agenticcolab/*.py tools/*.py
python3 -m json.tool data/shapes/*.json > /dev/null
python3 -m json.tool spec/*.json > /dev/null
python3 tools/gen.py && git diff --exit-code docs spec   # generated files in sync
```

## Issues

Shape, CLI version, exact error text, expected behaviour. A new row in `spec/errors.json` is
a sufficient PR on its own; the taxonomy is only useful complete.
