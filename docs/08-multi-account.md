# Multiple accounts and concurrent campaigns

> **Give every account and every campaign its own CLI `HOME`.**

## 1. The hazard

One session store per `HOME`: `$HOME/.config/colab-cli/sessions.json`. `colab sessions`
reconciles it against the server and **prunes every local row the calling account cannot
see**:

```
[colab] Pruned 2 stale local session(s).
```

Running `colab sessions` as account A deletes account B's handles. The VMs stay alive and
keep consuming quota but are no longer addressable by the CLI. The same applies to two
campaigns sharing one store when one lists sessions while the other's rows are not visible.
Present in 0.7.5 — `sync_sessions` source in [03 §2](03-session-lifecycle.md).

## 2. Isolated homes

```bash
#!/usr/bin/env bash
# one wrapper per account/campaign:  myproj-colab.sh [TIMEOUT] <colab args...>
set -uo pipefail
REAL_HOME=/home/you
export PATH="$REAL_HOME/.local/bin:$PATH"
export HOME="$REAL_HOME/.colab_homes/myproj"      # the isolation
T=300
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then T=$1; shift; fi
exec timeout "$T" "$REAL_HOME/.local/bin/colab" --auth=oauth2 "$@"
```

```bash
mkdir -p ~/.colab_homes/projA/.config/colab-cli
cp ~/.config/colab-cli/token.json ~/.colab_homes/projA/.config/colab-cli/   # oauth2 path
echo '{}' > ~/.colab_homes/projA/.config/colab-cli/sessions.json
chmod 600 ~/.colab_homes/projA/.config/colab-cli/sessions.json
```

ADC path: pin the credential directory **absolutely**, because `HOME` moved.

```bash
export HOME=/home/you/.colab_homes/projA
export CLOUDSDK_CONFIG=/home/you/.config/gcloud-projA
colab --auth=adc sessions
```

`scripts/colab-home.sh` is a parameterised version.

**`--config` is the lighter alternative:** `colab --config /path/projA_sessions.json
sessions` isolates the store without moving `HOME`. Good hygiene, better than sharing the
default store. Two caveats: it isolates nothing else under `HOME`, and it creates the
diagnostic trap in §4.

## 3. `--auth` is not an account selector

```bash
CLOUDSDK_CONFIG=~/.config/gcloud-A colab --auth=adc sessions   # account A
CLOUDSDK_CONFIG=~/.config/gcloud-B colab --auth=adc sessions   # account B
```

`--auth=adc` selects a *mechanism*. Reading it as "the other account" makes `--auth=adc`
against the wrong config look safe. With no `colab whoami` and ADC files that routinely have
an empty `account` field, **the config pointed at is the only thing that determines
identity** — set it explicitly on every invocation. For accounts that must stay rigorously
apart, such as separate billing entities, put the discipline in a per-account wrapper and
never call the bare binary.

## 4. An empty default `sessions.json` is NOT evidence of an orphan

A campaign launched with `--config /path/campaign_sessions.json` keeps its state **there**,
so the default store reads `{}` while everything is healthy. An audit that reads only the
default store sees an unlabelled, apparently unowned VM consuming quota.

Before calling any session an orphan:

1. **Harvest `--config` paths from running processes** and check those stores too.
2. Build the held-endpoint set **once** from process args, then test membership. Do **not**
   `grep ps` per endpoint in a loop — the grep matches your own auditing pipeline and yields
   a false negative. (Same family as the `pkill -f` self-match trap,
   [04 §4](04-supervisor-patterns.md).)
3. `[?]` means *no label in the store that was loaded*, nothing more.
4. `colab status -s <endpoint-id>` returning "Session not found" proves nothing: `-s` takes
   the local label, and with the wrong store loaded every command looks like the session is gone.

`agenticcolab/audit.py` is prune-safe along these lines: it enumerates every `--config` store
it can find, reports assignments without self-matching, checks the concurrency ceiling, and
**stops nothing** — releasing an unidentified session stays a human decision.

## 5. Separate quota pools

- **CPU runtimes are a separate, nearly-free pool.** A CPU session never steals a GPU slot.
  Put analysis and post-processing there → [07](07-cpu-parallelism.md).
- **Different GPU classes can behave as separate pools.** A T4 allocated while L4 was
  exhausted. If one class 412s or 503s, try another before concluding you're out.
- **Different accounts are separate payers, not necessarily different entitlements.** Two
  accounts measured on the same day had an identical accelerator matrix: a second wallet, not
  a higher tier. Measure rather than assume.

## 6. Shared accounts

An unidentified assignment may be another user's job.

- stop sessions by **your own label prefix** only;
- never blanket-stop or loop-and-unassign;
- a tool that cannot tell whose an assignment is should **refuse to act and say so** —
  `adopt_orphan.py` refuses when more than one assignment is unbound, for exactly this reason.

## 7. Concurrency ceiling

Small and empirical — 2 on the tiers measured here — and **orphans count against it**. On a
412 with nothing known to be running, check `colab usage`: authoritative count, no store
mutation.
