# Session lifecycle

A handle and an assignment are different objects. Conflating them produces most
session-related failures.

| | **assignment** (the VM) | **handle** (the local row) |
|---|---|---|
| lives | Google's side | `sessions.json` on the calling machine |
| id | `endpoint`, e.g. `gpu-a100-s-kkb-usc1f0-wkwy8aqbx3tz` | a local label, e.g. `train-run-7` |
| created by | `colab new` (server allocation) | `colab new` (local bookkeeping) |
| destroyed by | `colab stop`, idle reclamation, quota | failed auth, a prune, a deleted file, token expiry on ≤0.6.0 |
| on death | the job dies | **the job keeps running, invisibly** |

## 1. `-s` is always the local label

```bash
colab status -s train-run-7                           # works
colab status -s gpu-a100-s-kkb-usc1f0-wkwy8aqbx3tz    # "Session not found"
```

`status` and `stop` resolve local labels only. A VM whose handle has been lost cannot be
stopped by the CLI at all, only through the client API (§7).

## 2. `colab sessions` is not read-only

```python
# colab_cli.common.State.sync_sessions, 0.7.5
assignments = self.client.list_assignments()
by_endpoint = {a.endpoint: a for a in assignments}
for name, s in list(self._sessions.items()):
    if s.endpoint not in by_endpoint:
        self.prune_session(name)      # deletes the local row
        pruned += 1
    else:
        _apply_proxy_info(s, by_endpoint[s.endpoint].runtime_proxy_info)
        self.store.add(s)             # refreshes the token
```

- **Refreshes tokens** for sessions still present server-side. Periodic calls keep handles warm.
- **Prunes** every local row whose endpoint the *calling account* cannot see. Run as account
  A, it deletes account B's handles and orphans live VMs that are still billing →
  [08](08-multi-account.md).
- `[?]` in a listing means *no label for this assignment in the store that was loaded*. It is
  not evidence that a VM is unowned, broken, or safe to kill.

## 3. Handle expiry: ≤0.6.0 versus 0.7.x

Proxy-token TTL is 3600 s.

- **≤0.6.0** stored the token at `colab new` and never refreshed it. About an hour later
  every `exec`/`upload`/`download` against a healthy VM returned 401/404, and the CLI often
  deleted the handle (`Session 'X' appears to be lost (404/401). Cleaning up.`) while
  `colab sessions` still listed the machine. Polling did not extend it: handles stopped
  working at ~55–75 min whether watched or idle. The VM was unaffected.
- **0.7.x** added `token_expires_at` and refreshes on access:

```python
TOKEN_REFRESH_MARGIN = timedelta(seconds=300)

s = self.store.get(name)
if s and (not s.token_expires_at
          or s.token_expires_at - now(utc) <= TOKEN_REFRESH_MARGIN):
    by_endpoint = {a.endpoint: a for a in self.client.list_assignments()}
    if s.endpoint in by_endpoint:
        _apply_proxy_info(s, by_endpoint[s.endpoint].runtime_proxy_info)
        self.store.add(s)
    else:
        self.prune_session(name); s = None
```

`get_session` runs on every session-targeting command, so any periodic touch
(`colab status -s NAME` is cheapest) renews the credential indefinitely. No keep-alive
subprocess exists, and `SessionState` has no `keep_alive_pid` field.

Use ≥0.7.x. `agenticcolab/refresh_tokens.py` implements the manual refresh for 0.6.0 pins
and is legacy. Two constraints on it:

- it reads `~/.config/colab-cli/sessions.json` via `expanduser`, so it must run with the
  **same HOME** as the campaign whose handles it refreshes. Run from the wrong HOME it
  refreshes nothing, prints "none" and exits 0 — treat "refreshed nothing" as a **failure**;
- it needs an interpreter that can import `colab_cli`.

## 4. VM reclamation is a separate mechanism

Fixing handle expiry does not affect reclamation.

- Long-running GPU sessions do get reclaimed. Any multi-hour job must survive losing its VM
  at any moment.
- A session actively executing work is stickier than an idle one, but not guaranteed.
- Sustained heavy use of one accelerator class can start returning "no quota or entitlement"
  for that class with balance remaining. This is a **throttle, not exhaustion**: it clears in
  ~10–26 h with no purchase.

Design response: [checkpoint ratcheting](04-supervisor-patterns.md), not a longer-lived
session.

## 5. A failed `colab new` may still have allocated

```
ReadTimeout: HTTPSConnectionPool(host='colab.research.google.com', read timeout=120)
```

The creation read timeout is hardcoded. On a slow allocation the client gives up **after**
the backend created the machine. The VM then exists, consumes a concurrency slot and has no
label — a `[?]` orphan.

An allocation error is never proof that nothing was allocated. Re-list after any `colab new`
failure, or the retry competes with the orphan it just created. `scripts/run-probe.sh` and
`agenticcolab/audit.py` both implement this check.

## 6. Recovering a lost handle

A handle is only a row. Identity lives server-side and `list_assignments()` mints fresh
`runtime_proxy_info` on demand, so a lost handle is **rebuilt**:

```python
from colab_cli.common import state, AuthProvider, SessionState
state.auth_provider = AuthProvider.OAUTH2

[a] = [a for a in state.client.list_assignments() if a.endpoint == ENDPOINT]
row = SessionState(
    name="rebuilt-label",
    token=a.runtime_proxy_info.token,
    url=a.runtime_proxy_info.url,
    endpoint=a.endpoint,
    token_expires_at=a.runtime_proxy_info.expires_at(),
    variant=a.variant.name,              # NAME, not value
    accelerator=a.accelerator.name,
)
state.store.add(row)
```

A row, read off a live store:

```json
{
  "my-session": {
    "name": "my-session",
    "token": "eyJhbGciOiJFUzI1NiIs…",
    "url": "https://8080-t1v-n-….us-west4.prod.r.colab.dev",
    "endpoint": "tpu-v5e1-s-r-usw4a0-2whuxv43xqnxx",
    "token_expires_at": "…",
    "variant": "TPU",
    "accelerator": "V5E1",
    "machine_shape": "STANDARD",
    "kernel_id": "5457c271-efb9-47a5-bef0-00a9c828b46b",
    "session_id": "7f8370c5-7f99d21c1932f0beae0be717",
    "last_execution": ["stdin", null, "…"],
    "running": "exec(stdin)"
  }
}
```

Enum fields store **names**: `"TPU"`, `"V5E1"`, `"STANDARD"`. `Shape` and
`AssignmentVariant` are int-valued enums on the assignment object, so `str(x.value)` writes
`"0"`, which **validates fine** — the model declares `str` — and is wrong.

`agenticcolab/adopt_orphan.py` adds the rails: it **refuses when more than one assignment is
unbound**, because on a shared account an unidentified assignment may be another user's job;
it validates with `SessionState(**entry)` before writing; and it copies only the row *shape*
from a template, never live identity fields, because a borrowed `kernel_id`/`session_id`
points at a kernel that does not exist on the adopted machine.

A detached job keeps running on a VM that cannot currently be addressed, so probe an adopted
session before rebuilding anything.

### `sessions.json` fails closed

`StateStore._load_raw` constructs `SessionState(**v)` for every row inside a bare
`except: return {}`. **One malformed row makes the CLI report every session as "not found"**,
healthy ones included, with no error explaining why.

So: validate before writing; never copy `variant`/`accelerator`/`machine_shape` straight off
an assignment object, since they are enums there and `str` in the model and pydantic v2 will
not coerce them; keep timestamped backups, because restoring one plus a token refresh is a
complete recovery path.

## 7. Reaping orphans

`colab stop` cannot target an orphan. The client API can, instantly:

```python
from colab_cli.common import state, AuthProvider
state.auth_provider = AuthProvider.OAUTH2
for a in state.client.list_assignments():
    if a.endpoint == MINE:          # an endpoint provably yours
        state.client.unassign(a.endpoint)
```

This clears a 412 immediately rather than waiting out an idle timeout.

On a shared account, loop-and-unassign-everything kills another user's job. Filter to an
endpoint that matches a session you created. Never blanket-unassign.
`agenticcolab/reap_orphans.py` requires an explicit `--endpoint` or `--label-prefix` and
refuses a bare "all".

## 8. Counting assignments without side effects

`colab usage` reports `Active assignments: N` and does not touch the local store, unlike
`colab sessions`, which prunes.

```bash
$ colab usage
Current balance: 84.85 compute units
Usage rate: 8.90/hr
Active assignments: 1
```

```python
state.client.get_consumption_user_info()
# {'paid_compute_units_balance': 83.673…, 'consumption_rate_hourly': 2.92,
#  'assignments_count': 1}
```

## 9. Verify release

```bash
colab stop -s "$SESS"
sleep 8
colab usage          # expect 0.00/hr, 0 active assignments
```

A nonzero rate after everything is believed stopped indicates an orphan.
