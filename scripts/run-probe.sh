#!/usr/bin/env bash
# Allocate one Colab shape, inventory + benchmark it, release it. Also a worked
# example of the whole lifecycle: allocate -> re-list after any error -> read
# burn rate while live -> upload -> detached run -> bounded polls that never
# kill a live VM -> download -> stop -> verify released.
#
#   ./run-probe.sh <label> [colab new flags...]
#   ./run-probe.sh cpu
#   ./run-probe.sh a100 --gpu A100
#   ./run-probe.sh a100-hm --gpu A100 --high-mem
#   AGC_CPU_SCALING=1 ./run-probe.sh v6e1 --tpu v6e1
#
# Env: AGC_CLI AGC_CLI_HOME AGC_AUTH AGC_OUTDIR AGC_PROBE AGC_BENCH(1)
#      AGC_CPU_SCALING(0) AGC_POLL_MAX AGC_POLL_SLEEP
set -uo pipefail

LABEL="${1:?usage: run-probe.sh <label> [colab new flags...]}"; shift
SESS="probe-${LABEL}-$$"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${AGC_CLI:-colab}"
CLI_HOME="${AGC_CLI_HOME:-$HOME}"
AUTH="${AGC_AUTH:-oauth2}"
OUTDIR="${AGC_OUTDIR:-$HERE/../data/shapes}"
PROBE="${AGC_PROBE:-$HERE/../agenticcolab/probe.py}"
BENCH="${AGC_BENCH:-1}"
CPU_SCALING="${AGC_CPU_SCALING:-0}"
POLL_MAX="${AGC_POLL_MAX:-120}"
POLL_SLEEP="${AGC_POLL_SLEEP:-20}"

[[ -f "$PROBE" ]] || { echo "probe payload not found at $PROBE" >&2; exit 1; }
mkdir -p "$OUTDIR"
LOG="$OUTDIR/${LABEL}.log"; : > "$LOG"

# bounded externally: `colab exec --timeout` does not bound a lost websocket
cc()  { local t=$1; shift; HOME="$CLI_HOME" timeout "$t" "$CLI" --auth="$AUTH" "$@"; }
# exec takes code on stdin or via -f, never positionally
ccx() { local t=$1 code=$2; printf '%s\n' "$code" | cc "$t" exec -s "$SESS" --timeout 300; }
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== probe $LABEL (session $SESS) flags: ${*:-CPU} ==="

# `colab usage` reports the ACCOUNT-WIDE rate, so a reading is this shape's rate
# only while it is the ONLY live assignment. Refuse otherwise.
USAGE_BEFORE=$(cc 90 usage 2>&1 | tr '\n' ' ')
N_BEFORE=$(sed -n 's/.*Active assignments:[[:space:]]*\([0-9]*\).*/\1/p' <<<"$USAGE_BEFORE")
say "usage before: $USAGE_BEFORE"
if [[ "${N_BEFORE:-0}" != "0" && "${AGC_ALLOW_CONCURRENT:-0}" != "1" ]]; then
  say "REFUSING: ${N_BEFORE} assignment(s) already live; a CU rate read now is an"
  say "account aggregate, not this shape's rate. Release them first, or set"
  say "AGC_ALLOW_CONCURRENT=1 to probe anyway (cu_rate_clean will be false)."
  exit 3
fi

# allocate. never suppressed: the message is the only thing distinguishing
# entitlement / capacity / concurrency
T0=$(date +%s)
cc 300 new -s "$SESS" "$@" >>"$LOG" 2>&1
RC=$?
ALLOC_S=$(( $(date +%s) - T0 ))
say "allocate rc=$RC in ${ALLOC_S}s"

if [[ $RC -ne 0 ]]; then
  # a ReadTimeout can still have created the VM: an allocation error is never
  # proof that nothing was allocated
  say "allocation reported failure; re-listing before concluding anything"
  if ! cc 90 sessions 2>/dev/null | grep -q "$SESS"; then
    say "RESULT: not allocated. Reason:"
    grep -iE 'reject|quota|entitle|precondition|timeout|unavailable' "$LOG" | tail -3
    python3 - "$OUTDIR/${LABEL}.meta.json" "$LABEL" "$RC" "$ALLOC_S" "$*" <<'PY'
import json, sys
dest, label, rc, secs, flags = sys.argv[1:6]
json.dump({"label": label, "outcome": "not_allocated", "alloc_rc": int(rc),
           "alloc_seconds": int(secs), "new_flags": flags.strip()},
          open(dest, "w"), indent=2)
PY
    exit 2
  fi
  say "session EXISTS despite the error - continuing"
fi

SESS_LINE=$(cc 90 sessions 2>/dev/null | grep "$SESS" | head -1)
say "session: $SESS_LINE"

# burn rate, read WHILE live - the only way to get a per-shape CU rate
sleep 8
USAGE_LIVE=$(cc 90 usage 2>&1 | tr '\n' ' ')
say "usage while live: $USAGE_LIVE"
N_LIVE=$(sed -n 's/.*Active assignments:[[:space:]]*\([0-9]*\).*/\1/p' <<<"$USAGE_LIVE")
if [[ "${N_LIVE:-0}" != "1" ]]; then
  say "WARNING: ${N_LIVE} assignments live, so ${USAGE_LIVE##*Usage rate: } is an"
  say "aggregate. Recording cu_rate_clean=false; the rate will not be published."
fi

cc 240 upload -s "$SESS" "$PROBE" /content/probe.py >>"$LOG" 2>&1 \
  && say "payload uploaded" || say "WARN upload failed"

# detached launch: never hold a multi-minute exec open
LAUNCH=$(cat <<PYEOF
import subprocess
cmd = ("cd /content && AGC_ACCEL='$LABEL' AGC_BENCH='$BENCH' "
       "AGC_CPU_SCALING='$CPU_SCALING' AGC_OUT=/content/agc_probe.json "
       "setsid nohup python3 probe.py > /content/probe.out 2>&1 < /dev/null & echo \$!")
p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
print("LAUNCHED", p.stdout.strip(), p.stderr.strip()[:200])
PYEOF
)
ccx 180 "$LAUNCH" >>"$LOG" 2>&1
say "launched detached"

# poll: an empty poll means the WEBSOCKET dropped, not that the VM died
DONE=0; EMPTY=0
for ((i=1; i<=POLL_MAX; i++)); do
  sleep "$POLL_SLEEP"
  P=$(ccx 120 "import os
p='/content/probe.out'
print(open(p).read()[-700:] if os.path.exists(p) else 'NOFILE')" 2>&1)

  if grep -qE 'Usage: colab|unexpected extra argument|No such option' <<<"$P"; then
    say "CLI INVOCATION ERROR (a bug here, not a network event)"; echo "$P" >>"$LOG"; break
  fi
  if grep -q AGC_JSON_END <<<"$P"; then DONE=1; say "poll $i: COMPLETE"; break; fi
  if grep -q Traceback <<<"$P"; then say "poll $i: payload raised"; echo "$P" >>"$LOG"; break; fi

  if [[ -z "${P// /}" ]]; then
    EMPTY=$((EMPTY+1))
    alive=0
    for _ in 1 2 3; do
      cc 90 sessions 2>/dev/null | grep -q "$SESS" && { alive=1; break; }
      sleep 5
    done
    if (( alive )); then
      say "poll $i: empty (dropped websocket) - session live, continuing"; EMPTY=0
    else
      say "poll $i: empty AND absent; usage: $(cc 60 usage 2>&1 | tr '\n' ' ')"
      (( EMPTY >= 3 )) && { say "giving up"; break; }
    fi
  else
    EMPTY=0
    say "poll $i: running"
  fi
done

# bank BEFORE stopping, and download even after a non-zero exec: the CLI
# reports failure for cells that completed server-side
cc 300 download -s "$SESS" /content/agc_probe.json "$OUTDIR/${LABEL}.json" >>"$LOG" 2>&1 \
  && say "banked $OUTDIR/${LABEL}.json" || say "WARN download failed"
cc 180 download -s "$SESS" /content/probe.out "$OUTDIR/${LABEL}.out" >>"$LOG" 2>&1 || true

USAGE_AFTER=$(cc 90 usage 2>&1 | tr '\n' ' ')
python3 - "$OUTDIR/${LABEL}.meta.json" "$LABEL" "$ALLOC_S" "$RC" \
  "$USAGE_LIVE" "$USAGE_AFTER" "$SESS_LINE" "$USAGE_BEFORE" "$*" <<'PY'
import json, re, sys
dest, label, secs, rc, usage, usage2, line, usage0, flags = sys.argv[1:10]
num = lambda pat, s: (lambda m: float(m.group(1)) if m else None)(re.search(pat, s))
n_before = num(r"Active assignments:\s*([0-9]+)", usage0)
n_live = num(r"Active assignments:\s*([0-9]+)", usage)
# the rate is this shape's rate only if nothing else was assigned at any point
clean = (n_before == 0) and (n_live == 1)
json.dump({
    "label": label, "new_flags": flags.strip(), "outcome": "ok",
    "alloc_seconds": int(secs), "alloc_rc": int(rc), "session_line": line.strip(),
    "cu_per_hour": num(r"Usage rate:\s*([0-9.]+)", usage),
    "cu_rate_clean": clean,
    "balance_at_start": num(r"balance:\s*([0-9.]+)", usage),
    "balance_after": num(r"balance:\s*([0-9.]+)", usage2),
    "active_assignments_before": n_before,
    "active_assignments_while_live": n_live,
    "usage_raw_before": usage0.strip(),
    "usage_raw_live": usage.strip(), "usage_raw_after": usage2.strip(),
}, open(dest, "w"), indent=2)
PY

# release, then verify: an orphan burns rate and blocks the next allocation
cc 180 stop -s "$SESS" >>"$LOG" 2>&1
say "stop issued"
sleep 8
say "usage after stop: $(cc 90 usage 2>&1 | tr '\n' ' ')"
say "=== $LABEL done (payload_complete=$DONE) ==="
