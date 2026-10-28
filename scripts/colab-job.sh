#!/usr/bin/env bash
# Resilient supervisor for a long, unattended Colab job.
#   ./colab-job.sh <job-name> <payload-dir> [-- colab new flags...]
#   ./colab-job.sh train7 ./payload -- --gpu L4
#
# Survives dropped websockets, expired handles, reclaimed VMs, its own restart,
# and a second copy launched by mistake. NEVER stops a session in a failure
# path: an empty poll means the control channel dropped, not that the VM died.
#
# Payload dir -> tarball -> /content/work. Must contain a `run.sh` that is
# idempotent, writes incremental output under out/, prints progress, and prints
# the literal line JOB_COMPLETE when done.
#
# Env: AGC_CLI AGC_CLI_HOME AGC_AUTH AGC_BANK AGC_POLL_SLEEP AGC_MAX_HOURS
set -uo pipefail

JOB="${1:?usage: colab-job.sh <job-name> <payload-dir> [-- colab new flags...]}"
PAYLOAD_DIR="${2:?missing payload dir}"
shift 2
[[ "${1:-}" == "--" ]] && shift
NEW_FLAGS=("$@")

CLI="${AGC_CLI:-colab}"
CLI_HOME="${AGC_CLI_HOME:-$HOME}"
AUTH="${AGC_AUTH:-oauth2}"
BANK="${AGC_BANK:-$PWD/bank-$JOB}"           # cumulative result dir, never overwritten
STATE="$PWD/.state-$JOB"
LOCK="$PWD/.lock-$JOB"
POLL_SLEEP="${AGC_POLL_SLEEP:-30}"
MAX_HOURS="${AGC_MAX_HOURS:-24}"
SESS="$JOB"

mkdir -p "$BANK" "$STATE"
LOG="$STATE/supervisor.log"

# one supervisor only: two writing one output dir produces plausible garbage
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another supervisor holds $LOCK; exiting"; exit 0
fi

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# coreutils timeout on every call: `colab exec --timeout` does not bound a lost
# websocket and can hang forever
cc()    { local t=$1; shift; HOME="$CLI_HOME" timeout "$t" "$CLI" --auth="$AUTH" "$@"; }
ccbig() { cc 1800 "$@"; }                       # transfers get their own budget
# exec takes code on stdin or via -f, never positionally
ccx()   { local t=$1 code=$2; printf '%s\n' "$code" | cc "$t" exec -s "$SESS" --timeout 300; }

say "=== supervisor start: job=$JOB flags='${NEW_FLAGS[*]:-CPU}' pid=$$ ==="

# --------------------------------------------------------------- helpers

session_listed() {
  # retried: one dropped `sessions` call is not evidence of death
  local i
  for i in 1 2 3; do
    cc 90 sessions 2>/dev/null | grep -q "^\[$SESS\]" && return 0
    sleep 5
  done
  return 1
}

live_assignments() {
  # side-effect free, unlike `colab sessions` which prunes
  cc 60 usage 2>/dev/null | sed -n 's/.*Active assignments:[[:space:]]*\([0-9]*\).*/\1/p'
}

ensure_session() {
  if session_listed; then say "session '$SESS' is live"; return 0; fi

  say "allocating session '$SESS' ${NEW_FLAGS[*]:-(CPU)}"
  # never suppressed: the message is the only thing distinguishing entitlement
  # vs capacity vs concurrency
  if ! cc 300 new -s "$SESS" "${NEW_FLAGS[@]}" 2>&1 | tee -a "$LOG"; then
    say "allocation reported failure; re-listing before concluding anything"
    # a client-side ReadTimeout can still have created the VM
    if session_listed; then
      say "session exists despite the error - continuing"
    else
      say "no session; assignments=$(live_assignments). Backing off 120s."
      return 1
    fi
  fi
  rm -f "$STATE/bootstrapped"
  return 0
}

bootstrap() {
  [[ -f "$STATE/bootstrapped" ]] && { say "already bootstrapped"; return 0; }

  say "packing payload from $PAYLOAD_DIR"
  tar czf "$STATE/payload.tgz" -C "$PAYLOAD_DIR" .

  # chunked: one large upload dies with SSLEOFError through the runtime proxy.
  # deliberately not under `set -e` - a failed upload must not abort before the
  # extraction step
  rm -rf "$STATE/chunks"; mkdir -p "$STATE/chunks"
  split -b 16M "$STATE/payload.tgz" "$STATE/chunks/part_"
  ccx 120 "import os,shutil;shutil.rmtree('/content/chunks',ignore_errors=True);os.makedirs('/content/chunks',exist_ok=True);print('CHUNKDIR_OK')" \
    >>"$LOG" 2>&1
  local f n=0
  for f in "$STATE"/chunks/part_*; do
    local tries=0
    until ccbig upload -s "$SESS" "$f" "/content/chunks/$(basename "$f")" >>"$LOG" 2>&1; do
      tries=$((tries+1)); [[ $tries -ge 3 ]] && { say "upload of $(basename "$f") failed 3x"; return 1; }
      say "retrying upload of $(basename "$f") ($tries)"; sleep 10
    done
    n=$((n+1))
  done
  say "uploaded $n chunk(s)"

  local md5_local; md5_local=$(md5sum "$STATE/payload.tgz" | cut -d' ' -f1)
  local verify
  verify=$(ccx 300 "
import hashlib, subprocess
subprocess.run('cat /content/chunks/part_* > /content/payload.tgz', shell=True, check=True)
h = hashlib.md5(open('/content/payload.tgz','rb').read()).hexdigest()
print('MD5', h)
")
  if ! grep -q "$md5_local" <<<"$verify"; then
    say "MD5 MISMATCH after reassembly (local $md5_local): $verify"
    return 1
  fi
  say "payload md5 verified: $md5_local"

  ccx 600 "
import subprocess
subprocess.run(['bash','-lc',
  'mkdir -p /content/work /content/work/out && tar xzf /content/payload.tgz -C /content/work && echo EXTRACT_OK'],
  check=True)
" >>"$LOG" 2>&1 || { say "extract failed"; return 1; }

  touch "$STATE/bootstrapped"
  say "bootstrap complete"
}

restore_snapshot() {
  local latest
  latest=$(ls -1t "$BANK"/snap_*.tgz 2>/dev/null | head -1)
  [[ -z "$latest" ]] && { say "no snapshot to restore"; return 0; }
  say "restoring $latest"
  ccbig upload -s "$SESS" "$latest" /content/restore.tgz >>"$LOG" 2>&1 || {
    say "snapshot upload failed; continuing from scratch"; return 0; }
  ccx 600 "
import subprocess
subprocess.run(['bash','-lc','cd /content/work && tar xzf /content/restore.tgz && echo RESTORE_OK'], check=True)
" >>"$LOG" 2>&1 && say "snapshot restored"
}

launch_unit() {
  # bracketed pgrep pattern, appearing nowhere else in this remote command, or
  # it matches the launcher itself
  ccx 180 "
import subprocess
cmd = (\"if pgrep -f '[r]un[.]sh' >/dev/null; then echo ALREADY_RUNNING; else \"
       \"cd /content/work && setsid nohup bash ./run.sh > /content/work/job.out 2>&1 < /dev/null & \"
       \"echo LAUNCHED \$!; fi\")
p = subprocess.run(['bash','-lc',cmd], capture_output=True, text=True)
print(p.stdout.strip(), p.stderr.strip()[:200])
"
}

snapshot_and_bank() {
  local ts; ts=$(date +%s)
  # tar on the VM: never download a file the job is actively writing
  ccx 900 "
import subprocess
r = subprocess.run(['bash','-lc','cd /content/work && tar czf /content/snap.tgz out/ state/ 2>/dev/null || tar czf /content/snap.tgz out/; echo TAR_OK'],
                   capture_output=True, text=True)
print(r.stdout.strip()[-200:])
" >>"$LOG" 2>&1
  # always download even if exec reported failure: the CLI reports failure for
  # cells that completed server-side
  if ccbig download -s "$SESS" /content/snap.tgz "$BANK/snap_$ts.tgz" >>"$LOG" 2>&1; then
    say "banked snapshot snap_$ts.tgz"
    # merge into a persistent dir - never overwrite one archive, results come
    # from a succession of different VMs
    mkdir -p "$BANK/merged"
    tar xzf "$BANK/snap_$ts.tgz" -C "$BANK/merged" 2>/dev/null || true
  else
    say "snapshot download failed (will retry next cycle)"
  fi
}

# --------------------------------------------------------------- main loop

START=$(date +%s)
EMPTY_RUN=0
COMPLETE=0

while :; do
  ELAPSED_H=$(( ($(date +%s) - START) / 3600 ))
  if (( ELAPSED_H >= MAX_HOURS )); then
    say "max runtime ${MAX_HOURS}h reached; banking and stopping"
    snapshot_and_bank; break
  fi

  if ! ensure_session; then sleep 120; continue; fi
  if ! bootstrap;     then sleep 60;  continue; fi
  restore_snapshot
  say "launch: $(launch_unit 2>&1 | tr '\n' ' ')"

  # ---- poll
  while :; do
    sleep "$POLL_SLEEP"
    P=$(ccx 120 "import os
p='/content/work/job.out'
print(open(p).read()[-800:] if os.path.exists(p) else 'NOFILE')" 2>&1)

    # a CLI invocation error looks exactly like a dropped poll if the only test
    # is emptiness
    if grep -qE 'Usage: colab|unexpected extra argument|No such option' <<<"$P"; then
      say "CLI INVOCATION ERROR (a bug in this script, not a network event):"
      echo "$P" | tee -a "$LOG"; exit 3
    fi

    if grep -q 'JOB_COMPLETE' <<<"$P"; then
      say "job reports JOB_COMPLETE"; COMPLETE=1; snapshot_and_bank; break
    fi

    if [[ -z "${P// /}" || "$P" == *NOFILE* ]]; then
      EMPTY_RUN=$((EMPTY_RUN+1))
      if session_listed; then
        say "empty poll ($EMPTY_RUN) but session is listed -> dropped websocket, continuing"
        EMPTY_RUN=0
        continue
      fi
      say "empty poll and session NOT listed (assignments=$(live_assignments))"
      if (( EMPTY_RUN >= 3 )); then
        say "VM loss confirmed; will re-allocate and resume from last snapshot"
        rm -f "$STATE/bootstrapped"
        EMPTY_RUN=0
        break
      fi
      continue
    fi

    EMPTY_RUN=0
    say "progress: $(tail -c 200 <<<"$P" | tr '\n' ' ')"
    # bank on state-change: a new output unit, not a timer
    NUNITS=$(ccx 90 "import os,glob;print(len(glob.glob('/content/work/out/*')))" 2>/dev/null | tr -dc '0-9')
    if [[ -n "$NUNITS" ]] && [[ "${NUNITS:-0}" -gt "$(cat "$STATE/units" 2>/dev/null || echo 0)" ]]; then
      say "unit count rose to $NUNITS - banking"
      echo "$NUNITS" > "$STATE/units"
      snapshot_and_bank
    fi
  done

  (( COMPLETE )) && break
done

# release, then verify released - only on genuine completion
if (( COMPLETE )); then
  say "stopping session"
  cc 180 stop -s "$SESS" >>"$LOG" 2>&1
  sleep 8
  say "post-stop usage: $(cc 90 usage 2>&1 | tr '\n' ' ')"
  A=$(live_assignments)
  if [[ "${A:-0}" != "0" ]]; then
    say "WARNING: $A assignment(s) still active - check for an orphan"
  fi
  say "=== job $JOB COMPLETE; results merged in $BANK/merged ==="
else
  say "=== job $JOB ended without JOB_COMPLETE; session left ALONE on purpose ==="
  say "    inspect with: HOME=$CLI_HOME $CLI --auth=$AUTH sessions"
fi
