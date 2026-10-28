#!/usr/bin/env bash
# Chunked, MD5-verified upload of a large file to a Colab session.
#   ./upload-big.sh <session> <local-file> <remote-path> [chunk-size]
#
# One large `colab upload` dies intermittently through the runtime proxy with
# `SSLEOFError: EOF occurred in violation of protocol`. 16 MB chunks + retry +
# reassemble on the VM + MD5 compare works first time, every time.
#
# Env: AGC_CLI AGC_CLI_HOME AGC_AUTH AGC_RETRIES AGC_TIMEOUT
set -uo pipefail

SESS="${1:?usage: upload-big.sh <session> <local-file> <remote-path> [chunk-size]}"
SRC="${2:?missing local file}"
DST="${3:?missing remote path}"
CHUNK="${4:-16M}"

CLI="${AGC_CLI:-colab}"
CLI_HOME="${AGC_CLI_HOME:-$HOME}"
AUTH="${AGC_AUTH:-oauth2}"
RETRIES="${AGC_RETRIES:-3}"
T="${AGC_TIMEOUT:-1800}"

[[ -f "$SRC" ]] || { echo "no such file: $SRC" >&2; exit 1; }

cc()  { local t=$1; shift; HOME="$CLI_HOME" timeout "$t" "$CLI" --auth="$AUTH" "$@"; }
ccx() { printf '%s\n' "$2" | cc "$1" exec -s "$SESS" --timeout 600; }

WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

SIZE=$(stat -c %s "$SRC" 2>/dev/null || stat -f %z "$SRC")
MD5_LOCAL=$(md5sum "$SRC" | cut -d' ' -f1)
echo "source: $SRC ($SIZE bytes, md5 $MD5_LOCAL)"

split -b "$CHUNK" "$SRC" "$WORK/part_"
NCHUNK=$(find "$WORK" -name 'part_*' | wc -l)
echo "split into $NCHUNK chunk(s) of $CHUNK"

REMOTE_DIR="/content/.upload_$(basename "$DST").$$"
ccx 180 "import os; os.makedirs('$REMOTE_DIR', exist_ok=True); print('MKDIR_OK')" \
  || { echo "could not create remote chunk dir" >&2; exit 1; }

i=0
for f in "$WORK"/part_*; do
  i=$((i+1))
  b="$(basename "$f")"
  tries=0
  until cc "$T" upload -s "$SESS" "$f" "$REMOTE_DIR/$b" >/dev/null 2>&1; do
    tries=$((tries+1))
    if (( tries >= RETRIES )); then
      echo "FAILED to upload $b after $RETRIES attempts" >&2
      exit 1
    fi
    echo "  retry $tries for $b"
    sleep $(( tries * 10 ))
  done
  printf '  [%d/%d] %s\n' "$i" "$NCHUNK" "$b"
done

echo "reassembling on the VM"
OUT=$(ccx 900 "
import hashlib, os, subprocess
os.makedirs(os.path.dirname('$DST') or '/content', exist_ok=True)
subprocess.run('cat $REMOTE_DIR/part_* > $DST', shell=True, check=True)
h = hashlib.md5()
with open('$DST','rb') as fh:
    for blk in iter(lambda: fh.read(1<<22), b''):
        h.update(blk)
print('SIZE', os.path.getsize('$DST'))
print('MD5', h.hexdigest())
")
echo "$OUT"

if grep -q "$MD5_LOCAL" <<<"$OUT"; then
  echo "MD5 VERIFIED"
  ccx 180 "import shutil; shutil.rmtree('$REMOTE_DIR', ignore_errors=True); print('CLEANED')" >/dev/null
  echo "done: $DST"
  exit 0
fi

echo "MD5 MISMATCH - expected $MD5_LOCAL" >&2
echo "chunks left on the VM at $REMOTE_DIR for inspection" >&2
exit 1
