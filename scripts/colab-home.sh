#!/usr/bin/env bash
# Run the colab CLI under an ISOLATED home, with a bounded timeout.
#
# One session store per $HOME, and `colab sessions` deletes every local row the
# CALLING account cannot see. Two accounts (or two campaigns on different auth)
# sharing a home orphan each other's live VMs.
#
#   AGC_LANE=projA ./colab-home.sh --setup            # oauth2, copies token.json
#   AGC_LANE=projA AGC_AUTH=adc AGC_CLOUDSDK=~/.config/gcloud-projA ./colab-home.sh --setup
#   AGC_LANE=projA ./colab-home.sh sessions           # first numeric arg = timeout s
#   AGC_LANE=projA ./colab-home.sh 900 upload -s s1 big.tgz /content/big.tgz
#
# Copy to bin/<lane>-colab.sh with AGC_LANE baked in, so no invocation can use
# the wrong lane.
set -uo pipefail

LANE="${AGC_LANE:?set AGC_LANE to a short lane name, e.g. projA}"
AUTH="${AGC_AUTH:-oauth2}"
REAL_HOME="${AGC_REAL_HOME:-$HOME}"
LANE_HOME="${AGC_LANE_HOME:-$REAL_HOME/.colab_homes/$LANE}"
CLI="${AGC_CLI:-$REAL_HOME/.local/bin/colab}"
T_DEFAULT="${AGC_TIMEOUT:-300}"

if [[ ! -x "$CLI" ]]; then
  CLI="$(command -v colab || true)"
  [[ -z "$CLI" ]] && { echo "colab CLI not found; set AGC_CLI" >&2; exit 127; }
fi

if [[ "${1:-}" == "--setup" ]]; then
  mkdir -p "$LANE_HOME/.config/colab-cli"
  if [[ ! -f "$LANE_HOME/.config/colab-cli/sessions.json" ]]; then
    echo '{}' > "$LANE_HOME/.config/colab-cli/sessions.json"
    chmod 600 "$LANE_HOME/.config/colab-cli/sessions.json"
  fi
  if [[ "$AUTH" == "oauth2" ]]; then
    src="$REAL_HOME/.config/colab-cli/token.json"
    if [[ -f "$src" ]]; then
      cp "$src" "$LANE_HOME/.config/colab-cli/token.json"
      chmod 600 "$LANE_HOME/.config/colab-cli/token.json"
      echo "copied oauth token into $LANE_HOME"
    else
      echo "no token.json at $src - the first command will trigger the auth flow"
    fi
  fi
  echo "lane '$LANE' ready at $LANE_HOME (auth=$AUTH)"
  [[ -n "${AGC_CLOUDSDK:-}" ]] && echo "  will use CLOUDSDK_CONFIG=$AGC_CLOUDSDK"
  exit 0
fi

if [[ ! -d "$LANE_HOME" ]]; then
  echo "lane '$LANE' not set up; run with --setup first" >&2; exit 1
fi

T="$T_DEFAULT"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then T=$1; shift; fi

export HOME="$LANE_HOME"
# CLOUDSDK_CONFIG must be ABSOLUTE: HOME just moved, so a ~-relative path
# resolves inside the lane home. `--auth=adc` does not select an account; this
# variable does, and there is no `colab whoami` to check afterwards.
if [[ -n "${AGC_CLOUDSDK:-}" ]]; then
  export CLOUDSDK_CONFIG="$AGC_CLOUDSDK"
fi

# global flags MUST precede the subcommand: `colab sessions --auth=adc` fails
exec timeout "$T" "$CLI" --auth="$AUTH" "$@"
