#!/usr/bin/env python3
"""LEGACY: manually refresh runtime-proxy tokens in a CLI session store.

    python -m agenticcolab.refresh_tokens --prefix myjob-
    python -m agenticcolab.refresh_tokens --all --auth adc
    python -m agenticcolab.refresh_tokens --loop 600 --prefix myjob-

YOU PROBABLY DO NOT NEED THIS - fixed upstream in 0.7.x, where
`State.get_session` refreshes the proxy token within TOKEN_REFRESH_MARGIN
(5 min) of expiry, on every session-targeting command. Upgrade instead
(`colab version` >= 0.7.x).

For environments pinned to <=0.6.0, which stored the token minted at
`colab new` (TTL 3600 s) and never refreshed it: ~1 h later every
exec/upload/download against a healthy VM failed 401/404 and the CLI often
discarded the handle, while the VM itself stayed up.

Mechanism: `client.list_assignments()` mints fresh runtime_proxy_info (url +
token, exp = now + 3600) per live assignment; write it back into each row.

CRITICAL: reads the store via expanduser, so it MUST run with the same HOME
(and CLOUDSDK_CONFIG, for adc) as the campaign it refreshes. Wrong HOME ->
refreshes nothing, exits 0. "refreshed none" is a FAILURE, not "fine".

    HOME=~/.colab_homes/projA \
    CLOUDSDK_CONFIG=~/.config/gcloud-projA \
    /path/to/colab-cli/bin/python -m agenticcolab.refresh_tokens --prefix projA-
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _load_cli():
    try:
        from colab_cli.common import AuthProvider, SessionState, state
    except ImportError:
        sys.exit("colab_cli is not importable by this interpreter.\n"
                 "Use the python that has the CLI installed, e.g.\n"
                 "  ~/.local/share/uv/tools/google-colab-cli/bin/python")
    return state, AuthProvider, SessionState


def enum_str(v):
    """Prefer .name - sessions.json stores enum NAMES ("V5E1", "STANDARD"), and
    Shape/AssignmentVariant are int-valued so .value would write "0"/"2"."""
    name = getattr(v, "name", None)
    return str(name) if name else str(getattr(v, "value", v))


def refresh_once(state, SessionState, prefix, do_all, verbose=True):
    sessions = state.store.list()
    if not sessions:
        print("REFRESH none  (store is EMPTY - wrong HOME? this is a FAILURE)")
        return 0, []

    targets = {n: s for n, s in sessions.items()
               if do_all or (prefix and n.startswith(prefix))}
    if not targets:
        print(f"REFRESH none  (no session matches prefix {prefix!r}; "
              f"store has: {sorted(sessions)}) - this is a FAILURE")
        return 0, []

    by_endpoint = {a.endpoint: a for a in state.client.list_assignments()}

    ok, gone = [], []
    for name, s in targets.items():
        a = by_endpoint.get(s.endpoint)
        if a is None:
            gone.append(name)
            continue
        info = a.runtime_proxy_info
        row = s.model_dump() if hasattr(s, "model_dump") else dict(s.__dict__)
        row.update({
            "token": info.token,
            "url": info.url,
            "token_expires_at": info.expires_at(),
        })
        row.setdefault("variant", enum_str(a.variant))
        row.setdefault("accelerator", enum_str(a.accelerator))
        try:
            # Validate BEFORE writing: the store FAILS CLOSED, so one bad row
            # makes the CLI report every session as "not found".
            model = SessionState(**row)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {name}: would write an invalid row "
                  f"({type(exc).__name__}: {str(exc).splitlines()[0][:100]})")
            continue
        state.store.add(model)
        ok.append(name)

    if ok:
        print(f"REFRESH ok={ok}")
    else:
        print("REFRESH none  (nothing refreshed) - treat as FAILURE")
    if gone:
        print(f"  not in live assignments (VM gone): {gone}")
    return len(ok), ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", choices=["oauth2", "adc"], default="oauth2")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prefix", help="only refresh sessions whose label starts with this")
    g.add_argument("--all", action="store_true",
                   help="refresh every session in THIS store")
    ap.add_argument("--loop", type=int, metavar="SECONDS", default=0,
                    help="keep refreshing every N seconds (use < 3000)")
    args = ap.parse_args()

    state, AuthProvider, SessionState = _load_cli()
    state.auth_provider = (AuthProvider.ADC if args.auth == "adc"
                           else AuthProvider.OAUTH2)

    print(f"HOME={os.environ.get('HOME')}")
    print(f"CLOUDSDK_CONFIG={os.environ.get('CLOUDSDK_CONFIG', '(unset)')}")

    if not args.loop:
        n, _ = refresh_once(state, SessionState, args.prefix, args.all)
        return 0 if n else 1

    while True:
        try:
            refresh_once(state, SessionState, args.prefix, args.all)
        except Exception as exc:  # noqa: BLE001
            print(f"refresh error: {type(exc).__name__}: {exc}", file=sys.stderr)
        time.sleep(args.loop)


if __name__ == "__main__":
    sys.exit(main())
