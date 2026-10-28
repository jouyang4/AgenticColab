#!/usr/bin/env python3
"""Rebuild a lost CLI handle for a live Colab assignment.

    python -m agenticcolab.adopt_orphan --list
    python -m agenticcolab.adopt_orphan --name recovered-1 [--endpoint gpu-a100-...]

A handle is only a row in sessions.json. Identity lives server-side and
`list_assignments()` mints fresh runtime_proxy_info (url + token) on demand, so
a discarded handle is REBUILT, not recovered.

A detached job keeps running on a VM that cannot currently be addressed, so
PROBE an adopted session before rebuilding anything: the work may be finished.

Rails:
  * refuses when >1 assignment is unbound unless --endpoint is given; on a
    shared account an unidentified assignment may be a colleague's job;
  * validates with SessionState(**row) BEFORE writing - sessions.json FAILS
    CLOSED, one bad row makes every session report "not found";
  * converts enum fields to str (pydantic v2 will not coerce them);
  * never copies kernel_id/session_id from a template - they point at a kernel
    that does not exist on the adopted machine;
  * backs up the store before writing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time


def _load_cli():
    try:
        from colab_cli.common import AuthProvider, SessionState, state
    except ImportError:
        sys.exit("colab_cli is not importable by this interpreter.\n"
                 "Use the python that has the CLI installed.")
    return state, AuthProvider, SessionState


def enum_str(v):
    """Convert an assignment enum to the string the CLI itself stores.

    Verified against a live sessions.json: rows carry enum NAMES - "TPU",
    "V5E1", "STANDARD". `.value` would write "2"/"0" (Shape and
    AssignmentVariant are int-valued), which validates and is quietly wrong.
    """
    name = getattr(v, "name", None)
    if name:
        return str(name)
    return str(getattr(v, "value", v))


def store_path() -> str:
    return os.path.expanduser(
        os.environ.get("COLAB_CLI_SESSIONS",
                       "~/.config/colab-cli/sessions.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", choices=["oauth2", "adc"], default="oauth2")
    ap.add_argument("--name", help="local label to create")
    ap.add_argument("--endpoint", help="assignment endpoint to adopt "
                                       "(required if >1 assignment is unbound)")
    ap.add_argument("--list", action="store_true", help="list assignments and exit")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing row with the same name")
    args = ap.parse_args()

    state, AuthProvider, SessionState = _load_cli()
    state.auth_provider = (AuthProvider.ADC if args.auth == "adc"
                           else AuthProvider.OAUTH2)

    assignments = state.client.list_assignments()
    existing = state.store.list()
    bound = {s.endpoint for s in existing.values()}
    unbound = [a for a in assignments if a.endpoint not in bound]

    if args.list or not args.name:
        print(f"{len(assignments)} assignment(s); {len(unbound)} unbound\n")
        for a in assignments:
            label = next((n for n, s in existing.items()
                          if s.endpoint == a.endpoint), None)
            print(f"  {a.endpoint}")
            print(f"    accel={enum_str(a.accelerator)} "
                  f"shape={enum_str(a.machine_shape)} "
                  f"variant={enum_str(a.variant)}")
            print(f"    local label: {label or '(none - unbound)'}")
            print(f"    token expires in {a.runtime_proxy_info.token_expires_in_seconds}s")
        if not args.name:
            print("\nPass --name LABEL to adopt one.")
        return 0

    # pick the target
    if args.endpoint:
        target = next((a for a in assignments if a.endpoint == args.endpoint), None)
        if target is None:
            sys.exit(f"endpoint {args.endpoint} is not in the live assignment list")
    else:
        if len(unbound) == 0:
            sys.exit("no unbound assignments: every live assignment already has a "
                     "local handle. Nothing to adopt.")
        if len(unbound) > 1:
            print("REFUSING TO GUESS: more than one assignment is unbound.",
                  file=sys.stderr)
            for a in unbound:
                print(f"  {a.endpoint}  ({enum_str(a.accelerator)})", file=sys.stderr)
            sys.exit("Re-run with --endpoint <the one you can prove is yours>.")
        target = unbound[0]

    if args.name in existing and not args.force:
        sys.exit(f"a row named '{args.name}' already exists; use --force to replace")

    info = target.runtime_proxy_info
    row = {
        "name": args.name,
        "token": info.token,
        "url": info.url,
        "endpoint": target.endpoint,
        "token_expires_at": info.expires_at(),
        "variant": enum_str(target.variant),
        "accelerator": enum_str(target.accelerator),
        "machine_shape": enum_str(target.machine_shape),
    }

    # validate BEFORE writing: the store fails closed on a bad row
    try:
        model = SessionState(**row)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"refusing to write an invalid row: {type(exc).__name__}: {exc}")

    # back up the store
    p = store_path()
    if os.path.isfile(p):
        bak = f"{p}.bak.{int(time.time())}"
        shutil.copy2(p, bak)
        print(f"backed up {p} -> {bak}")

    state.store.add(model)
    print(f"adopted {target.endpoint} as '{args.name}'")
    print(f"  accel={row['accelerator']} shape={row['machine_shape']}")
    print()
    print("PROBE before rebuilding anything - the job may have finished:")
    print(f"  colab status -s {args.name}")
    print(f"  echo 'import os;print(os.listdir(\"/content\"))' | colab exec -s {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
