#!/usr/bin/env python3
"""Release Colab assignments the CLI cannot target.

    python -m agenticcolab.reap_orphans --list
    python -m agenticcolab.reap_orphans --endpoint gpu-t4-s-... --yes
    python -m agenticcolab.reap_orphans --label-prefix myjob- --yes

`colab stop` resolves LOCAL LABELS only, so an assignment whose handle has been
lost cannot be stopped with the CLI at all. The client API unassigns by
endpoint, instantly, which clears a 412 TooManyAssignmentsError without waiting
out an idle timeout.

SAFETY: no "release everything" mode, on purpose - on a shared account
loop-and-unassign kills another user's job. Requires an explicit --endpoint or
a --label-prefix matching a local row, and confirms unless --yes.

A detached job keeps running on a VM that cannot currently be addressed.
Consider adopting it instead (adopt_orphan.py) and checking whether the work
finished.
"""
from __future__ import annotations

import argparse
import sys


def _load_cli():
    try:
        from colab_cli.common import AuthProvider, state
    except ImportError:
        sys.exit("colab_cli is not importable by this interpreter.")
    return state, AuthProvider


def enum_str(v):
    """Prefer .name - Shape/AssignmentVariant are int-valued, so .value is '0'."""
    name = getattr(v, "name", None)
    return str(name) if name else str(getattr(v, "value", v))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", choices=["oauth2", "adc"], default="oauth2")
    ap.add_argument("--endpoint", action="append", default=[],
                    help="endpoint to release (repeatable)")
    ap.add_argument("--label-prefix",
                    help="release assignments whose LOCAL label starts with this")
    ap.add_argument("--list", action="store_true", help="list and exit")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    args = ap.parse_args()

    state, AuthProvider = _load_cli()
    state.auth_provider = (AuthProvider.ADC if args.auth == "adc"
                           else AuthProvider.OAUTH2)

    assignments = state.client.list_assignments()
    local = state.store.list()
    label_of = {}
    for name, s in local.items():
        label_of[s.endpoint] = name

    if args.list or (not args.endpoint and not args.label_prefix):
        print(f"{len(assignments)} live assignment(s)\n")
        for a in assignments:
            print(f"  {a.endpoint}")
            print(f"    accel={enum_str(a.accelerator)} "
                  f"shape={enum_str(a.machine_shape)}")
            print(f"    local label: {label_of.get(a.endpoint) or '(none)'}")
        if not args.endpoint and not args.label_prefix:
            print("\nNothing selected. Pass --endpoint or --label-prefix.")
            print("There is deliberately no 'release all' mode.")
        return 0

    targets = []
    for a in assignments:
        if a.endpoint in args.endpoint:
            targets.append((a, "explicit --endpoint"))
        elif args.label_prefix:
            lbl = label_of.get(a.endpoint)
            if lbl and lbl.startswith(args.label_prefix):
                targets.append((a, f"label '{lbl}' matches prefix"))

    unknown = [e for e in args.endpoint
               if e not in {a.endpoint for a in assignments}]
    for e in unknown:
        print(f"note: {e} is not in the live assignment list (already gone?)")

    if not targets:
        print("no live assignments matched; nothing to do")
        return 0

    print("about to RELEASE:")
    for a, why in targets:
        print(f"  {a.endpoint}  ({enum_str(a.accelerator)})  <- {why}")
    print("\nThis destroys the VM and everything on /content that you have not "
          "already downloaded.")

    if not args.yes:
        try:
            ans = input("type 'release' to proceed: ").strip()
        except EOFError:
            ans = ""
        if ans != "release":
            print("aborted")
            return 1

    rc = 0
    for a, _ in targets:
        try:
            state.client.unassign(a.endpoint)
            print(f"released {a.endpoint}")
            lbl = label_of.get(a.endpoint)
            if lbl:
                try:
                    state.store.remove(lbl)
                    print(f"  removed local row '{lbl}'")
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {a.endpoint}: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1

    try:
        cu = state.client.get_consumption_user_info()
        print(f"\nconsumption now: {cu}")
    except Exception:  # noqa: BLE001
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
