#!/usr/bin/env python3
"""Prune-safe audit of Colab assignments and local session stores.

    python -m agenticcolab.audit [--auth adc] [--json]

`colab sessions` is not read-only: it deletes local rows the calling account
cannot see, orphaning another account's live VMs. So this reads assignments
through the client API and enumerates local stores directly.

Avoids the two ways audits get it wrong:
  * `--config <path>` campaigns keep state in THAT file, so the default store
    reads `{}` while everything is healthy -> harvest `--config` paths from
    running processes and read those stores too.
  * grepping `ps` once per endpoint matches the audit's own pipeline and returns
    a false negative -> build the set of endpoints mentioned by other processes
    ONCE, then test membership.

Reports; stops nothing. Releasing an unidentified assignment is a human
decision, especially on a shared account.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any

DEFAULT_STORE = "~/.config/colab-cli/sessions.json"


def _load_cli():
    try:
        from colab_cli.common import AuthProvider, state
    except ImportError:
        sys.exit("colab_cli is not importable by this interpreter.\n"
                 "Run this with the python that has the CLI installed, e.g.\n"
                 "  ~/.local/share/uv/tools/google-colab-cli/bin/python -m agenticcolab.audit")
    return state, AuthProvider


def ps_lines() -> list[str]:
    """Full command lines of all processes, once."""
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return []
    return [l.strip() for l in out.splitlines() if l.strip()]


def harvest_config_paths(lines: list[str]) -> list[str]:
    """Session stores other campaigns are using, from their own process args."""
    found: set[str] = set()
    for line in lines:
        for m in re.finditer(r"--config[= ]([^\s]+)", line):
            found.add(m.group(1))
        # an isolated CLI HOME implies a store underneath it
        for m in re.finditer(r"HOME=([^\s]+)", line):
            found.add(os.path.join(m.group(1), ".config/colab-cli/sessions.json"))
    return sorted(found)


def read_store(path: str) -> dict[str, Any]:
    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        # itself a finding: the CLI fails CLOSED, reporting EVERY session as
        # "not found" when one row is malformed
        return {"__unparseable__": str(exc)}


def validate_store(path: str) -> list[str]:
    """Flag rows the CLI itself would choke on."""
    try:
        from colab_cli.common import SessionState
    except ImportError:
        return []
    problems = []
    data = read_store(path)
    if "__unparseable__" in data:
        return [f"{path}: FILE DOES NOT PARSE ({data['__unparseable__']})"]
    for name, row in data.items():
        if not isinstance(row, dict):
            problems.append(f"{path}:{name}: row is not an object")
            continue
        try:
            SessionState(**row)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path}:{name}: invalid row -> {type(exc).__name__}: "
                            f"{str(exc).splitlines()[0][:120]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", choices=["oauth2", "adc"], default="oauth2")
    ap.add_argument("--store", action="append", default=[],
                    help="extra session store to inspect (repeatable)")
    ap.add_argument("--max-assignments", type=int, default=2,
                    help="expected concurrency ceiling (default 2)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    state, AuthProvider = _load_cli()
    state.auth_provider = (AuthProvider.ADC if args.auth == "adc"
                           else AuthProvider.OAUTH2)

    lines = ps_lines()
    stores = [DEFAULT_STORE] + harvest_config_paths(lines) + args.store
    # dedupe, preserve order
    seen: set[str] = set()
    stores = [s for s in stores if not (os.path.expanduser(s) in seen
                                        or seen.add(os.path.expanduser(s)))]

    # endpoint/label strings mentioned by OTHER processes, built once
    other_proc_text = "\n".join(l for l in lines
                                if "agenticcolab.audit" not in l and "audit.py" not in l)

    local: dict[str, dict[str, Any]] = {}   # endpoint -> {store, label}
    store_report = []
    for s in stores:
        data = read_store(s)
        rows = {k: v for k, v in data.items() if k != "__unparseable__"}
        store_report.append({"store": s, "exists": os.path.isfile(os.path.expanduser(s)),
                             "rows": len(rows),
                             "unparseable": "__unparseable__" in data,
                             "labels": sorted(rows)})
        for label, row in rows.items():
            ep = row.get("endpoint") if isinstance(row, dict) else None
            if ep:
                local.setdefault(ep, {"store": s, "label": label})

    try:
        assignments = state.client.list_assignments()
    except SystemExit:
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"could not list assignments: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        cu = state.client.get_consumption_user_info()
    except Exception:  # noqa: BLE001
        cu = None

    def enum_str(v):
        """Prefer .name: these enums carry ints (Shape.STANDARD == 0)."""
        name = getattr(v, "name", None)
        if name:
            return str(name)
        return str(getattr(v, "value", v))

    rows = []
    for a in assignments:
        ep = a.endpoint
        known = local.get(ep)
        mentioned = ep in other_proc_text
        rows.append({
            "endpoint": ep,
            "accelerator": enum_str(a.accelerator),
            "machine_shape": enum_str(a.machine_shape),
            "variant": enum_str(a.variant),
            "local_label": known["label"] if known else None,
            "local_store": known["store"] if known else None,
            "mentioned_by_a_process": mentioned,
            "token_expires_in_s": getattr(a.runtime_proxy_info,
                                          "token_expires_in_seconds", None),
            "classification": ("bound" if known else
                               "unbound_but_referenced" if mentioned else
                               "UNBOUND"),
        })

    problems = []
    for s in stores:
        problems += validate_store(s)
    if len(assignments) > args.max_assignments:
        problems.append(f"{len(assignments)} assignments exceed the expected "
                        f"ceiling of {args.max_assignments}")

    report = {
        "auth": args.auth,
        "assignment_count": len(assignments),
        "assignments": rows,
        "stores": store_report,
        "consumption": (None if cu is None else {
            k: v for k, v in (cu.model_dump() if hasattr(cu, "model_dump")
                              else dict(cu)).items()}),
        "problems": problems,
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(f"auth={args.auth}   live assignments: {len(assignments)}")
    if cu is not None:
        print(f"consumption: {report['consumption']}")
    print()
    print("ASSIGNMENTS")
    if not rows:
        print("  (none)")
    for r in rows:
        tag = {"bound": "ok", "unbound_but_referenced": "referenced",
               "UNBOUND": "UNBOUND"}[r["classification"]]
        print(f"  [{tag:<11}] {r['endpoint']}")
        print(f"      {r['accelerator']}/{r['machine_shape']}/{r['variant']}"
              f"   token_exp_in={r['token_expires_in_s']}s")
        if r["local_label"]:
            print(f"      label '{r['local_label']}' in {r['local_store']}")
        elif r["mentioned_by_a_process"]:
            print("      no local row, but a running process references this "
                  "endpoint - NOT an orphan")
        else:
            print("      no local row and no process references it")
    print()
    print("SESSION STORES")
    for s in store_report:
        flag = " [UNPARSEABLE]" if s["unparseable"] else ""
        print(f"  {s['store']}  exists={s['exists']}  rows={s['rows']}{flag}")
        for lbl in s["labels"]:
            print(f"      - {lbl}")
    print()
    if problems:
        print("PROBLEMS")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("no problems detected")
    print()
    print("Stops nothing. UNBOUND is not proof of an orphan - on a shared account it")
    print("may be a colleague's job. Confirm ownership before releasing anything")
    print("(agenticcolab/reap_orphans.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
