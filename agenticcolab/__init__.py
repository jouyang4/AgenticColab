"""Tools for driving Google Colab from a CLI or an agent.

Each module is a standalone script as well as an importable module, so a single
file can be copied into another project with no dependency on this package.

    python -m agenticcolab.audit           # prune-safe assignment audit
    python -m agenticcolab.adopt_orphan    # rebuild a lost session handle
    python -m agenticcolab.reap_orphans    # release assignments the CLI can't target
    python -m agenticcolab.refresh_tokens  # legacy token refresh for CLI <= 0.6.0
    python  agenticcolab/probe.py          # runs ON a Colab VM: inventory + benchmarks

All but `probe` need an interpreter that can import `colab_cli`. Under a
`uv tool` install that is not system python; use e.g.
`~/.local/share/uv/tools/google-colab-cli/bin/python`.

`probe` is stdlib-only: it runs on every Colab runtime image with no install.
"""

__all__ = ["adopt_orphan", "audit", "probe", "reap_orphans", "refresh_tokens"]
