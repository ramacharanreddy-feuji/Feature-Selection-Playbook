"""The `fsp` command-line entry (wired in pyproject `[project.scripts]`)."""

from __future__ import annotations

import argparse

from .scaffold import scaffold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fsp", description="Feature Selection Playbook — deterministic screening tools."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="scaffold the guide docs + a starter driver into a folder")
    init.add_argument("dir", nargs="?", default=".", help="target folder (default: current)")
    init.add_argument("--overwrite", action="store_true", help="overwrite existing files")

    args = parser.parse_args(argv)
    if args.command == "init":
        written = scaffold(args.dir, overwrite=args.overwrite)
        if written:
            print(f"fsp: wrote {', '.join(written)} + runs/ to {args.dir}")
        else:
            print(f"fsp: already scaffolded in {args.dir} (use --overwrite to refresh)")
        print("Next: open the folder in Claude Code and fill in `analysis/screening.py` "
              "one part at a time (PLAYBOOK.md §3.1) — run a part, read the output, then the next.")
    return 0
