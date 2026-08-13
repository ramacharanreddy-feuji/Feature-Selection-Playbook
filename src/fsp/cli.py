"""The `fsp` command-line entry (wired in pyproject `[project.scripts]`)."""

from __future__ import annotations

import argparse

from .scaffold import scaffold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fsp", description="Feature Selection Playbook — deterministic screening tools."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="scaffold the playbook docs into a project folder")
    init.add_argument("dir", nargs="?", default=".", help="target folder (default: current)")
    init.add_argument("--overwrite", action="store_true", help="overwrite existing docs")

    args = parser.parse_args(argv)
    if args.command == "init":
        written = scaffold(args.dir, overwrite=args.overwrite)
        if written:
            print(f"fsp: wrote {', '.join(written)} + runs/ to {args.dir}")
        else:
            print(f"fsp: docs already present in {args.dir} (use --overwrite to refresh)")
        print("Next: open the folder in Claude Code, then `import fsp` in your notebook.")
    return 0
