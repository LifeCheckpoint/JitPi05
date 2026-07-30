from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from jitpi05.reporting.profiles import PROFILES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate JitPi05 report assets.")
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = PROFILES[args.profile]
    if args.profile == "signed-reward":
        from jitpi05.reporting import signed as engine
    else:
        from jitpi05.reporting import positive as engine

    engine.configure(
        artifact_dir=args.artifact_dir or profile.artifact_dir,
        report_dir=args.report_dir or profile.report_dir,
    )
    engine.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
