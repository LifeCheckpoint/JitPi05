"""Compatibility shim; prefer ``uv run jitpi05-eval-jitrl``."""

from jitpi05.cli.jitrl import main

if __name__ == "__main__":
    raise SystemExit(main())
