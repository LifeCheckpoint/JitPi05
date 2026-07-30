"""Compatibility shim; prefer ``uv run jitpi05-eval-sim``."""

from jitpi05.cli.simulation import main

if __name__ == "__main__":
    main()
