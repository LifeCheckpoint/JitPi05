"""Compatibility shim; prefer ``uv run jitpi05-offline``."""

from jitpi05.cli.offline import main

if __name__ == "__main__":
    main()
