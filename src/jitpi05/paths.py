from __future__ import annotations

import os
from pathlib import Path


def artifact_root() -> Path:
    """Return the default artifact root relative to the invocation directory."""
    return Path("artifacts")


def gemini_credentials_path() -> Path:
    """Resolve Gemini credentials from the environment or the local secret file."""
    configured = os.environ.get("JITPI05_GEMINI_CREDENTIALS")
    return Path(configured).expanduser() if configured else Path(".secrets/gemini.json")
