from pathlib import Path

from jitpi05.paths import artifact_root, gemini_credentials_path


def test_default_paths_are_relative_to_invocation_directory(monkeypatch) -> None:
    monkeypatch.delenv("JITPI05_GEMINI_CREDENTIALS", raising=False)
    assert artifact_root() == Path("artifacts")
    assert gemini_credentials_path() == Path(".secrets/gemini.json")


def test_gemini_credentials_environment_override(monkeypatch, tmp_path: Path) -> None:
    credential_path = tmp_path / "gemini.json"
    monkeypatch.setenv("JITPI05_GEMINI_CREDENTIALS", str(credential_path))
    assert gemini_credentials_path() == credential_path


def test_jitrl_free_methods_and_configuration() -> None:
    from jitpi05.config import (
        JITRL_FREE_ACTION_SIM_THRESHOLD,
        JITRL_FREE_CANDIDATES,
        JITRL_FREE_PLAN_TOKENS,
        JITRL_METHODS,
    )

    assert "jitrl-free" in JITRL_METHODS
    assert "static-free" in JITRL_METHODS
    assert JITRL_FREE_CANDIDATES == 5
    assert 0.0 < JITRL_FREE_ACTION_SIM_THRESHOLD <= 1.0
    assert JITRL_FREE_PLAN_TOKENS > 0
