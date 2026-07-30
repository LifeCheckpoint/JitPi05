from pathlib import Path

from jitpi05.jitrl.metrics import run_dir_for


def test_run_directory_isolated_by_task_method_and_seed() -> None:
    task = {"name": "libero_90_task79"}
    assert run_dir_for(Path("artifacts/test"), task, "jitrl", 17) == Path(
        "artifacts/test/libero_90_task79/jitrl/seed_17"
    )
