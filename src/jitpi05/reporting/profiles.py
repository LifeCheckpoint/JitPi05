from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportProfile:
    name: str
    artifact_dir: Path
    report_dir: Path
    task_order: tuple[str, ...]
    task_labels: dict[str, str]


PROFILES = {
    "signed-reward": ReportProfile(
        name="signed-reward",
        artifact_dir=Path("artifacts/jitrl_eval_libero90_5tasks_seed17_qwen4b_beta040"),
        report_dir=Path("reports/signed_reward"),
        task_order=(
            "libero_90_task19",
            "libero_90_task27",
            "libero_90_task60",
            "libero_90_task62",
            "libero_90_task79",
        ),
        task_labels={
            "libero_90_task19": "T19 Moka pot",
            "libero_90_task27": "T27 Wine bottle",
            "libero_90_task60": "T60 Black bowl",
            "libero_90_task62": "T62 Salad dressing",
            "libero_90_task79": "T79 Book",
        },
    ),
    "positive-reward": ReportProfile(
        name="positive-reward",
        artifact_dir=Path(
            "artifacts/jitrl_eval_libero90_mid5_seed17_qwen4b_positive_v3"
        ),
        report_dir=Path("reports/positive_reward"),
        task_order=(
            "libero_90_task18",
            "libero_90_task53",
            "libero_90_task59",
            "libero_90_task69",
            "libero_90_task79",
        ),
        task_labels={
            "libero_90_task18": "T18 Frying pan",
            "libero_90_task53": "T53 Orange juice",
            "libero_90_task59": "T59 Tomato sauce",
            "libero_90_task69": "T69 Pudding",
            "libero_90_task79": "T79 Book",
        },
    ),
}
