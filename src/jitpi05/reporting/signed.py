from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jitpi05.reporting.profiles import PROFILES

PROFILE = PROFILES["signed-reward"]
REPORT_DIR = PROFILE.report_dir
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
ARTIFACT_DIR = PROFILE.artifact_dir
SUMMARY_PATH = ARTIFACT_DIR / "summary.json"

TASK_ORDER = PROFILE.task_order
METHOD_ORDER = ("jitrl", "static")
TASK_LABELS = PROFILE.task_labels
METHOD_LABELS = {"jitrl": "JitRL", "static": "Static"}
METHOD_COLORS = {"jitrl": "#3264A8", "static": "#E07A2F"}
MECHANISM_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")


def configure(*, artifact_dir: Path, report_dir: Path) -> None:
    global ARTIFACT_DIR, FIGURE_DIR, REPORT_DIR, SUMMARY_PATH, TABLE_DIR
    ARTIFACT_DIR = artifact_dir
    REPORT_DIR = report_dir
    FIGURE_DIR = REPORT_DIR / "figures"
    TABLE_DIR = REPORT_DIR / "tables"
    SUMMARY_PATH = ARTIFACT_DIR / "summary.json"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_dir(task: str, method: str) -> Path:
    return ARTIFACT_DIR / task / method / "seed_17"


def load_records() -> tuple[
    dict[str, Any], dict[str, dict[str, list]], dict[str, dict[str, dict]]
]:
    summary = read_json(SUMMARY_PATH)
    episodes: dict[str, dict[str, list]] = {}
    metrics: dict[str, dict[str, dict]] = {}
    for task in TASK_ORDER:
        episodes[task] = {}
        metrics[task] = {}
        for method in METHOD_ORDER:
            episodes[task][method] = read_json(run_dir(task, method) / "episodes.json")
            metrics[task][method] = read_json(run_dir(task, method) / "metrics.json")
    return summary, episodes, metrics


def exact_mcnemar_two_sided(jitrl_only: int, static_only: int) -> float:
    discordant = jitrl_only + static_only
    if discordant == 0:
        return 1.0
    observed_probability = math.comb(discordant, jitrl_only) / (2**discordant)
    probability = sum(
        math.comb(discordant, index) / (2**discordant)
        for index in range(discordant + 1)
        if math.comb(discordant, index) / (2**discordant)
        <= observed_probability + 1e-15
    )
    return min(1.0, probability)


def success_vector(rows: list[dict[str, Any]]) -> list[bool]:
    return [bool(row["success"]) for row in rows]


def rate(values: list[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def cumulative_rate(values: list[bool]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.cumsum(array) / np.arange(1, len(array) + 1)


def moving_rate(values: list[bool], window: int = 5) -> np.ndarray:
    return np.asarray(
        [rate(values[max(0, end - window) : end]) for end in range(1, len(values) + 1)],
        dtype=np.float64,
    )


def derive_statistics(
    summary: dict[str, Any],
    episodes: dict[str, dict[str, list]],
    metrics: dict[str, dict[str, dict]],
) -> dict[str, Any]:
    per_task: list[dict[str, Any]] = []
    paired_total = Counter()
    pooled_successes = Counter()
    pooled_success_steps: dict[str, list[int]] = {method: [] for method in METHOD_ORDER}

    for task in TASK_ORDER:
        method_stats: dict[str, dict[str, Any]] = {}
        for method in METHOD_ORDER:
            rows = episodes[task][method]
            successes = success_vector(rows)
            successful_steps = [
                int(row["steps"]) for row in rows if bool(row["success"])
            ]
            pooled_successes[method] += sum(successes)
            pooled_success_steps[method].extend(successful_steps)
            method_stats[method] = {
                "success_count": sum(successes),
                "success_rate": rate(successes),
                "first_5_success_rate": rate(successes[:5]),
                "final_10_success_rate": rate(successes[-10:]),
                "mean_success_steps": mean(successful_steps)
                if successful_steps
                else None,
                "success_sequence": "".join(
                    "1" if value else "0" for value in successes
                ),
            }

        pairs = list(
            zip(
                success_vector(episodes[task]["jitrl"]),
                success_vector(episodes[task]["static"]),
            )
        )
        paired = {
            "both_success": sum(jitrl and static for jitrl, static in pairs),
            "jitrl_only": sum(jitrl and not static for jitrl, static in pairs),
            "static_only": sum(static and not jitrl for jitrl, static in pairs),
            "neither_success": sum(not jitrl and not static for jitrl, static in pairs),
        }
        paired["exact_mcnemar_p"] = exact_mcnemar_two_sided(
            paired["jitrl_only"], paired["static_only"]
        )
        paired_total.update(
            {key: value for key, value in paired.items() if key != "exact_mcnemar_p"}
        )
        per_task.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "description": summary["tasks"][task]["task"]["description"],
                "jitrl": method_stats["jitrl"],
                "static": method_stats["static"],
                "success_rate_delta": (
                    method_stats["jitrl"]["success_rate"]
                    - method_stats["static"]["success_rate"]
                ),
                "paired": paired,
            }
        )

    paired_total["exact_mcnemar_p"] = exact_mcnemar_two_sided(
        paired_total["jitrl_only"], paired_total["static_only"]
    )
    step_reduction = summary["task_macro"]["paired_differences"][
        "success_step_reduction"
    ]
    success_rate_delta = summary["task_macro"]["paired_differences"]["success_rate"]
    final_10_delta = summary["task_macro"]["paired_differences"][
        "final_10_success_rate"
    ]
    total_episodes_per_method = sum(len(episodes[task]["jitrl"]) for task in TASK_ORDER)
    return {
        "configuration": summary["configuration"],
        "per_task": per_task,
        "aggregate": {
            "episodes_per_method": total_episodes_per_method,
            "jitrl_success_count": pooled_successes["jitrl"],
            "static_success_count": pooled_successes["static"],
            "jitrl_success_rate": pooled_successes["jitrl"] / total_episodes_per_method,
            "static_success_rate": pooled_successes["static"]
            / total_episodes_per_method,
            "success_rate_delta": (
                pooled_successes["jitrl"] - pooled_successes["static"]
            )
            / total_episodes_per_method,
            "jitrl_mean_success_steps": mean(pooled_success_steps["jitrl"]),
            "static_mean_success_steps": mean(pooled_success_steps["static"]),
            "success_step_reduction": step_reduction["mean"],
            "success_step_reduction_ci": step_reduction[
                "descriptive_task_bootstrap_ci"
            ],
            "paired": dict(paired_total),
            "success_rate_task_macro_delta": success_rate_delta["mean"],
            "success_rate_task_macro_delta_ci": success_rate_delta[
                "descriptive_task_bootstrap_ci"
            ],
            "final_10_task_macro_delta": final_10_delta["mean"],
            "final_10_task_macro_delta_ci": final_10_delta[
                "descriptive_task_bootstrap_ci"
            ],
        },
    }


def export_tables(
    derived: dict[str, Any],
    episodes: dict[str, dict[str, list]],
    metrics: dict[str, dict[str, dict]],
) -> None:
    per_task_rows: list[dict[str, Any]] = []
    for record in derived["per_task"]:
        per_task_rows.append(
            {
                "task": record["task"],
                "description": record["description"],
                "jitrl_successes": record["jitrl"]["success_count"],
                "static_successes": record["static"]["success_count"],
                "jitrl_success_rate": record["jitrl"]["success_rate"],
                "static_success_rate": record["static"]["success_rate"],
                "delta": record["success_rate_delta"],
                "jitrl_first_5": record["jitrl"]["first_5_success_rate"],
                "static_first_5": record["static"]["first_5_success_rate"],
                "jitrl_final_10": record["jitrl"]["final_10_success_rate"],
                "static_final_10": record["static"]["final_10_success_rate"],
                "jitrl_mean_success_steps": record["jitrl"]["mean_success_steps"],
                "static_mean_success_steps": record["static"]["mean_success_steps"],
                "both_success": record["paired"]["both_success"],
                "jitrl_only": record["paired"]["jitrl_only"],
                "static_only": record["paired"]["static_only"],
                "neither_success": record["paired"]["neither_success"],
                "exact_mcnemar_p": record["paired"]["exact_mcnemar_p"],
                "jitrl_sequence": record["jitrl"]["success_sequence"],
                "static_sequence": record["static"]["success_sequence"],
            }
        )
    write_csv(TABLE_DIR / "per_task_results.csv", per_task_rows)

    mechanism_fields = (
        "neighbor_action_coverage_rate",
        "nonzero_advantage_rate",
        "choice_change_rate",
        "augmentation_choice_change_rate",
        "ucb_application_rate",
        "memory_only_candidate_rate",
        "memory_only_selection_rate",
        "mean_evaluator_score",
        "negative_evaluator_score_rate",
        "positive_evaluator_score_rate",
        "mean_absolute_logit_shift",
        "mean_neighbor_count",
        "memory_size",
    )
    mechanism_rows = []
    for task in TASK_ORDER:
        row = {"task": task, "task_label": TASK_LABELS[task]}
        row.update({field: metrics[task]["jitrl"][field] for field in mechanism_fields})
        mechanism_rows.append(row)
    write_csv(TABLE_DIR / "jitrl_mechanism_metrics.csv", mechanism_rows)

    episode_rows = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            for episode in episodes[task][method]:
                episode_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "episode_index": episode["episode_index"],
                        "success": int(bool(episode["success"])),
                        "steps": episode["steps"],
                        "high_level_steps": episode["high_level_step_count"],
                        "low_level_chunks": episode["low_level_chunk_count"],
                    }
                )
    write_csv(TABLE_DIR / "episode_results.csv", episode_rows)

    for task, episode_index in (
        ("libero_90_task60", 1),
        ("libero_90_task79", 14),
    ):
        case_rows = []
        for method in METHOD_ORDER:
            episode = episodes[task][method][episode_index]
            for chunk in episode["chunks"]:
                case_rows.append(
                    {
                        "task": task,
                        "episode": episode_index,
                        "method": method,
                        "success": int(bool(episode["success"])),
                        "chunk": chunk["chunk_index"],
                        "action_start_step": chunk["action_start_step"],
                        "action_end_step": chunk["action_end_step"],
                        "source": chunk.get("selected_source"),
                        "semantic_key": chunk["selected_candidate"]["semantic_key"],
                        "action_text": chunk["selected_candidate"]["text"],
                        "evaluator_score": chunk.get("evaluator_score"),
                        "step_reward": chunk.get("step_reward"),
                        "discounted_return": chunk.get("return"),
                        "evaluator_result": chunk.get("evaluator", {}).get("result"),
                    }
                )
        write_csv(
            TABLE_DIR / f"case_{task}_episode_{episode_index}_chunks.csv",
            case_rows,
        )


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_success_rates(derived: dict[str, Any]) -> None:
    x = np.arange(len(TASK_ORDER))
    width = 0.36
    jitrl = [record["jitrl"]["success_rate"] * 100 for record in derived["per_task"]]
    static = [record["static"]["success_rate"] * 100 for record in derived["per_task"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    left = ax.bar(
        x - width / 2,
        jitrl,
        width,
        label="JitRL",
        color=METHOD_COLORS["jitrl"],
    )
    right = ax.bar(
        x + width / 2,
        static,
        width,
        label="Static",
        color=METHOD_COLORS["static"],
    )
    ax.bar_label(left, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(right, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Environment success rate (%)")
    ax.set_title("Per-task success rate (15 episodes per method)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", ncol=2)
    aggregate = derived["aggregate"]
    ax.text(
        0.5,
        -0.18,
        (
            f"Aggregate: JitRL {aggregate['jitrl_success_count']}/75 = "
            f"{aggregate['jitrl_success_rate']:.1%}; Static "
            f"{aggregate['static_success_count']}/75 = "
            f"{aggregate['static_success_rate']:.1%}"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    save_figure(fig, "fig01_success_rate_by_task.png")


def plot_episode_outcomes(episodes: dict[str, dict[str, list]]) -> None:
    rows = []
    labels = []
    for task in TASK_ORDER:
        for method in METHOD_ORDER:
            rows.append(
                [int(value) for value in success_vector(episodes[task][method])]
            )
            labels.append(f"{TASK_LABELS[task]} | {METHOD_LABELS[method]}")
    matrix = np.asarray(rows, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["#E6E8EB", "#3A9D5D"]),
        vmin=0,
        vmax=1,
    )
    del image
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(15), np.arange(1, 16))
    ax.set_xlabel("Episode / initialization-state index")
    ax.set_title("Episode outcomes (green = success, gray = failure)")
    ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    save_figure(fig, "fig02_episode_outcomes.png")


def plot_online_curves(episodes: dict[str, dict[str, list]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.0), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    x = np.arange(1, 16)
    for index, task in enumerate(TASK_ORDER):
        ax = axes_flat[index]
        for method in METHOD_ORDER:
            values = success_vector(episodes[task][method])
            ax.plot(
                x,
                cumulative_rate(values) * 100,
                color=METHOD_COLORS[method],
                linewidth=2.1,
                marker="o",
                markersize=3.2,
                label=f"{METHOD_LABELS[method]} cumulative",
            )
            ax.plot(
                x,
                moving_rate(values, 5) * 100,
                color=METHOD_COLORS[method],
                linewidth=1.2,
                linestyle="--",
                alpha=0.65,
                label=f"{METHOD_LABELS[method]} moving-5",
            )
        ax.set_title(TASK_LABELS[task])
        ax.set_xlim(1, 15)
        ax.set_ylim(-4, 104)
        ax.grid(alpha=0.25)
        ax.set_ylabel("Success rate (%)")
    axes_flat[4].set_xlabel("Episode")
    axes_flat[3].set_xlabel("Episode")
    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[5].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle(
        "Online success trajectories: cumulative rate and moving-5 rate",
        y=1.01,
        fontsize=13,
    )
    save_figure(fig, "fig03_online_success_curves.png")


def plot_mechanism_metrics(metrics: dict[str, dict[str, dict]]) -> None:
    fields = (
        ("neighbor_action_coverage_rate", "Neighbor action coverage"),
        ("augmentation_choice_change_rate", "Augmentation changed choice"),
        ("choice_change_rate", "Advantage changed choice"),
        ("memory_only_selection_rate", "Memory-only selection"),
    )
    x = np.arange(len(TASK_ORDER))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for field_index, (field, label) in enumerate(fields):
        values = [metrics[task]["jitrl"][field] * 100 for task in TASK_ORDER]
        offset = (field_index - (len(fields) - 1) / 2) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=MECHANISM_COLORS[field_index],
        )
    ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylim(0, 72)
    ax.set_ylabel("Rate over high-level decisions / candidates (%)")
    ax.set_title("How strongly memory changed high-level decisions")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, loc="upper center")
    save_figure(fig, "fig04_memory_mechanism_metrics.png")


def plot_evaluator_scores(metrics: dict[str, dict[str, dict]]) -> None:
    negative = np.asarray(
        [metrics[task]["jitrl"]["negative_evaluator_score_rate"] for task in TASK_ORDER]
    )
    positive = np.asarray(
        [metrics[task]["jitrl"]["positive_evaluator_score_rate"] for task in TASK_ORDER]
    )
    neutral = np.clip(1.0 - negative - positive, 0.0, 1.0)
    means = np.asarray(
        [metrics[task]["jitrl"]["mean_evaluator_score"] for task in TASK_ORDER]
    )
    x = np.arange(len(TASK_ORDER))
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.bar(x, negative * 100, label="Negative", color="#C44E52")
    ax.bar(x, neutral * 100, bottom=negative * 100, label="Neutral", color="#BABFC5")
    ax.bar(
        x,
        positive * 100,
        bottom=(negative + neutral) * 100,
        label="Positive",
        color="#55A868",
    )
    ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Chunk evaluator distribution (%)")
    ax.set_title("Gemini step-evaluator score distribution in JitRL runs")
    ax.grid(axis="y", alpha=0.2)
    second = ax.twinx()
    second.plot(
        x,
        means,
        color="#222222",
        marker="D",
        linewidth=1.8,
        label="Mean score",
    )
    second.axhline(0, color="#222222", linewidth=0.8, linestyle=":")
    second.set_ylim(-0.8, 1.0)
    second.set_ylabel("Mean score (-3 to +3)")
    handles, labels = ax.get_legend_handles_labels()
    second_handles, second_labels = second.get_legend_handles_labels()
    ax.legend(
        handles + second_handles, labels + second_labels, ncol=4, loc="upper center"
    )
    save_figure(fig, "fig05_evaluator_score_distribution.png")


def plot_task79_reward_trace(episodes: dict[str, dict[str, list]]) -> None:
    episode = episodes["libero_90_task79"]["jitrl"][14]
    chunks = episode["chunks"]
    x = np.arange(len(chunks))
    scores = np.asarray([chunk["evaluator_score"] for chunk in chunks], dtype=float)
    returns = np.asarray([chunk["return"] for chunk in chunks], dtype=float)
    sources = [chunk["selected_source"] for chunk in chunks]
    colors = ["#F28E2B" if source == "memory" else "#4E79A7" for source in sources]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True)
    axes[0].bar(x, scores, color=colors, edgecolor="#222222", linewidth=0.4)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_ylabel("Evaluator score")
    axes[0].set_ylim(-3.4, 3.4)
    axes[0].set_title(
        "Task 79, JitRL episode 14: positive local credit despite environment failure"
    )
    axes[0].grid(axis="y", alpha=0.2)
    for index, source in enumerate(sources):
        if source == "memory":
            axes[0].text(index, scores[index] + 0.22, "M", ha="center", fontsize=8)

    axes[1].plot(x, returns, color="#7A5195", marker="o", linewidth=2)
    axes[1].fill_between(x, 0, returns, color="#7A5195", alpha=0.15)
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_ylabel("Discounted return")
    axes[1].set_xlabel("High-level chunk index")
    axes[1].grid(alpha=0.2)
    axes[1].set_xticks(x)
    axes[1].text(
        0.99,
        0.06,
        "Environment success = False; final book location = wrong caddy compartment",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    save_figure(fig, "fig06_task79_episode14_reward_trace.png")


def ffmpeg_extract_frame(video: Path, frame_index: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-frames:v",
        "1",
        str(destination),
    ]
    subprocess.run(command, check=True)
    if not destination.exists():
        raise RuntimeError(f"ffmpeg did not extract frame {frame_index} from {video}")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def contain(image: Image.Image, width: int, height: int) -> Image.Image:
    result = Image.new("RGB", (width, height), "white")
    resized = image.convert("RGB")
    resized.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    result.paste(resized, (x, y))
    return result


def make_storyboard(
    filename: str,
    title: str,
    rows: list[dict[str, Any]],
    temporary_dir: Path,
) -> None:
    cell_width = 330
    image_height = 240
    caption_height = 58
    label_width = 125
    title_height = 65
    columns = max(len(row["frames"]) for row in rows)
    canvas = Image.new(
        "RGB",
        (
            label_width + columns * cell_width,
            title_height + len(rows) * (image_height + caption_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font_size = 25
    title_font = font(title_font_size, bold=True)
    while (
        draw.textbbox((0, 0), title, font=title_font)[2] > canvas.width - 36
        and title_font_size > 14
    ):
        title_font_size -= 1
        title_font = font(title_font_size, bold=True)
    draw.text((18, 16), title, fill="#111111", font=title_font)
    for row_index, row in enumerate(rows):
        y = title_height + row_index * (image_height + caption_height)
        draw.rectangle(
            (0, y, label_width, y + image_height + caption_height),
            fill=row.get("label_color", "#F3F4F6"),
        )
        draw.multiline_text(
            (12, y + 22),
            row["label"],
            fill="#111111",
            font=font(18, bold=True),
            spacing=6,
        )
        for column_index, frame in enumerate(row["frames"]):
            temp_path = temporary_dir / f"{filename}_{row_index}_{column_index}.png"
            ffmpeg_extract_frame(row["video"], frame["index"], temp_path)
            image = contain(Image.open(temp_path), cell_width, image_height)
            x = label_width + column_index * cell_width
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + cell_width - 1, y + image_height - 1),
                outline="#D0D4D8",
                width=1,
            )
            draw.multiline_text(
                (x + 8, y + image_height + 5),
                frame["caption"],
                fill="#222222",
                font=font(14),
                spacing=2,
            )
    canvas.save(FIGURE_DIR / filename, quality=94)


def build_storyboards() -> None:
    temporary_dir = REPORT_DIR / ".frame_cache"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    try:
        make_storyboard(
            "fig07_case_task60_episode1.jpg",
            "Case A — Task 60, episode 1: recovery, redundant actions, and failure",
            [
                {
                    "label": "JitRL\nFAIL\n400 steps",
                    "label_color": "#DCE8F6",
                    "video": run_dir("libero_90_task60", "jitrl")
                    / "videos"
                    / "episode_1.mp4",
                    "frames": [
                        {"index": 0, "caption": "step 0\ninitial state"},
                        {"index": 60, "caption": "step 60\nwrong-object detour"},
                        {"index": 180, "caption": "step 180\nre-aligned to bowl"},
                        {"index": 270, "caption": "step 270\napparent placement"},
                        {"index": 400, "caption": "step 400\nenvironment failure"},
                    ],
                },
                {
                    "label": "Static\nSUCCESS\n119 steps",
                    "label_color": "#F9E4D4",
                    "video": run_dir("libero_90_task60", "static")
                    / "videos"
                    / "episode_1.mp4",
                    "frames": [
                        {"index": 0, "caption": "step 0\ninitial state"},
                        {"index": 30, "caption": "step 30\napproach"},
                        {"index": 60, "caption": "step 60\ngrasp"},
                        {"index": 90, "caption": "step 90\ntransport"},
                        {"index": 119, "caption": "step 119\nsuccess"},
                    ],
                },
            ],
            temporary_dir,
        )
        make_storyboard(
            "fig08_case_task79_episode14.jpg",
            "Case B — Task 79, episode 14: locally rewarded wrong-compartment placement",
            [
                {
                    "label": "JitRL\nFAIL\n400 steps",
                    "label_color": "#DCE8F6",
                    "video": run_dir("libero_90_task79", "jitrl")
                    / "videos"
                    / "episode_14.mp4",
                    "frames": [
                        {"index": 0, "caption": "step 0\ninitial state"},
                        {"index": 90, "caption": "step 90\nbook grasped"},
                        {"index": 150, "caption": "step 150\nmoved toward caddy"},
                        {"index": 390, "caption": "step 390\nreleased in wrong slot"},
                        {"index": 400, "caption": "step 400\nfailed correction"},
                    ],
                },
                {
                    "label": "Static\nSUCCESS\n374 steps",
                    "label_color": "#F9E4D4",
                    "video": run_dir("libero_90_task79", "static")
                    / "videos"
                    / "episode_14.mp4",
                    "frames": [
                        {"index": 0, "caption": "step 0\ninitial state"},
                        {"index": 90, "caption": "step 90\ngrip established"},
                        {"index": 180, "caption": "step 180\ntransport"},
                        {"index": 300, "caption": "step 300\nleft-slot alignment"},
                        {"index": 374, "caption": "step 374\nsuccess"},
                    ],
                },
            ],
            temporary_dir,
        )
        make_storyboard(
            "fig09_floor_task_final_frames.jpg",
            "Case C — Floor tasks: initial and final frames remain far from completion",
            [
                {
                    "label": "T27 JitRL\nFAIL",
                    "label_color": "#DCE8F6",
                    "video": run_dir("libero_90_task27", "jitrl")
                    / "videos"
                    / "episode_14.mp4",
                    "frames": [
                        {"index": 0, "caption": "step 0\ninitial state"},
                        {"index": 400, "caption": "step 400\nbottle not on rack"},
                    ],
                },
                {
                    "label": "T27 Static\nFAIL",
                    "label_color": "#F9E4D4",
                    "video": run_dir("libero_90_task27", "static")
                    / "videos"
                    / "episode_14.mp4",
                    "frames": [
                        {"index": 0, "caption": "step 0\ninitial state"},
                        {"index": 400, "caption": "step 400\nbottle not on rack"},
                    ],
                },
                {
                    "label": "T62 JitRL\nFAIL",
                    "label_color": "#DCE8F6",
                    "video": run_dir("libero_90_task62", "jitrl")
                    / "videos"
                    / "episode_14.mp4",
                    "frames": [
                        {"index": 0, "caption": "step 0\ninitial state"},
                        {"index": 400, "caption": "step 400\ntarget bottle untouched"},
                    ],
                },
                {
                    "label": "T62 Static\nFAIL",
                    "label_color": "#F9E4D4",
                    "video": run_dir("libero_90_task62", "static")
                    / "videos"
                    / "episode_14.mp4",
                    "frames": [
                        {"index": 0, "caption": "step 0\ninitial state"},
                        {"index": 400, "caption": "step 400\ntarget bottle untouched"},
                    ],
                },
            ],
            temporary_dir,
        )
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"missing experiment summary: {SUMMARY_PATH}")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary, episodes, metrics = load_records()
    derived = derive_statistics(summary, episodes, metrics)
    write_json(TABLE_DIR / "derived_statistics.json", derived)
    export_tables(derived, episodes, metrics)
    plot_success_rates(derived)
    plot_episode_outcomes(episodes)
    plot_online_curves(episodes)
    plot_mechanism_metrics(metrics)
    plot_evaluator_scores(metrics)
    plot_task79_reward_trace(episodes)
    build_storyboards()
    print(
        "Generated report assets:",
        len(list(FIGURE_DIR.iterdir())),
        "figures and",
        len(list(TABLE_DIR.iterdir())),
        "tables/data files",
    )


if __name__ == "__main__":
    main()
