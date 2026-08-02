"""Fixed semantic-action workspace for Qwen-only JitRL planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

ACTION_TYPES = (
    "approach",
    "align",
    "grasp",
    "lift",
    "transport",
    "place",
    "release",
    "retract",
    "stop",
)
ACTION_LABELS = {action_type: str(index) for index, action_type in enumerate(ACTION_TYPES, 1)}


@dataclass(frozen=True)
class ActionSpec:
    parameters: tuple[str, ...]
    description: str


ACTION_SPECS = {
    "approach": ActionSpec(("target",), "move the gripper toward a target"),
    "align": ActionSpec(("gripper", "target"), "align the gripper with a target"),
    "grasp": ActionSpec(("target",), "close the gripper around a target"),
    "lift": ActionSpec(("target",), "lift a grasped target"),
    "transport": ActionSpec(
        ("target", "destination"), "carry a grasped target toward a destination"
    ),
    "place": ActionSpec(
        ("target", "destination"), "move a grasped target into placement contact"
    ),
    "release": ActionSpec(
        ("target", "destination"), "open the gripper to release a placed target"
    ),
    "retract": ActionSpec(("direction",), "move the gripper away to recover or clear"),
    "stop": ActionSpec((), "stop because the task is complete or cannot continue"),
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def action_schema_text() -> str:
    """Return the immutable action vocabulary shown to Qwen."""

    lines = []
    for action_type in ACTION_TYPES:
        spec = ACTION_SPECS[action_type]
        signature = ", ".join(spec.parameters)
        lines.append(
            f"{ACTION_LABELS[action_type]}. {action_type}({signature}): {spec.description}"
        )
    return "\n".join(lines)


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-") or "none"


def action_key(action: dict[str, Any]) -> str:
    """Build the stable memory key for one instantiated action."""

    action_type = action["type"]
    if action_type == "stop":
        return "stop|task|done"
    if action_type == "retract":
        return f"retract|gripper|{_slug(action['direction'])}"
    target = _slug(action["target"])
    destination = _slug(action.get("destination", "none"))
    return f"{action_type}|{target}|{destination}"


def render_action(action: dict[str, Any]) -> str:
    """Render one workspace instance as a short π₀.₅ semantic command."""

    action_type = action["type"]
    target = action.get("target", "")
    destination = action.get("destination", "")
    direction = action.get("direction", "")
    if action_type == "approach":
        return f"move toward {target}"
    if action_type == "align":
        return f"align the gripper with {target}"
    if action_type == "grasp":
        return f"grasp {target}"
    if action_type == "lift":
        return f"lift {target}"
    if action_type == "transport":
        return f"carry {target} toward {destination}"
    if action_type == "place":
        return f"place {target} at {destination}"
    if action_type == "release":
        return f"release {target} at {destination}"
    if action_type == "retract":
        return f"retract {direction}"
    return "stop"


def instantiate_workspace(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and render exactly one binding for each fixed action type."""

    action_types = [binding["type"] for binding in bindings]
    missing = [action_type for action_type in ACTION_TYPES if action_type not in action_types]
    unexpected = [action_type for action_type in action_types if action_type not in ACTION_TYPES]
    duplicates = sorted(
        {action_type for action_type in action_types if action_types.count(action_type) > 1}
    )
    if missing or unexpected or duplicates:
        raise ValueError(
            "workspace action types invalid: "
            f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
        )
    by_type = {binding["type"]: binding for binding in bindings}

    workspace = []
    for action_type in ACTION_TYPES:
        binding = by_type[action_type]
        action = {
            "label": ACTION_LABELS[action_type],
            "type": action_type,
            "enabled": bool(binding["enabled"]),
            "terminates_rollout": action_type == "stop",
        }
        for parameter in ACTION_SPECS[action_type].parameters:
            value = str(binding[parameter]).strip()
            if not value:
                raise ValueError(f"{action_type}.{parameter} must be non-empty")
            action[parameter] = value
        action["text"] = render_action(action)
        action["semantic_key"] = action_key(action)
        workspace.append(action)
    return workspace


def workspace_from_binding(binding: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct the fixed workspace from Qwen's compact scene binding."""

    enabled_actions = list(binding["enabled_actions"])
    unexpected = [
        action_type for action_type in enabled_actions if action_type not in ACTION_TYPES
    ]
    duplicates = sorted(
        {
            action_type
            for action_type in enabled_actions
            if enabled_actions.count(action_type) > 1
        }
    )
    if unexpected or duplicates:
        raise ValueError(
            "enabled action types invalid: "
            f"unexpected={unexpected}, duplicates={duplicates}"
        )

    target = str(binding["object"]).strip()
    destination = str(binding["destination"]).strip()
    approach_target = str(binding["approach_target"]).strip()
    align_target = str(binding["align_target"]).strip()
    direction = str(binding["retract_direction"]).strip()
    bindings = [
        {"type": "approach", "target": approach_target},
        {"type": "align", "gripper": "the gripper", "target": align_target},
        {"type": "grasp", "target": target},
        {"type": "lift", "target": target},
        {"type": "transport", "target": target, "destination": destination},
        {"type": "place", "target": target, "destination": destination},
        {"type": "release", "target": target, "destination": destination},
        {"type": "retract", "direction": direction},
        {"type": "stop"},
    ]
    for action in bindings:
        action["enabled"] = action["type"] in enabled_actions
    return instantiate_workspace(bindings)


def active_actions(workspace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the current Qwen-bound action set without adding memory actions."""

    actions = [action for action in workspace if action["enabled"]]
    if not actions:
        raise ValueError("Qwen workspace must enable at least one action")
    return actions
