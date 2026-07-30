"""Lightweight per-task experience memory for the first JitRL stage."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import NotRequired, TypedDict

_TOKEN_RE = re.compile(r"[a-z0-9]+")

MemoryEntry = TypedDict(
    "MemoryEntry",
    {
        "id": int,
        "state": str,
        "action_key": str,
        "action_text": str,
        "return": float,
        "episode_index": int,
        "chunk_index": int,
    },
)

RetrievedEntry = TypedDict(
    "RetrievedEntry",
    {
        "id": int,
        "state": str,
        "action_key": str,
        "action_text": str,
        "return": float,
        "episode_index": int,
        "chunk_index": int,
        "similarity": float,
    },
)


class ChunkRecord(TypedDict):
    """A state-action record buffered during one episode."""

    state: str
    action_key: str
    action_text: str
    chunk_index: NotRequired[int]


class CandidateValue(TypedDict):
    """Local value estimate for one candidate action."""

    action_key: str
    Q: float
    advantage: float
    normalized_advantage: float
    neighbor_count: int
    seen: bool
    exploration_uniform: float | None
    exploration_applied: bool
    exploration_branch: str
    ucb_bonus: float


class ValueEstimate(TypedDict):
    """State and candidate values estimated from retrieved neighbors."""

    V: float
    neighbor_count: int
    neighbor_action_coverage: dict[str, int]
    candidates: list[CandidateValue]


def tokenize(text: str) -> set[str]:
    """Extract a set of lowercase English/alphanumeric tokens."""

    return set(_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(left: str, right: str) -> float:
    """Return token-set Jaccard similarity, or zero for two empty sets."""

    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def discounted_returns(rewards: Sequence[float], gamma: float = 0.95) -> list[float]:
    """Compute signed reward-to-go values for one completed chunk trajectory."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def discounted_chunk_returns(
    length: int, success: bool, gamma: float = 0.95
) -> list[float]:
    """Compatibility helper for the earlier terminal-only reward definition."""

    rewards = [0.0] * length
    if success and rewards:
        rewards[-1] = 1.0
    return discounted_returns(rewards, gamma=gamma)


class JitRLMemory:
    """In-memory JitRL experience store intended for a single task."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self._entries: list[MemoryEntry] = []
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._entries)

    def retrieve(self, state: str, k: int | None = None) -> list[RetrievedEntry]:
        """Retrieve top-k entries by Jaccard score, breaking ties by stable id."""

        limit = self.top_k if k is None else k
        if limit <= 0:
            return []

        scored = [
            (jaccard_similarity(state, entry["state"]), entry)
            for entry in self._entries
        ]
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [
            RetrievedEntry(**entry, similarity=similarity)
            for similarity, entry in scored[:limit]
        ]

    def estimate_candidate_values(
        self,
        state: str,
        candidate_action_keys: Sequence[str],
        k: int | None = None,
        *,
        neighbors: Sequence[RetrievedEntry] | None = None,
        exploration_uniforms: Sequence[float] | None = None,
        exploration_rate: float = 0.0,
        ucb_alpha: float = 0.0,
    ) -> ValueEstimate:
        """Estimate local V/Q/A with the paper's stochastic unseen-action rule."""

        if not 0.0 <= exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be in [0, 1]")
        if ucb_alpha < 0.0:
            raise ValueError("ucb_alpha must be non-negative")
        local_neighbors = list(
            self.retrieve(state, k) if neighbors is None else neighbors
        )
        if exploration_uniforms is None:
            uniforms: list[float | None] = [None] * len(candidate_action_keys)
        else:
            uniforms = [float(value) for value in exploration_uniforms]
            if len(uniforms) != len(candidate_action_keys):
                raise ValueError(
                    "exploration_uniforms and candidate_action_keys must have the same length"
                )
            if any(not 0.0 <= value < 1.0 for value in uniforms):
                raise ValueError("exploration uniforms must be in [0, 1)")

        value = (
            sum(neighbor["return"] for neighbor in local_neighbors)
            / len(local_neighbors)
            if local_neighbors
            else 0.0
        )
        action_returns: dict[str, list[float]] = defaultdict(list)
        for neighbor in local_neighbors:
            action_returns[neighbor["action_key"]].append(neighbor["return"])

        candidates: list[CandidateValue] = []
        for action_key, exploration_uniform in zip(candidate_action_keys, uniforms):
            returns = action_returns.get(action_key, [])
            seen = bool(returns)
            exploration_applied = False
            exploration_branch = "known"
            ucb_bonus = 0.0
            if seen:
                q_value = sum(returns) / len(returns)
            elif not local_neighbors:
                q_value = 0.0
                exploration_branch = "empty_memory"
            elif (
                exploration_uniform is not None
                and exploration_uniform < exploration_rate
            ):
                ucb_bonus = ucb_alpha / len(local_neighbors)
                q_value = value + ucb_bonus
                exploration_applied = True
                exploration_branch = "optimistic"
            else:
                q_value = 0.0
                exploration_branch = "zero"
            advantage = q_value - value
            candidates.append(
                {
                    "action_key": action_key,
                    "Q": q_value,
                    "advantage": advantage,
                    "normalized_advantage": 0.0,
                    "neighbor_count": len(returns),
                    "seen": seen,
                    "exploration_uniform": exploration_uniform,
                    "exploration_applied": exploration_applied,
                    "exploration_branch": exploration_branch,
                    "ucb_bonus": ucb_bonus,
                }
            )

        scale = max((abs(item["advantage"]) for item in candidates), default=0.0)
        if scale:
            for item in candidates:
                item["normalized_advantage"] = item["advantage"] / scale

        coverage = dict(Counter(neighbor["action_key"] for neighbor in local_neighbors))
        return {
            "V": value,
            "neighbor_count": len(local_neighbors),
            "neighbor_action_coverage": coverage,
            "candidates": candidates,
        }

    def append_episode(
        self,
        records: Sequence[ChunkRecord],
        returns: Sequence[float],
        episode_index: int,
    ) -> list[MemoryEntry]:
        """Atomically append one episode's buffered records after it ends."""

        if len(records) != len(returns):
            raise ValueError("records and returns must have the same length")

        new_entries: list[MemoryEntry] = []
        for offset, (record, return_value) in enumerate(zip(records, returns)):
            new_entries.append(
                {
                    "id": self._next_id + offset,
                    "state": record["state"],
                    "action_key": record["action_key"],
                    "action_text": record["action_text"],
                    "return": float(return_value),
                    "episode_index": episode_index,
                    "chunk_index": record.get("chunk_index", offset),
                }
            )

        self._entries.extend(new_entries)
        self._next_id += len(new_entries)
        return [entry.copy() for entry in new_entries]

    def snapshot(self) -> list[MemoryEntry]:
        """Return a detached, JSON-safe list for per-episode persistence."""

        return [entry.copy() for entry in self._entries]
