"""Lightweight per-task experience memory for the first JitRL stage."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import NotRequired, Sequence, TypedDict

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


def discounted_chunk_returns(
    length: int, success: bool, gamma: float = 0.95
) -> list[float]:
    """Assign a discounted terminal +1 to a successful chunk trajectory."""

    if not success:
        return [0.0] * length
    return [gamma ** (length - 1 - t) for t in range(length)]


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
    ) -> ValueEstimate:
        """Estimate local V, Q, and normalized advantages for candidates."""

        neighbors = self.retrieve(state, k)
        value = (
            sum(neighbor["return"] for neighbor in neighbors) / len(neighbors)
            if neighbors
            else 0.0
        )

        action_returns: dict[str, list[float]] = defaultdict(list)
        for neighbor in neighbors:
            action_returns[neighbor["action_key"]].append(neighbor["return"])

        candidates: list[CandidateValue] = []
        for action_key in candidate_action_keys:
            returns = action_returns.get(action_key, [])
            seen = bool(returns)
            q_value = sum(returns) / len(returns) if seen else value
            advantage = q_value - value
            candidates.append(
                {
                    "action_key": action_key,
                    "Q": q_value,
                    "advantage": advantage,
                    "normalized_advantage": 0.0,
                    "neighbor_count": len(returns),
                    "seen": seen,
                }
            )

        scale = max((abs(item["advantage"]) for item in candidates), default=0.0)
        if scale:
            for item in candidates:
                item["normalized_advantage"] = item["advantage"] / scale

        coverage = dict(Counter(neighbor["action_key"] for neighbor in neighbors))
        return {
            "V": value,
            "neighbor_count": len(neighbors),
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
