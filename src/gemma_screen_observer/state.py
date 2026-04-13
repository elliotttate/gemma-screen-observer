"""State management and change log for game screen observations."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import LogConfig

logger = logging.getLogger(__name__)


def _iso_timestamp(ts: float) -> str:
    """Convert a Unix timestamp to ISO 8601 format."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass
class ChangeEntry:
    """A single change detected between frames."""

    timestamp: float
    frame_number: int
    category: str
    element: str
    from_state: str | None
    to_state: str
    significance: str
    summary: str | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "time": _iso_timestamp(self.timestamp),
            "frame_number": self.frame_number,
            "category": self.category,
            "element": self.element,
            "from": self.from_state,
            "to": self.to_state,
            "significance": self.significance,
            "summary": self.summary,
        }


@dataclass
class StateSnapshot:
    """A full state snapshot at a point in time."""

    timestamp: float
    frame_number: int
    scene: str
    description: str
    elements: dict
    text_on_screen: list[str]
    raw: dict

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "time": _iso_timestamp(self.timestamp),
            "frame_number": self.frame_number,
            "scene": self.scene,
            "description": self.description,
            "elements": self.elements,
            "text_on_screen": self.text_on_screen,
        }


class StateManager:
    """Tracks game state over time and maintains a change log."""

    def __init__(self, config: LogConfig):
        self.config = config
        self._current: StateSnapshot | None = None
        self._previous: StateSnapshot | None = None
        self._change_log: deque[ChangeEntry] = deque(maxlen=config.max_entries)
        self._snapshot_count: int = 0
        self._log_file: Path | None = None
        self._scene_history: deque[tuple[float, str]] = deque(maxlen=200)

        if config.persist:
            self._log_file = Path(config.output_file)
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    @property
    def current_state(self) -> StateSnapshot | None:
        return self._current

    @property
    def previous_state(self) -> StateSnapshot | None:
        return self._previous

    @property
    def change_count(self) -> int:
        return len(self._change_log)

    @property
    def snapshot_count(self) -> int:
        return self._snapshot_count

    def update_state(self, frame_number: int, analysis: dict) -> StateSnapshot:
        """Update the current state from a frame analysis result."""
        self._previous = self._current

        snapshot = StateSnapshot(
            timestamp=time.time(),
            frame_number=frame_number,
            scene=analysis.get("scene", "unknown"),
            description=analysis.get("description", ""),
            elements=analysis.get("elements", {}),
            text_on_screen=analysis.get("text_on_screen", []),
            raw=analysis,
        )

        self._current = snapshot
        self._snapshot_count += 1
        self._scene_history.append((snapshot.timestamp, snapshot.scene))

        if self.config.persist and self._log_file:
            self._persist_snapshot(snapshot)

        return snapshot

    def record_changes(self, frame_number: int, change_data: dict) -> list[ChangeEntry]:
        """Record detected changes from a change detection result."""
        entries = []
        now = time.time()

        if not change_data.get("has_changes", False):
            return entries

        summary = change_data.get("summary")

        for change in change_data.get("changes", []):
            entry = ChangeEntry(
                timestamp=now,
                frame_number=frame_number,
                category=change.get("category", "unknown"),
                element=change.get("element", "unknown"),
                from_state=change.get("from"),
                to_state=change.get("to", ""),
                significance=change.get("significance", "low"),
                summary=summary,
            )
            self._change_log.append(entry)
            entries.append(entry)

            if self.config.persist and self._log_file:
                self._persist_change(entry)

        return entries

    def get_recent_changes(
        self,
        count: int = 50,
        category: str | None = None,
        min_significance: str | None = None,
    ) -> list[dict]:
        """Get recent changes, optionally filtered."""
        significance_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = significance_order.get(min_significance or "low", 0)

        results = []
        for entry in reversed(self._change_log):
            if category and entry.category != category:
                continue
            if significance_order.get(entry.significance, 0) < min_level:
                continue
            results.append(entry.to_dict())
            if len(results) >= count:
                break

        return results

    def get_scene_history(self) -> list[dict]:
        """Get the history of scene transitions."""
        transitions = []
        prev_scene = None
        for ts, scene in self._scene_history:
            if scene != prev_scene:
                transitions.append({"timestamp": ts, "time": _iso_timestamp(ts), "scene": scene})
                prev_scene = scene
        return transitions

    def get_state_summary(self) -> dict:
        """Get a summary of the current observation state."""
        return {
            "observing": self._current is not None,
            "snapshots_taken": self._snapshot_count,
            "changes_recorded": len(self._change_log),
            "current_scene": self._current.scene if self._current else None,
            "current_description": self._current.description if self._current else None,
            "last_update": self._current.timestamp if self._current else None,
            "last_update_time": _iso_timestamp(self._current.timestamp) if self._current else None,
            "scene_transitions": len(self.get_scene_history()),
        }

    def log_frame_tick(self, frame_number: int, diff_score: float, changed: bool) -> None:
        """Log a tick entry only when a visual change was detected."""
        if not changed:
            return
        if self.config.persist and self._log_file:
            now = time.time()
            record = {
                "type": "tick",
                "timestamp": now,
                "time": _iso_timestamp(now),
                "frame_number": frame_number,
                "diff_score": round(diff_score, 4),
            }
            with self._log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def clear(self) -> None:
        """Clear all state and history."""
        self._current = None
        self._previous = None
        self._change_log.clear()
        self._scene_history.clear()
        self._snapshot_count = 0
        logger.info("State manager cleared")

    def _persist_snapshot(self, snapshot: StateSnapshot) -> None:
        """Append a snapshot entry to the JSONL log file."""
        record = {"type": "snapshot", **snapshot.to_dict()}
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _persist_change(self, entry: ChangeEntry) -> None:
        """Append a change entry to the JSONL log file."""
        record = {"type": "change", **entry.to_dict()}
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
