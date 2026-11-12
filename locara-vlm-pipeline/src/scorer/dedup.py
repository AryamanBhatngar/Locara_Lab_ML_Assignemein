"""Near-duplicate detection with perceptual hashes.

Each clip is reduced to the dHash of its segment-centre frames. Two clips are compared by
aligning their hash sequences proportionally in time and taking the median Hamming distance.
Re-encoding, resizing or a small brightness change moves a dHash by a few bits; a different
scene moves it by around 32. The registry can be persisted, so duplicates are caught across
batches as well as within one.

Ids come from the content, so an exact re-upload has the same id as its original. The registry
keeps each clip's origin to tell the two cases apart: the same id from the same place is a re-run,
and the same id from anywhere else is a duplicate.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config
from .frame_sampler import hamming


@dataclass
class DuplicateMatch:
    video_id: str
    distance: float

    @property
    def is_duplicate(self) -> bool:
        return self.distance <= config.DUPLICATE_MAX_HAMMING


def sequence_distance(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 64.0
    n = min(len(a), len(b))
    ia = np.linspace(0, len(a) - 1, n).round().astype(int)
    ib = np.linspace(0, len(b) - 1, n).round().astype(int)
    # A uniform frame hashes to 0, so two blank frames would match whatever the clips are.
    distances = [hamming(a[i], b[j]) for i, j in zip(ia, ib, strict=True) if a[i] and b[j]]
    return float(np.median(distances)) if distances else 64.0


@dataclass
class _Entry:
    hashes: list[int]
    origin: str


class HashRegistry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        if self.path and self.path.is_file():
            self._entries = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> dict[str, _Entry]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                video_id: _Entry([int(h, 16) for h in entry["hashes"]], entry["origin"])
                for video_id, entry in data.items()
            }
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError(
                f"hash registry {path} is corrupt or from an older version: {exc}"
            ) from exc

    def __len__(self) -> int:
        return len(self._entries)

    def _closest(self, video_id: str, hashes: list[int], origin: str) -> DuplicateMatch | None:
        best: DuplicateMatch | None = None
        for other, entry in self._entries.items():
            if other == video_id:
                if entry.origin == origin:
                    break  # a re-run: only clips registered before this one can be its original
                return DuplicateMatch(other, 0.0)
            d = sequence_distance(hashes, entry.hashes)
            if best is None or d < best.distance:
                best = DuplicateMatch(other, d)
        return best

    def _register(self, video_id: str, hashes: list[int], origin: str) -> None:
        known = self._entries.get(video_id)
        if known is None or known.origin == origin:  # the first location stays the original
            self._entries[video_id] = _Entry(list(hashes), origin)

    def best_match(
        self, video_id: str, hashes: list[int], origin: str = ""
    ) -> DuplicateMatch | None:
        with self._lock:
            return self._closest(video_id, hashes, origin)

    def add(self, video_id: str, hashes: list[int], origin: str = "") -> None:
        with self._lock:
            self._register(video_id, hashes, origin)

    def check_and_add(
        self, video_id: str, hashes: list[int], origin: str = ""
    ) -> DuplicateMatch | None:
        """Return the closest earlier clip (if any), then register this one.

        Both steps share one lock, so two copies checked at once cannot both miss each other.
        """
        with self._lock:
            match = self._closest(video_id, hashes, origin)
            self._register(video_id, hashes, origin)
        return match

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            snapshot = {
                video_id: {"origin": entry.origin, "hashes": [f"{h:016x}" for h in entry.hashes]}
                for video_id, entry in self._entries.items()
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        finally:
            tmp.unlink(missing_ok=True)
