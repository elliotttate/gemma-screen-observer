"""Fast frame differencing for high-frequency change detection.

Uses pixel-level comparison and perceptual hashing to detect visual changes
between consecutive frames in <10ms — no LLM needed. Only frames that exceed
the change threshold get sent to Gemma 4 for full analysis.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

from PIL import Image

logger = logging.getLogger(__name__)

# Downscale resolution for fast comparison (small = fast)
COMPARE_SIZE = (160, 90)


@dataclass
class DiffResult:
    """Result of comparing two frames."""

    changed: bool
    score: float  # 0.0 = identical, 1.0 = completely different
    elapsed_ms: float
    region_scores: list[float] | None = None  # per-region change scores


@dataclass
class FrameDiffer:
    """Fast frame comparison using pixel difference and grid-based region analysis.

    Divides each frame into a grid and computes per-region change scores.
    This lets us detect localized changes (like a health bar updating) even
    when the overall frame looks similar.
    """

    threshold: float = 0.02  # Minimum change score to count as "changed"
    grid: tuple[int, int] = (4, 3)  # 4x3 grid = 12 regions
    _prev_small: Image.Image | None = field(default=None, init=False, repr=False)
    _prev_hash: str | None = field(default=None, init=False)

    def compare(self, current: Image.Image) -> DiffResult:
        """Compare current frame against the previous one.

        Returns a DiffResult with the change score and per-region breakdown.
        Runs in <10ms for typical game frames.
        """
        t0 = time.perf_counter()

        # Downscale for fast comparison
        small = current.resize(COMPARE_SIZE, Image.NEAREST).convert("RGB")

        if self._prev_small is None:
            self._prev_small = small
            self._prev_hash = _quick_hash(small)
            elapsed = (time.perf_counter() - t0) * 1000
            return DiffResult(changed=True, score=1.0, elapsed_ms=elapsed)

        # Quick hash check — if identical, skip pixel comparison
        curr_hash = _quick_hash(small)
        if curr_hash == self._prev_hash:
            self._prev_small = small
            self._prev_hash = curr_hash
            elapsed = (time.perf_counter() - t0) * 1000
            return DiffResult(changed=False, score=0.0, elapsed_ms=elapsed)

        # Pixel-level mean absolute difference
        score, region_scores = _pixel_diff(self._prev_small, small, self.grid)

        self._prev_small = small
        self._prev_hash = curr_hash
        elapsed = (time.perf_counter() - t0) * 1000

        return DiffResult(
            changed=score >= self.threshold,
            score=score,
            elapsed_ms=elapsed,
            region_scores=region_scores,
        )

    def reset(self) -> None:
        """Clear the previous frame reference."""
        self._prev_small = None
        self._prev_hash = None


def _quick_hash(img: Image.Image) -> str:
    """Fast perceptual hash: downscale to 8x8, grayscale, threshold at median."""
    tiny = img.resize((8, 8), Image.NEAREST).convert("L")
    pixels = list(tiny.getdata())
    median = sorted(pixels)[32]
    bits = "".join("1" if p > median else "0" for p in pixels)
    return hashlib.md5(bits.encode()).hexdigest()


def _pixel_diff(
    prev: Image.Image, curr: Image.Image, grid: tuple[int, int]
) -> tuple[float, list[float]]:
    """Compute mean absolute pixel difference, overall and per-region.

    Returns (overall_score, [region_scores]) where scores are 0.0-1.0.
    """
    w, h = prev.size
    prev_data = prev.tobytes()
    curr_data = curr.tobytes()

    # Overall difference
    total_diff = 0
    n = len(prev_data)
    for i in range(0, n, 3):  # Step by 3 (RGB)
        total_diff += (
            abs(prev_data[i] - curr_data[i])
            + abs(prev_data[i + 1] - curr_data[i + 1])
            + abs(prev_data[i + 2] - curr_data[i + 2])
        )
    overall = total_diff / (n * 255 / 3)  # Normalize to 0-1

    # Per-region difference
    gx, gy = grid
    rw, rh = w // gx, h // gy
    region_scores = []

    for row in range(gy):
        for col in range(gx):
            region_diff = 0
            region_pixels = 0
            for y in range(row * rh, min((row + 1) * rh, h)):
                for x in range(col * rw, min((col + 1) * rw, w)):
                    idx = (y * w + x) * 3
                    region_diff += (
                        abs(prev_data[idx] - curr_data[idx])
                        + abs(prev_data[idx + 1] - curr_data[idx + 1])
                        + abs(prev_data[idx + 2] - curr_data[idx + 2])
                    )
                    region_pixels += 1
            if region_pixels > 0:
                region_scores.append(region_diff / (region_pixels * 255 * 3))
            else:
                region_scores.append(0.0)

    return overall, region_scores
