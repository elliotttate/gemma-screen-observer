"""Orchestrator — two-tier capture loop for high-frequency observation.

Tier 1 (every frame): Fast pixel-based diff (~5ms). Logs every capture with
        timestamp + change score. No LLM call.
Tier 2 (on change):   When visual change exceeds threshold, sends the frame
        to Gemma 4 for full analysis and structured description.

This lets us capture at 1fps while only burning GPU inference time when the
screen actually changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from .capture import Frame, ScreenCapture, enumerate_windows, list_monitors
from .config import ObserverConfig
from .fast_diff import DiffResult, FrameDiffer
from .observer import ScreenObserver
from .state import StateManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Two-tier observation: fast diff every frame, LLM analysis on changes."""

    def __init__(self, config: ObserverConfig):
        self.config = config
        self.capturer = ScreenCapture(config.capture)
        self.observer = ScreenObserver(config.model)
        self.state = StateManager(config.log)
        self.differ = FrameDiffer(threshold=config.capture.change_threshold)

        # Frame storage directory — saved frames for later LLM lookup
        self._frames_dir = Path(config.log.output_file).parent / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._saved_frames: dict[int, Path] = {}  # frame_number -> path

        self._running = False
        self._task: asyncio.Task | None = None
        self._analysis_task: asyncio.Task | None = None
        self._last_frame: Frame | None = None
        self._last_analyzed_frame: Frame | None = None
        self._pending_analysis: Frame | None = None
        self._analyzing = False
        self._errors: list[dict] = []
        self._lock = asyncio.Lock()

        # Stats
        self._frames_total = 0
        self._frames_changed = 0
        self._frames_analyzed = 0
        self._avg_diff_ms = 0.0
        self._avg_analysis_ms = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the observation loop."""
        if self._running:
            logger.warning("Observation already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="observation-loop")
        logger.info(
            "Observation started (interval=%.1fs, threshold=%.3f, model=%s)",
            self.config.capture.interval,
            self.config.capture.change_threshold,
            self.config.model.model_name,
        )

    async def stop(self) -> None:
        """Stop the observation loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("Observation stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Core loop — two-tier
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Main capture loop. Captures at interval, diffs fast, analyzes on change."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                error_info = {"time": time.time(), "error": str(exc), "type": type(exc).__name__}
                self._errors.append(error_info)
                if len(self._errors) > 100:
                    self._errors = self._errors[-50:]
                logger.error("Observation tick failed: %s", exc, exc_info=True)

            await asyncio.sleep(self.config.capture.interval)

    async def _tick(self) -> None:
        """Single tick: capture → fast diff → maybe queue analysis."""
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, self.capturer.capture)
        self._frames_total += 1

        # Tier 1: Fast pixel diff (~5ms)
        diff = await loop.run_in_executor(None, self.differ.compare, frame.image)

        # Update rolling average
        self._avg_diff_ms = (self._avg_diff_ms * 0.9) + (diff.elapsed_ms * 0.1)

        # Log every frame to the state manager (lightweight entry)
        frame_path_str = None
        if diff.changed:
            frame_path_str = str(self._frames_dir / f"frame_{frame.frame_number:06d}.jpg")
        self.state.log_frame_tick(frame.frame_number, diff.score, diff.changed, frame_path_str)

        if diff.changed:
            self._frames_changed += 1
            self._last_frame = frame

            # Save the frame to disk for later lookup
            frame_path = self._frames_dir / f"frame_{frame.frame_number:06d}.jpg"
            await loop.run_in_executor(None, self._save_frame, frame, frame_path)
            self._saved_frames[frame.frame_number] = frame_path

            # Tier 2: Queue for LLM analysis (non-blocking)
            if not self._analyzing:
                self._analyzing = True
                self._analysis_task = asyncio.create_task(
                    self._analyze_frame(frame), name="analysis"
                )
        else:
            self._last_frame = frame

    async def _analyze_frame(self, frame: Frame) -> None:
        """Run Gemma 4 analysis on a changed frame (runs in background)."""
        try:
            t0 = time.perf_counter()

            async with self._lock:
                analysis = await self.observer.analyze(frame)
                frame_path = self._saved_frames.get(frame.frame_number)
                self.state.update_state(
                    frame.frame_number, analysis,
                    frame_path=str(frame_path) if frame_path else None,
                )

                # Change detection against last analyzed frame
                if self._last_analyzed_frame is not None:
                    changes = await self.observer.detect_changes(
                        self._last_analyzed_frame, frame
                    )
                    self.state.record_changes(frame.frame_number, changes)

                self._last_analyzed_frame = frame
                self._frames_analyzed += 1

            elapsed = (time.perf_counter() - t0) * 1000
            self._avg_analysis_ms = (self._avg_analysis_ms * 0.8) + (elapsed * 0.2)
            logger.debug(
                "Analysis complete: frame %d in %.0fms", frame.frame_number, elapsed
            )
        except Exception as exc:
            logger.error("Analysis failed: %s", exc, exc_info=True)
        finally:
            self._analyzing = False

    @staticmethod
    def _save_frame(frame: Frame, path: Path) -> None:
        """Save a frame to disk as JPEG."""
        rgb = frame.image.convert("RGB") if frame.image.mode != "RGB" else frame.image
        rgb.save(path, format="JPEG", quality=90)

    # ------------------------------------------------------------------
    # On-demand actions (called by MCP tools)
    # ------------------------------------------------------------------

    async def take_snapshot(self) -> dict:
        """Force an immediate capture and analysis."""
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, self.capturer.capture)

        async with self._lock:
            previous = self._last_analyzed_frame
            analysis = await self.observer.analyze(frame)
            snapshot = self.state.update_state(frame.frame_number, analysis)

            changes_data = None
            if previous is not None:
                changes_data = await self.observer.detect_changes(previous, frame)
                self.state.record_changes(frame.frame_number, changes_data)

            self._last_frame = frame
            self._last_analyzed_frame = frame

        return {
            "snapshot": snapshot.to_dict(),
            "changes": changes_data,
            "frame_number": frame.frame_number,
            "source": frame.source,
        }

    async def query_screen(self, question: str) -> str:
        """Ask a free-form question about the current screen."""
        if self._last_frame is None:
            loop = asyncio.get_running_loop()
            self._last_frame = await loop.run_in_executor(None, self.capturer.capture)
        return await self.observer.query(self._last_frame, question)

    def get_current_state(self) -> dict | None:
        if self.state.current_state is None:
            return None
        return self.state.current_state.to_dict()

    def get_recent_changes(
        self,
        count: int = 50,
        category: str | None = None,
        min_significance: str | None = None,
    ) -> list[dict]:
        return self.state.get_recent_changes(count, category, min_significance)

    def get_screenshot_base64(self) -> str | None:
        if self._last_frame is None:
            return None
        return self._last_frame.compressed_base64(max_kb=self.config.capture.max_image_size_kb)

    def get_status(self) -> dict:
        window = self.capturer.target_window
        return {
            "running": self._running,
            "backend": self.config.model.backend,
            "model": self.config.model.model_name,
            "capture_interval": self.config.capture.interval,
            "change_threshold": self.config.capture.change_threshold,
            "target_window": window.to_dict() if window else None,
            "frames_captured": self._frames_total,
            "frames_changed": self._frames_changed,
            "frames_analyzed": self._frames_analyzed,
            "avg_diff_ms": round(self._avg_diff_ms, 1),
            "avg_analysis_ms": round(self._avg_analysis_ms, 0),
            "analyzing_now": self._analyzing,
            "state": self.state.get_state_summary(),
            "recent_errors": self._errors[-5:] if self._errors else [],
        }

    def get_scene_history(self) -> list[dict]:
        return self.state.get_scene_history()

    def list_available_windows(self) -> list[dict]:
        windows = enumerate_windows()
        return [w.to_dict() for w in windows]

    def list_available_monitors(self) -> list[dict]:
        return list_monitors()

    def set_target_window(self, title: str | None = None, process_name: str | None = None) -> dict:
        self.config.capture.window_title = title
        self.config.capture.process_name = process_name
        self.capturer.config = self.config.capture
        window = self.capturer.refresh_window()
        if window:
            return {"success": True, "window": window.to_dict()}
        return {"success": False, "error": "No matching window found"}

    def set_interval(self, interval: float) -> None:
        self.config.capture.interval = max(0.1, min(30.0, interval))

    def get_saved_frame_path(self, frame_number: int) -> Path | None:
        """Get the file path of a saved frame by number."""
        return self._saved_frames.get(frame_number)

    def list_saved_frames(self) -> list[dict]:
        """List all saved frames with their numbers and paths."""
        return [
            {"frame_number": num, "path": str(path)}
            for num, path in sorted(self._saved_frames.items())
        ]

    async def analyze_saved_frame(self, frame_number: int, question: str | None = None) -> dict:
        """Re-analyze a previously saved frame, optionally with a custom question."""
        path = self._saved_frames.get(frame_number)
        if path is None or not path.exists():
            return {"error": f"Frame {frame_number} not found"}

        from PIL import Image as PILImage
        img = PILImage.open(path)
        frame = Frame(image=img, timestamp=0, frame_number=frame_number, source=f"saved:{path}")

        if question:
            answer = await self.observer.query(frame, question)
            return {"frame_number": frame_number, "question": question, "answer": answer, "frame_path": str(path)}
        else:
            analysis = await self.observer.analyze(frame)
            return {"frame_number": frame_number, "analysis": analysis, "frame_path": str(path)}

    async def close(self) -> None:
        await self.stop()
        await self.observer.close()
        logger.info("Orchestrator shut down")
