"""Orchestrator — runs the capture-analyze-log loop and exposes state to the MCP server."""

from __future__ import annotations

import asyncio
import logging
import time

from .capture import Frame, ScreenCapture, enumerate_windows, find_windows, list_monitors
from .config import ObserverConfig
from .observer import ScreenObserver
from .state import StateManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Manages the observation lifecycle: capture → analyze → update state → log.

    Runs as a background asyncio task while the MCP server handles tool calls.
    All public methods are safe to call from any coroutine.
    """

    def __init__(self, config: ObserverConfig):
        self.config = config
        self.capturer = ScreenCapture(config.capture)
        self.observer = ScreenObserver(config.model)
        self.state = StateManager(config.log)

        self._running = False
        self._task: asyncio.Task | None = None
        self._last_frame: Frame | None = None
        self._errors: list[dict] = []
        self._lock = asyncio.Lock()

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
            "Observation started (interval=%.1fs, backend=%s)",
            self.config.capture.interval,
            self.config.model.backend,
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
        logger.info("Observation stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Main capture-analyze loop."""
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
        """Single observation cycle: capture → analyze → detect changes → update state."""
        # Capture runs in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, self.capturer.capture)

        async with self._lock:
            previous = self._last_frame

            # Analyze the current frame
            analysis = await self.observer.analyze(frame)
            self.state.update_state(frame.frame_number, analysis)

            # Detect changes if we have a previous frame
            if previous is not None:
                changes = await self.observer.detect_changes(previous, frame)
                self.state.record_changes(frame.frame_number, changes)

            self._last_frame = frame

    # ------------------------------------------------------------------
    # On-demand actions (called by MCP tools)
    # ------------------------------------------------------------------

    async def take_snapshot(self) -> dict:
        """Force an immediate capture and analysis, outside the normal loop."""
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(None, self.capturer.capture)

        async with self._lock:
            previous = self._last_frame
            analysis = await self.observer.analyze(frame)
            snapshot = self.state.update_state(frame.frame_number, analysis)

            changes_data = None
            if previous is not None:
                changes_data = await self.observer.detect_changes(previous, frame)
                self.state.record_changes(frame.frame_number, changes_data)

            self._last_frame = frame

        return {
            "snapshot": snapshot.to_dict(),
            "changes": changes_data,
            "frame_number": frame.frame_number,
            "source": frame.source,
        }

    async def query_screen(self, question: str) -> str:
        """Ask a free-form question about the current screen."""
        if self._last_frame is None:
            # Capture a fresh frame
            loop = asyncio.get_running_loop()
            self._last_frame = await loop.run_in_executor(None, self.capturer.capture)

        return await self.observer.query(self._last_frame, question)

    def get_current_state(self) -> dict | None:
        """Get the current state snapshot as a dict."""
        if self.state.current_state is None:
            return None
        return self.state.current_state.to_dict()

    def get_recent_changes(
        self,
        count: int = 50,
        category: str | None = None,
        min_significance: str | None = None,
    ) -> list[dict]:
        """Get recent changes from the log."""
        return self.state.get_recent_changes(count, category, min_significance)

    def get_screenshot_base64(self) -> str | None:
        """Get the latest screenshot as compressed base64 JPEG."""
        if self._last_frame is None:
            return None
        return self._last_frame.compressed_base64(max_kb=self.config.capture.max_image_size_kb)

    def get_status(self) -> dict:
        """Get the full status of the observation system."""
        window = self.capturer.target_window
        return {
            "running": self._running,
            "backend": self.config.model.backend,
            "model": self.config.model.model_name,
            "capture_interval": self.config.capture.interval,
            "target_window": window.to_dict() if window else None,
            "frames_captured": self.capturer.frame_count,
            "state": self.state.get_state_summary(),
            "recent_errors": self._errors[-5:] if self._errors else [],
        }

    def get_scene_history(self) -> list[dict]:
        """Get the history of scene transitions."""
        return self.state.get_scene_history()

    def list_available_windows(self) -> list[dict]:
        """List all visible windows that can be targeted."""
        windows = enumerate_windows()
        return [w.to_dict() for w in windows]

    def list_available_monitors(self) -> list[dict]:
        """List available monitors."""
        return list_monitors()

    def set_target_window(self, title: str | None = None, process_name: str | None = None) -> dict:
        """Change the target window at runtime."""
        self.config.capture.window_title = title
        self.config.capture.process_name = process_name
        self.capturer.config = self.config.capture
        window = self.capturer.refresh_window()
        if window:
            return {"success": True, "window": window.to_dict()}
        return {"success": False, "error": "No matching window found"}

    def set_interval(self, interval: float) -> None:
        """Change the capture interval at runtime."""
        self.config.capture.interval = max(0.1, min(30.0, interval))

    async def close(self) -> None:
        """Shut down the orchestrator and release resources."""
        await self.stop()
        await self.observer.close()
        logger.info("Orchestrator shut down")
