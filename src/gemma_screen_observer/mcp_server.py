"""MCP server exposing screen observation tools and resources.

Other MCP clients can connect to this server to:
- Start/stop screen observation
- Query the current game state
- Get the change log
- Take on-demand snapshots
- Ask free-form questions about the screen
- Target specific windows for background capture
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from .config import ObserverConfig
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# The orchestrator is initialized when the server starts.
# It's stored at module level so all tools/resources can access it.
_orchestrator: Orchestrator | None = None
_config: ObserverConfig | None = None


def create_server(config: ObserverConfig) -> FastMCP:
    """Create and configure the MCP server with all tools and resources."""
    global _orchestrator, _config
    _config = config
    _orchestrator = Orchestrator(config)

    mcp = FastMCP(
        config.mcp.name,
        instructions=(
            "Game screen observer powered by Gemma 4 vision. "
            "Captures screenshots (including minimized/background windows on Windows) "
            "and uses Gemma 4 to describe game state and detect changes over time. "
            "Use start_observation to begin, then query state and changes as needed."
        ),
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def start_observation() -> str:
        """Start observing the screen at the configured interval.

        If a window_title or process_name is configured, captures that specific
        window in the background (even if minimized). Otherwise captures the
        configured monitor.
        """
        await _orchestrator.start()
        status = _orchestrator.get_status()
        return json.dumps({"status": "started", **status}, indent=2)

    @mcp.tool()
    async def stop_observation() -> str:
        """Stop the observation loop."""
        await _orchestrator.stop()
        return json.dumps({"status": "stopped", "frames_captured": _orchestrator.capturer.frame_count})

    @mcp.tool()
    async def get_state() -> str:
        """Get the current game state as analyzed by Gemma 4.

        Returns structured data including scene type, visible elements,
        UI state, and text on screen.
        """
        state = _orchestrator.get_current_state()
        if state is None:
            return json.dumps({"error": "No state available. Start observation or take a snapshot first."})
        return json.dumps(state, indent=2)

    @mcp.tool()
    async def get_changes(
        count: int = 50,
        category: str | None = None,
        min_significance: str | None = None,
    ) -> str:
        """Get recent changes detected in the game.

        Args:
            count: Number of recent changes to return (default 50)
            category: Filter by category (ui, player, enemy, environment, scene, text, animation)
            min_significance: Minimum significance level (low, medium, high, critical)
        """
        changes = _orchestrator.get_recent_changes(count, category, min_significance)
        return json.dumps({"count": len(changes), "changes": changes}, indent=2)

    @mcp.tool()
    async def take_snapshot() -> str:
        """Force an immediate capture and analysis right now.

        Captures a fresh screenshot, analyzes it with Gemma 4, and returns
        the full state and any changes since the last frame.
        """
        result = await _orchestrator.take_snapshot()
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def query_screen(question: str) -> str:
        """Ask a free-form question about what's currently on screen.

        Uses Gemma 4 to analyze the latest screenshot and answer your question.

        Args:
            question: Any question about the current screen contents
        """
        answer = await _orchestrator.query_screen(question)
        return json.dumps({"question": question, "answer": answer})

    @mcp.tool()
    async def get_screenshot() -> str:
        """Get the latest screenshot as a base64-encoded JPEG.

        Returns the compressed image data suitable for display or further analysis.
        """
        b64 = _orchestrator.get_screenshot_base64()
        if b64 is None:
            return json.dumps({"error": "No screenshot available. Start observation or take a snapshot first."})
        return json.dumps({"format": "jpeg", "encoding": "base64", "data": b64})

    @mcp.tool()
    async def get_status() -> str:
        """Get the full status of the observation system.

        Includes running state, backend info, capture stats, target window,
        and recent errors.
        """
        return json.dumps(_orchestrator.get_status(), indent=2)

    @mcp.tool()
    async def list_windows() -> str:
        """List all visible windows that can be targeted for capture.

        Returns window titles, process names, and handles. Use set_target_window
        to select one for observation.
        """
        windows = _orchestrator.list_available_windows()
        return json.dumps({"count": len(windows), "windows": windows}, indent=2)

    @mcp.tool()
    async def list_monitors() -> str:
        """List available monitors and their dimensions."""
        monitors = _orchestrator.list_available_monitors()
        return json.dumps({"count": len(monitors), "monitors": monitors}, indent=2)

    @mcp.tool()
    async def set_target_window(
        window_title: str | None = None,
        process_name: str | None = None,
    ) -> str:
        """Change the target window for capture at runtime.

        Uses PrintWindow for background capture — the target window can be
        minimized or behind other windows.

        Args:
            window_title: Window title substring to match (case-insensitive)
            process_name: Process name to match (e.g. 'game' or 'game.exe')
        """
        if not window_title and not process_name:
            return json.dumps({"error": "Provide at least one of window_title or process_name"})
        result = _orchestrator.set_target_window(title=window_title, process_name=process_name)
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def set_interval(interval: float) -> str:
        """Change the capture interval in seconds.

        Args:
            interval: Seconds between captures (0.1 to 30.0)
        """
        _orchestrator.set_interval(interval)
        return json.dumps({"interval": _orchestrator.config.capture.interval})

    @mcp.tool()
    async def get_scene_history() -> str:
        """Get the history of scene transitions (e.g. menu → gameplay → combat).

        Useful for understanding the flow of gameplay over time.
        """
        history = _orchestrator.get_scene_history()
        return json.dumps({"transitions": len(history), "history": history}, indent=2)

    @mcp.tool()
    async def clear_state() -> str:
        """Clear all observation state and change history.

        Use this to reset when starting observation of a new game or session.
        """
        _orchestrator.state.clear()
        return json.dumps({"status": "cleared"})

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    @mcp.resource("observer://state/current")
    async def resource_current_state() -> str:
        """Current game state as analyzed by Gemma 4."""
        state = _orchestrator.get_current_state()
        if state is None:
            return json.dumps({"status": "no_data", "message": "Observation not started"})
        return json.dumps(state, indent=2)

    @mcp.resource("observer://log/changes")
    async def resource_change_log() -> str:
        """Recent change log entries."""
        changes = _orchestrator.get_recent_changes(100)
        return json.dumps({"count": len(changes), "changes": changes}, indent=2)

    @mcp.resource("observer://status")
    async def resource_status() -> str:
        """Observation system status."""
        return json.dumps(_orchestrator.get_status(), indent=2)

    @mcp.resource("observer://scenes")
    async def resource_scene_history() -> str:
        """Scene transition history."""
        history = _orchestrator.get_scene_history()
        return json.dumps({"transitions": len(history), "history": history}, indent=2)

    return mcp
