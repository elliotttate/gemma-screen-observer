"""Configuration management for Gemma Screen Observer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]


class CaptureConfig(BaseModel):
    """Screen capture settings."""

    interval: float = Field(default=1.0, ge=0.1, le=30.0, description="Seconds between captures")
    monitor: int | str = Field(
        default="primary",
        description="Monitor: 'all', 'primary', or 1-based index. Ignored when window_title or process_name is set.",
    )
    window_title: str | None = Field(
        default=None,
        description="Target a specific window by title substring (case-insensitive). Enables background capture.",
    )
    process_name: str | None = Field(
        default=None,
        description="Target a specific window by process name (e.g. 'game.exe' or 'game'). Enables background capture.",
    )
    window_index: int | None = Field(
        default=None,
        description="When multiple windows match, select which one (1-based). None = auto-select first.",
    )
    region: tuple[int, int, int, int] | None = Field(
        default=None, description="Capture region as (left, top, width, height). None = full window/monitor"
    )
    resize: tuple[int, int] = Field(
        default=(1280, 720), description="Resize captured frames to this resolution for inference"
    )
    max_image_size_kb: int = Field(
        default=950, description="Max image size in KB for MCP transfer. Images are compressed progressively."
    )
    save_screenshots: bool = Field(
        default=False, description="Save screenshots to disk alongside analysis"
    )
    screenshot_dir: str = Field(default="screenshots", description="Directory for saved screenshots")


class ModelConfig(BaseModel):
    """Gemma 4 model backend settings."""

    backend: Literal["ollama", "google_ai"] = Field(
        default="ollama", description="Inference backend"
    )
    model_name: str = Field(default="gemma4", description="Model name/identifier")
    endpoint: str = Field(
        default="http://localhost:11434", description="API endpoint (for Ollama backend)"
    )
    api_key: str | None = Field(
        default=None, description="API key (for Google AI backend). Falls back to GOOGLE_API_KEY env var"
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, ge=256, le=8192, description="Max output tokens")
    timeout: float = Field(default=120.0, description="Request timeout in seconds (first call loads model into memory, needs extra time)")


class LogConfig(BaseModel):
    """State and change log settings."""

    max_entries: int = Field(default=5000, description="Maximum change log entries to keep in memory")
    persist: bool = Field(default=True, description="Persist change log to disk")
    output_file: str = Field(default="game_log.jsonl", description="JSONL log file path")
    log_screenshots: bool = Field(
        default=False, description="Include base64 screenshots in log entries"
    )


class McpConfig(BaseModel):
    """MCP server settings."""

    name: str = Field(default="gemma-screen-observer", description="MCP server name")
    transport: Literal["stdio"] = Field(default="stdio", description="MCP transport")


class ObserverConfig(BaseModel):
    """Root configuration."""

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> ObserverConfig:
        """Load configuration from a TOML file."""
        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ObserverConfig:
        """Load config from file if it exists, otherwise return defaults."""
        if path is not None:
            return cls.from_toml(path)

        # Check common locations
        candidates = [
            Path("config.toml"),
            Path("gemma-screen-observer.toml"),
            Path.home() / ".config" / "gemma-screen-observer" / "config.toml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return cls.from_toml(candidate)

        return cls()
