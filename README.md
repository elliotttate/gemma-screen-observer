# Gemma Screen Observer

An MCP server that uses **Gemma 4** vision to observe game screens in real-time, describe what's happening, and maintain a structured change log that other MCP clients can query.

Designed for **automated game testing** — point it at a game window and it will continuously capture, analyze, and log game state transitions, UI changes, and on-screen events.

## Features

- **Background window capture** — Captures game windows even when minimized or behind other windows (Windows, via `PrintWindow` API with `PW_RENDERFULLCONTENT`, same approach as OBS Studio)
- **Gemma 4 vision analysis** — Each frame is analyzed for scene type, visible elements, UI state, text, and environment
- **Change detection** — Consecutive frames are compared to detect meaningful game state changes
- **Structured state log** — All observations and changes are logged as timestamped JSONL with ISO 8601 timestamps
- **MCP interface** — Full set of tools and resources for other MCP clients to start/stop observation, query state, and read the change log
- **Multiple backends** — Run Gemma 4 locally via Ollama or in the cloud via Google AI
- **Progressive image compression** — Screenshots are automatically compressed for efficient MCP transfer

## Quick Start

### 1. Install

```bash
pip install -e .
```

### 2. Set up Gemma 4

**Option A: Ollama (local, recommended)**
```bash
ollama pull gemma4
ollama serve
```

**Option B: Google AI**
```bash
export GOOGLE_API_KEY=your-key-here
```

### 3. Configure

```bash
cp config.example.toml config.toml
# Edit config.toml — set window_title or process_name for your game
```

### 4. Run

```bash
gemma-screen-observer
```

Or add it to your MCP client config (e.g., Claude Code `settings.json`):

```json
{
  "mcpServers": {
    "game-observer": {
      "command": "gemma-screen-observer",
      "args": ["-c", "path/to/config.toml"]
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `start_observation` | Begin capturing and analyzing frames at the configured interval |
| `stop_observation` | Stop the observation loop |
| `take_snapshot` | Force an immediate capture and analysis |
| `get_state` | Get the current game state (scene, elements, UI, text) |
| `get_changes` | Get recent changes with optional filtering by category/significance |
| `query_screen` | Ask a free-form question about what's on screen |
| `get_screenshot` | Get the latest screenshot as base64 JPEG |
| `get_status` | Full status of the observation system |
| `list_windows` | List all visible windows available for capture |
| `list_monitors` | List available monitors |
| `set_target_window` | Change the target window at runtime |
| `set_interval` | Change the capture interval |
| `get_scene_history` | Get history of scene transitions |
| `clear_state` | Reset all observation state |

## MCP Resources

| URI | Description |
|-----|-------------|
| `observer://state/current` | Current game state |
| `observer://log/changes` | Recent change log |
| `observer://status` | Observation system status |
| `observer://scenes` | Scene transition history |

## State Output Format

Each frame analysis produces structured JSON:

```json
{
  "timestamp": 1713024000.0,
  "time": "2026-04-13T16:00:00+00:00",
  "frame_number": 42,
  "scene": "combat",
  "description": "Player character fighting a dragon in a mountain area",
  "elements": {
    "player": {"visible": true, "health": "75%", "position": "center-left", "action": "attacking"},
    "enemies": [{"type": "dragon", "health": "50%", "position": "right", "action": "breathing fire"}],
    "ui": {
      "health_bar": "75%",
      "mana_bar": "100%",
      "minimap": "visible",
      "score": "1500"
    },
    "environment": {"location": "mountain peak", "time_of_day": "sunset"}
  },
  "text_on_screen": ["Level 5", "Score: 1500"]
}
```

Change entries include significance levels:

```json
{
  "timestamp": 1713024001.0,
  "time": "2026-04-13T16:00:01+00:00",
  "frame_number": 43,
  "category": "player",
  "element": "health",
  "from": "75%",
  "to": "60%",
  "significance": "high",
  "summary": "Player took damage from dragon fire breath"
}
```

## Configuration

See [`config.example.toml`](config.example.toml) for all options. Key settings:

```toml
[capture]
interval = 1.0                          # Seconds between captures
window_title = "Slay the Spire"         # Target by window title
# process_name = "SlayTheSpire"         # Or target by process name
resize = [1280, 720]                    # Resize for inference

[model]
backend = "ollama"                      # "ollama" or "google_ai"
model_name = "gemma4"
endpoint = "http://localhost:11434"     # Ollama endpoint
```

## CLI Utilities

```bash
# List all visible windows (useful for finding your game)
gemma-screen-observer --list-windows

# List monitors
gemma-screen-observer --list-monitors

# Run with verbose logging
gemma-screen-observer -v

# Log to file
gemma-screen-observer --log-file observer.log
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│   Screen     │────▶│   Gemma 4    │────▶│    State      │────▶│  MCP Server │
│   Capture    │     │   Observer   │     │    Manager    │     │  (FastMCP)  │
│              │     │              │     │              │     │             │
│ - PrintWindow│     │ - Analyze    │     │ - Snapshots  │     │ - Tools     │
│ - mss        │     │ - Detect Δ   │     │ - Change Log │     │ - Resources │
│ - Window enum│     │ - Query      │     │ - JSONL file │     │ - stdio     │
└─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘
       ▲                    ▲                                         │
       │                    │                                         │
       └────────────────────┴──────── Orchestrator ◀──────────────────┘
```

## Requirements

- Python 3.11+
- Windows (for background window capture via PrintWindow; monitor capture works cross-platform)
- Gemma 4 via [Ollama](https://ollama.com) or Google AI API key

## License

MIT
