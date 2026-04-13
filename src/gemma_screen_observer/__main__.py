"""CLI entry point for the Gemma Screen Observer MCP server."""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gemma-screen-observer",
        description="Vision-based screen observer for automated game testing, exposed as an MCP server.",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to config.toml (default: auto-detect)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write logs to a file instead of stderr",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List visible windows and exit (Windows only)",
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="List available monitors and exit",
    )
    args = parser.parse_args()

    # Configure logging — logs go to stderr (or file) so they don't interfere with MCP stdio
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_handlers: list[logging.Handler] = []
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    else:
        log_handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=log_handlers,
    )

    from .config import ObserverConfig

    config = ObserverConfig.load(args.config)

    # Quick utility modes
    if args.list_windows:
        from .capture import enumerate_windows
        windows = enumerate_windows()
        for i, w in enumerate(windows, 1):
            print(f"  {i}. [{w.process_name}] {w.title} (PID {w.pid})")
        print(f"\n{len(windows)} windows found.")
        return

    if args.list_monitors:
        from .capture import list_monitors
        monitors = list_monitors()
        for m in monitors:
            print(f"  {m['label']}: {m['width']}x{m['height']} at ({m['left']}, {m['top']})")
        return

    # Start MCP server
    from .mcp_server import create_server

    logger = logging.getLogger(__name__)
    logger.info("Starting Gemma Screen Observer MCP server")
    logger.info("Backend: %s, Model: %s", config.model.backend, config.model.model_name)
    if config.capture.window_title or config.capture.process_name:
        logger.info(
            "Window target: title=%r process=%r",
            config.capture.window_title,
            config.capture.process_name,
        )
    else:
        logger.info("Capture target: monitor %s", config.capture.monitor)

    mcp = create_server(config)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
