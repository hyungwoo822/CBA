"""CLI commands for brain-agent."""
from __future__ import annotations

import asyncio
import sys


def main():
    """Entry point for brain-agent CLI."""
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]
    if command == "run":
        asyncio.run(run_agent())
    elif command == "memory":
        if len(sys.argv) > 2 and sys.argv[2] == "stats":
            asyncio.run(show_memory_stats())
        else:
            print("Usage: brain-agent memory stats")
    elif command == "version":
        from brain_agent import __version__

        print(f"brain-agent v{__version__}")
    elif command == "dashboard":
        port = 3000
        if len(sys.argv) > 2 and sys.argv[2] == "--port" and len(sys.argv) > 3:
            port = int(sys.argv[3])
        run_dashboard(port)
    elif command == "serve":
        asyncio.run(serve_all())
    elif command in ("--help", "-h"):
        print_help()
    else:
        print(f"Unknown command: {command}")
        print_help()


def print_help():
    print(
        """brain-agent -- Neuroscience-faithful AI agent framework

Usage:
  brain-agent run                  Start the agent (interactive)
  brain-agent serve                Start all enabled channels (dashboard + telegram + discord)
  brain-agent dashboard [--port N] Start the dashboard server (default port 3000)
  brain-agent memory stats         Show memory statistics
  brain-agent version              Show version
  brain-agent --help               Show this help
"""
    )


async def run_agent():
    """Interactive agent loop."""
    import os
    from dotenv import load_dotenv
    from brain_agent.agent import BrainAgent

    load_dotenv()
    model = os.environ.get("BRAIN_AGENT_MODEL", None)
    print("brain-agent interactive mode (Ctrl+C to exit)")
    print("Type your messages below:\n")

    async with BrainAgent(use_mock_embeddings=True, model=model) as agent:
        while True:
            try:
                text = input("> ")
                if not text.strip():
                    continue
                if text.strip().lower() in ("exit", "quit", "/quit"):
                    break
                result = await agent.process(text)
                print(f"\n{result.response}\n")
                print(
                    f"  [mode: {result.network_mode} | signals: {result.signals_processed}]\n"
                )
            except (KeyboardInterrupt, EOFError):
                break

    print("\nGoodbye!")


async def show_memory_stats():
    """Show memory statistics."""
    from brain_agent.agent import BrainAgent

    async with BrainAgent(use_mock_embeddings=True) as agent:
        stats = await agent.memory.stats()
        print("Memory Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


async def serve_all():
    """Start dashboard + all enabled channel adapters concurrently."""
    import os
    import signal
    from dotenv import load_dotenv
    from brain_agent.agent import BrainAgent

    load_dotenv()
    model = os.environ.get("BRAIN_AGENT_MODEL", None)
    adapters = []

    async with BrainAgent(use_mock_embeddings=True, model=model) as agent:
        # Telegram
        tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if tg_token:
            from brain_agent.channels.telegram_adapter import TelegramAdapter
            tg = TelegramAdapter(agent, token=tg_token)
            await tg.start()
            adapters.append(tg)
            print("[serve] Telegram bot started")

        # Discord
        dc_token = os.environ.get("DISCORD_BOT_TOKEN", "")
        if dc_token:
            from brain_agent.channels.discord_adapter import DiscordAdapter
            dc = DiscordAdapter(agent, token=dc_token)
            await dc.start()
            adapters.append(dc)
            print("[serve] Discord bot started")

        if not adapters:
            print("[serve] No channel tokens found. Set TELEGRAM_BOT_TOKEN or DISCORD_BOT_TOKEN in .env")
            return

        print(f"[serve] {len(adapters)} channel(s) active. Ctrl+C to stop.")

        # Wait until interrupted
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        await stop_event.wait()

        # Graceful shutdown
        print("\n[serve] Shutting down...")
        for adapter in adapters:
            await adapter.stop()

    print("[serve] Done.")


def run_dashboard(port: int = 3000):
    """Start the dashboard server, serving the React build as static files."""
    import subprocess
    from pathlib import Path
    from brain_agent.dashboard.server import run_server

    # Find dashboard directory (relative to package or cwd)
    candidates = [
        Path(__file__).parent.parent.parent / "dashboard",  # dev: repo root
        Path.cwd() / "dashboard",
    ]
    dashboard_dir = None
    for c in candidates:
        if (c / "package.json").exists():
            dashboard_dir = c
            break

    static_dir = None
    if dashboard_dir:
        dist_dir = dashboard_dir / "dist"
        if not dist_dir.exists():
            print("Building dashboard frontend...")
            subprocess.run(["npm", "run", "build"], cwd=str(dashboard_dir), check=True)
        if dist_dir.exists():
            static_dir = str(dist_dir)
            print(f"Serving dashboard from {static_dir}")

    print(f"\n  Brain Agent Dashboard: http://localhost:{port}")
    print(f"  WebSocket endpoint:    ws://localhost:{port}/ws")
    print(f"  API status:            http://localhost:{port}/api/status\n")
    run_server(port=port, static_dir=static_dir)
