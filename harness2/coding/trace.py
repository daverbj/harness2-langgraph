from __future__ import annotations

import json
import os
import select
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.text import Text


class ReliableWriter:
    """Handle nonblocking terminal backpressure without losing a coding run."""

    def __init__(self, underlying):
        self.underlying = underlying

    def __getattr__(self, name):
        return getattr(self.underlying, name)

    def write(self, text):
        try:
            descriptor = self.underlying.fileno()
        except (AttributeError, OSError):
            return self.underlying.write(text)
        data = text.encode(getattr(self.underlying, "encoding", None) or "utf-8", errors="replace")
        position = 0
        while position < len(data):
            try:
                position += os.write(descriptor, data[position:])
            except BlockingIOError:
                select.select([], [descriptor], [], 1)
            except BrokenPipeError:
                return len(text)
        return len(text)

    def flush(self):
        try:
            self.underlying.flush()
        except (BlockingIOError, BrokenPipeError):
            pass


def reliable_output():
    if not isinstance(sys.stdout, ReliableWriter):
        try:
            sys.stdout.flush()
        except BlockingIOError:
            pass
        sys.stdout = ReliableWriter(sys.stdout)


_LOCK = threading.RLock()
_ACTIVE_STREAM = None
_REPLAY_STREAM = None
CONSOLE = Console(highlight=False)
STYLES = {
    "stage": "bold cyan",
    "tool": "yellow",
    "result": "green",
    "error": "bold red",
    "diff": "magenta",
    "model": "blue",
    "quality": "bold green",
    "retry": "bold yellow",
}


class Trace:
    def __init__(self, path: Path, run: str, *, full: bool = False, secrets=(), thinking=False):
        self.path, self.run, self.full = path, run, full
        self.started = time.monotonic()
        self.thinking = thinking
        self._stream_buffers = {}
        self._last_stream_channel = None
        self.secrets = tuple(s for s in secrets if s)
        path.parent.mkdir(parents=True, exist_ok=True)

    def clean(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {str(k): self.clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        return value

    def _write(self, event):
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def stream(self, channel, text):
        if channel == "thinking" and not self.thinking:
            return
        with _LOCK:
            prior = self._last_stream_channel
            if prior and prior != channel and self._stream_buffers.get(prior):
                self._stream_output(prior, self.clean(self._stream_buffers[prior]))
                self._stream_buffers[prior] = ""
            self._last_stream_channel = channel
            buffer = self.clean(self._stream_buffers.get(channel, "") + text)
            # Hold only a suffix that could become a secret when the next chunk
            # arrives. Ordinary tokens must not wait behind a fixed-size buffer.
            keep = max(
                (
                    length
                    for secret in self.secrets
                    for length in range(1, min(len(secret), len(buffer) + 1))
                    if buffer.endswith(secret[:length])
                ),
                default=0,
            )
            output = buffer[:-keep] if keep else buffer
            self._stream_buffers[channel] = buffer[-keep:] if keep else ""
            if output:
                self._stream_output(channel, output)

    def _stream_output(self, channel, output):
        global _ACTIVE_STREAM
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "run": self.run,
            "event": channel + "_delta",
            "message": channel,
            "content": output,
        }
        self._write(event)
        identity = (str(self.path), channel)
        if _ACTIVE_STREAM != identity:
            if _ACTIVE_STREAM is not None:
                CONSOLE.print()
            style = (
                "dim cyan"
                if channel == "thinking"
                else "green"
                if channel == "command"
                else "bright_white"
            )
            CONSOLE.print(Text(f"[{self.run}] {channel.upper()} ", style=style), end="")
            _ACTIVE_STREAM = identity
        CONSOLE.print(
            Text(output, style="dim cyan" if channel == "thinking" else None),
            end="",
            soft_wrap=True,
        )
        CONSOLE.file.flush()

    def end_stream(self):
        global _ACTIVE_STREAM
        with _LOCK:
            for channel, buffer in list(self._stream_buffers.items()):
                if buffer:
                    self._stream_output(channel, self.clean(buffer))
                self._stream_buffers[channel] = ""
            if _ACTIVE_STREAM and _ACTIVE_STREAM[0] == str(self.path):
                CONSOLE.print()
                _ACTIVE_STREAM = None

    def emit(self, kind: str, message: str, **data):
        global _ACTIVE_STREAM
        with _LOCK:
            event = self.clean(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": round(time.monotonic() - self.started, 3),
                    "run": self.run,
                    "event": kind,
                    "message": message,
                    **data,
                }
            )
            self.end_stream()
            if _ACTIVE_STREAM is not None:
                CONSOLE.print()
                _ACTIVE_STREAM = None
            self._write(event)
            show_event(event, full=self.full)


def show_event(event, *, full=False):
    global _REPLAY_STREAM
    kind = event["event"]
    if kind.endswith("_delta"):
        identity = (event["run"], kind)
        if _REPLAY_STREAM != identity:
            if _REPLAY_STREAM:
                CONSOLE.print()
            CONSOLE.print(
                Text(f"[{event['run']}] {kind.removesuffix('_delta').upper()} ", style="cyan"),
                end="",
            )
            _REPLAY_STREAM = identity
        CONSOLE.print(
            Text(event.get("content", ""), style="dim cyan" if kind == "thinking_delta" else None),
            end="",
        )
        return
    if _REPLAY_STREAM:
        CONSOLE.print()
        _REPLAY_STREAM = None
    if kind == "assistant" and event.get("streamed"):
        return
    prefix = f"[{event['run']}] {kind.upper()} "
    CONSOLE.print(Text(prefix, style=STYLES.get(kind, "white")) + Text(event["message"]))
    for key in ("args", "output", "diff", "content"):
        if key not in event or event.get("streamed") and key == "content":
            continue
        value = event[key]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if not full and len(text) > 1800:
            text = text[:1800] + "\n… full content in trace.jsonl (--full-trace to print it)"
        CONSOLE.print(Text(text), soft_wrap=True)
