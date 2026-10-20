from __future__ import annotations

import difflib
import fnmatch
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

IGNORED = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".h2",
    ".DS_Store",
}
SECRET_PATTERNS = (".env", ".env.*", "*.pem", "*.key", "credentials.json")


def excluded(path: Path, *, secrets=True):
    return any(part in IGNORED for part in path.parts) or (
        secrets
        and any(
            fnmatch.fnmatch(part, pattern) for part in path.parts for pattern in SECRET_PATTERNS
        )
    )


def safe_path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("Use a relative path inside the workspace")
    path = (root / raw).resolve()
    if not path.is_relative_to(root.resolve()) or excluded(raw):
        raise ValueError("Path is outside the permitted workspace or is excluded")
    return path


def snapshot(source: Path, destination: Path, *, excluded_roots=()):
    source, destination = source.resolve(), destination.resolve()
    blocked = tuple(Path(p).resolve() for p in excluded_roots) + (destination,)

    def ignore(directory, names):
        ignored = []
        for name in names:
            path = Path(directory) / name
            if (
                excluded(Path(name))
                or path.is_symlink()
                or any(path.resolve() == b or path.resolve().is_relative_to(b) for b in blocked)
            ):
                ignored.append(name)
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def iter_files(root: Path):
    """Walk lazily and prune excluded directories before entering them."""
    if root.is_file():
        if not root.is_symlink() and not excluded(Path(root.name)):
            yield root
        return
    for directory, folders, names in os.walk(root, followlinks=False):
        current = Path(directory)
        folders[:] = sorted(
            name
            for name in folders
            if not excluded((current / name).relative_to(root))
            and not (current / name).is_symlink()
        )
        for name in sorted(names):
            file = current / name
            if not file.is_symlink() and not excluded(file.relative_to(root)):
                yield file


def text_files(root: Path):
    result = {}
    for path in iter_files(root):
        if not path.is_file() or path.is_symlink() or excluded(path.relative_to(root)):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        try:
            result[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeError:
            pass
    return result


def patch_between(before: Path, after: Path):
    old, new = text_files(before), text_files(after)
    chunks = []
    for name in sorted(old.keys() | new.keys()):
        if old.get(name) == new.get(name):
            continue
        chunks.extend(
            difflib.unified_diff(
                old.get(name, "").splitlines(keepends=True),
                new.get(name, "").splitlines(keepends=True),
                fromfile=f"a/{name}" if name in old else "/dev/null",
                tofile=f"b/{name}" if name in new else "/dev/null",
            )
        )
    return "".join(chunks)


def child_environment():
    # Models run in the parent; tools receive no API credentials by default.
    environment = {
        k: v
        for k, v in os.environ.items()
        if not any(
            word in k.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        )
        and k not in ("BASH_ENV", "ENV")
    }
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def command(
    command: str, workspace: Path, timeout: int = 60, *, output_limit=24000, cancel_event=None
):
    """Drain to a file rather than risking unbounded captured subprocess output."""
    with tempfile.TemporaryFile() as output:
        environment = child_environment()
        environment["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
        )
        environment["PYTHONPATH"] = os.pathsep.join((str(workspace), str(workspace / "src")))
        process = subprocess.Popen(
            command,
            cwd=workspace,
            shell=True,
            executable="/bin/bash",
            stdout=output,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        timed_out = False
        try:
            if cancel_event is None:
                process.wait(timeout=timeout)
            else:
                deadline = time.monotonic() + timeout if timeout else float("inf")
                while process.poll() is None:
                    if cancel_event.is_set():
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        raise InterruptedError("Validation cancelled")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        process.wait(timeout=min(0.2, remaining))
                    except subprocess.TimeoutExpired:
                        continue
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        output.seek(0, 2)
        size = output.tell()
        output.seek(max(0, size - output_limit))
        text = output.read(output_limit).decode("utf-8", errors="replace")
        if size > output_limit:
            text = "[Output truncated; showing last bytes]\n" + text
    return {"exit_code": process.returncode, "timed_out": timed_out, "output": text}


async def async_command(command_text, workspace, timeout=60, *, output_callback=None):
    """Stream command output and terminate its whole process group on cancellation."""
    import asyncio
    import codecs

    environment = child_environment()
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    environment["PYTHONPATH"] = os.pathsep.join((str(workspace), str(workspace / "src")))
    process = await asyncio.create_subprocess_exec(
        "/bin/bash",
        "-o",
        "pipefail",
        "-c",
        command_text,
        cwd=workspace,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    chunks, size = [], 0

    async def read():
        nonlocal size
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while data := await process.stdout.read(2048):
            text = decoder.decode(data)
            if text:
                chunks.append(text)
                size += len(text)
                while size > 24000 and len(chunks) > 1:
                    size -= len(chunks.pop(0))
                if output_callback:
                    output_callback(text)
        tail = decoder.decode(b"", final=True)
        if tail:
            chunks.append(tail)
            if output_callback:
                output_callback(tail)

    reader = asyncio.create_task(read())
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout or None)
        await reader
    except asyncio.TimeoutError:
        timed_out = True
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
        await reader
    except BaseException:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        raise
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "output": "".join(chunks)[-24000:],
    }
