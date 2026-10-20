"""Persistent streaming coding chat; /improve delegates to real Harness2."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_to_dict
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from harness2 import config
from harness2.core import tree
from harness2.core.library import Library
from harness2.core.schedule import mode_token, run_task

from .adapter import DEFAULT_HARNESS, SURFACE, CodingAdapter, CodingProposer
from .agent import Budget, make_model, run_agent
from .files import async_command, iter_files, patch_between, safe_path, snapshot
from .report import build_report
from .trace import CONSOLE, Trace


def private_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(data, file, indent=2)
    os.replace(temporary, path)


def file_hash(path):
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def changes(before, after):
    old = {p.relative_to(before).as_posix(): p for p in iter_files(before)}
    new = {p.relative_to(after).as_posix(): p for p in iter_files(after)}
    return [
        (name, old.get(name), new.get(name))
        for name in sorted(old.keys() | new.keys())
        if (file_hash(old[name]) if name in old else None)
        != (file_hash(new[name]) if name in new else None)
    ]


def apply_changes(before, after, target, backup):
    """Preflight every changed path before copying binary/text files or deletions."""
    edits = changes(before, after)
    for name, old, new in edits:
        destination = safe_path(target, name)
        actual = file_hash(destination)
        expected = file_hash(old) if old else None
        wanted = file_hash(new) if new else None
        if actual != expected and actual != wanted:
            raise ValueError(
                f"Conflict: {name} changed outside this session; no changes were applied"
            )
    backup.mkdir(parents=True, exist_ok=True)
    for name, _, new in edits:
        destination = safe_path(target, name)
        if destination.exists():
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
        if new:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".h2-tmp-" + uuid4().hex[:8])
            shutil.copy2(new, temporary)
            os.replace(temporary, destination)
        else:
            destination.unlink(missing_ok=True)
    private_json(backup / "manifest.json", {"changes": [name for name, _, _ in edits]})
    return len(edits)


def strip_outer_quotes(text):
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1]
    return text


class ChatSession:
    def __init__(self, root, settings, key):
        self.root, self.settings, self.key = root, settings, key
        self.workspace, self.harness = root / "workspace", root / "harness"
        self.conversation = root / "conversation.json"
        self.budget = Budget(settings.get("max_calls", 0))
        usage = settings.get("usage", {})
        self.budget.calls = usage.get("calls", 0)
        self.budget.input_tokens = usage.get("input_tokens", 0)
        self.budget.output_tokens = usage.get("output_tokens", 0)
        self.cancel = threading.Event()
        self.active = None
        self.current_trace = None

    @classmethod
    def open(cls, args):
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY in .env before starting chat")
        if args.session:
            root = args.session.expanduser().resolve()
            settings = json.loads((root / "session.json").read_text())
        else:
            origin = None
            if args.from_run:
                origin = args.from_run.expanduser().resolve()
                old = json.loads((origin / "config.json").read_text())
                candidates = sorted(
                    (origin / "executions").glob("*/*/*/workspace"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if not candidates:
                    raise ValueError("No retained workspace exists in that run")
                source = candidates[0]
                repository = Path(old["repo"]).expanduser()
                if not repository.is_absolute():
                    repository = Path(old.get("cwd", config.ROOT)) / repository
            else:
                if not args.repo:
                    raise ValueError("Provide --repo PATH, --from-run RUN, or --session SESSION")
                repository = args.repo.expanduser().resolve()
                source = repository
                old = {}
            if not source.is_dir():
                raise ValueError(f"Project directory does not exist: {source}")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
            root = (args.output or Path("runs/chats") / stamp).expanduser().resolve()
            if root.exists():
                raise ValueError("Choose a new --output directory, or reopen it with --session")
            root.mkdir(parents=True, mode=0o700)
            blocked = (root,)
            if root.is_relative_to(source) and root.relative_to(source).parts[0] == "runs":
                blocked = (source / "runs",)
            baseline = origin / "source" if origin else source
            snapshot(baseline, root / "original", excluded_roots=blocked)
            snapshot(source, root / "workspace", excluded_roots=blocked)
            active_harness = candidates[0].parent / "harness" if origin else args.harness
            if active_harness and Path(active_harness).is_dir():
                shutil.copytree(active_harness, root / "harness")
            else:
                tree.write_tree(SURFACE, DEFAULT_HARNESS, root / "harness")
            prior_state = candidates[0].parent / "state.json" if origin else None
            if prior_state and prior_state.exists():
                shutil.copyfile(prior_state, root / "conversation.json")
                (root / "conversation.json").chmod(0o600)
            settings = {
                "repo": str(repository.resolve()),
                "model": old.get("model") or os.getenv("OPENROUTER_MODEL") or "minimax/minimax-m3",
                "max_tokens": 0,
                "max_steps": 0,
                "max_calls": 0,
                "tool_timeout": 180,
                "thinking": True,
                "k": 1,
                "mode": "sequential",
                "test_command": None,
                "turn": 0,
                "improvements": 0,
                "pending": None,
                "last_request": None,
                "mcp_config": None,
            }
            if origin:
                settings["test_command"] = old.get("test_command")
                if prior_state and prior_state.exists():
                    settings["pending"] = "chat"
                    settings["last_request"] = old.get("task", "Continue the unfinished task")
        for name in (
            "model",
            "max_tokens",
            "stream_idle_timeout",
            "max_steps",
            "max_calls",
            "tool_timeout",
            "thinking",
            "reasoning_effort",
            "provider",
            "k",
            "mode",
            "test_command",
        ):
            value = getattr(args, name, None)
            if value is not None:
                settings[name] = value
        if args.mcp_config:
            settings["mcp_config"] = str(args.mcp_config.expanduser().resolve())
        session = cls(root, settings, key)
        session.save()
        return session

    def trace(self, label, path=None):
        return Trace(
            path or self.root / "turns" / label / "trace.jsonl",
            label,
            secrets=(self.key,),
            thinking=self.settings["thinking"],
        )

    def save(self):
        self.settings["usage"] = {
            "calls": self.budget.calls,
            "input_tokens": self.budget.input_tokens,
            "output_tokens": self.budget.output_tokens,
        }
        private_json(self.root / "session.json", self.settings)

    def model(self):
        return make_model(
            self.settings["model"],
            self.key,
            self.settings["max_tokens"],
            stream_idle_timeout=self.settings.get("stream_idle_timeout", 0),
            reasoning_effort=self.settings.get("reasoning_effort"),
            provider=self.settings.get("provider"),
        )

    def mcp(self):
        path = self.settings.get("mcp_config")
        return json.loads(Path(path).read_text()) if path else None

    def append_exchange(self, request, response):
        from langchain_core.messages import messages_from_dict

        old = (
            json.loads(self.conversation.read_text())
            if self.conversation.exists()
            else {"messages": []}
        )
        messages = messages_from_dict(old["messages"])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            # Improvement uses a separate conversation; do not invalidate a pending tool turn.
            messages += [
                ToolMessage(
                    content="Interrupted by a new improvement request; no result was committed.",
                    tool_call_id=call["id"],
                )
                for call in messages[-1].tool_calls
            ]
        messages += [HumanMessage(content=request), AIMessage(content=response)]
        private_json(
            self.conversation,
            {"messages": messages_to_dict(messages), "steps": 0, "completed": True},
        )

    async def chat(self, text, retry=False):
        self.settings["turn"] += 1
        self.settings["pending"], self.settings["last_request"] = "chat", text
        self.save()
        trace = self.trace(f"chat-{self.settings['turn']:04d}")
        self.current_trace = trace
        answer = await run_agent(
            workspace=self.workspace,
            harness=self.harness,
            task=text,
            model=self.model,
            budget=self.budget,
            trace=trace,
            max_steps=self.settings["max_steps"],
            tool_timeout=self.settings["tool_timeout"],
            mcp_config=self.mcp(),
            state_file=self.conversation,
            new_turn=not retry,
            interactive=True,
            stream=True,
            cancel_event=self.cancel,
        )
        self.settings["pending"] = None
        self.save()
        return answer

    async def improve(self, text, retry=False):
        if retry and self.settings.get("pending_improvement"):
            output = Path(self.settings["pending_improvement"])
        else:
            self.settings["improvements"] += 1
            output = self.root / "improvements" / f"improve-{self.settings['improvements']:04d}"
            output.mkdir(parents=True)
            snapshot(self.workspace, output / "source")
        self.settings.update(pending="improve", last_request=text, pending_improvement=str(output))
        self.save()
        trace = self.trace("harness2", output / "trace.jsonl")
        self.current_trace = trace
        trace.emit(
            "stage",
            f"HARNESS2: snapshot → baseline → propose → probe → compose → final → learn; {output}",
        )
        adapter = CodingAdapter(
            root=output,
            source=output / "source",
            task=text,
            model=self.model,
            model_id=self.settings["model"],
            budget=self.budget,
            trace=trace,
            max_steps=self.settings["max_steps"],
            tool_timeout=self.settings["tool_timeout"],
            harness=self.harness,
            mcp_config=self.mcp(),
            stream=True,
            cancel_event=self.cancel,
            resume=retry,
        )
        proposer = CodingProposer(adapter)
        mode = mode_token(self.settings["mode"])
        library = Library(self.root / "library" / mode, "coding", SURFACE)
        result = await asyncio.to_thread(
            run_task,
            adapter.task_id,
            self.settings["k"],
            adapter=adapter,
            prop=proposer,
            selector_prop=proposer,
            runs_root=adapter.spec.runs_root,
            modes=(mode,),
            libraries={mode: library},
        )
        if self.cancel.is_set():
            raise asyncio.CancelledError()
        checks = (
            {"project_tests": self.settings["test_command"]}
            if self.settings.get("test_command")
            else {}
        )
        report = await asyncio.to_thread(
            build_report,
            adapter,
            result=result.to_dict(),
            commands=checks,
            timeout=self.settings["tool_timeout"],
            usage={
                "calls": self.budget.calls,
                "input_tokens": self.budget.input_tokens,
                "output_tokens": self.budget.output_tokens,
            },
        )
        selected = next(iter(result.selections.values()), {})
        chosen = selected.get("chosen_run_dir")
        if result.status in ("ok", "skipped") and chosen:
            row = next(
                (row for row in report["rows"] if row["label"] == report["selected_label"]), None
            )
            self.settings["candidate"] = str(Path(chosen) / "workspace")
            self.settings["candidate_source"] = str(output / "source")
            self.settings["candidate_harness"] = str(Path(chosen) / "harness")
            if row and all(check["passed"] for check in row["checks"].values()):
                self.adopt()
                summary = f"Harness2 completed and adopted the selected changes in the chat workspace. {report['verdict']}"
            else:
                summary = "Selected output failed validation; workspace kept unchanged. Inspect /diff and the report, or /adopt explicitly."
        else:
            summary = "Harness2 did not finish successfully; workspace kept unchanged. Use /retry to resume its experiment."
        trace.emit("result", summary + f" Report: {output / 'report.md'}")
        if result.status in ("ok", "skipped"):
            self.settings["pending"] = None
            self.append_exchange("/improve " + text, summary)
        self.save()
        return report

    def adopt(self):
        candidate = self.settings.get("candidate")
        if not candidate:
            raise ValueError("No improvement candidate is available")
        count = apply_changes(
            Path(self.settings["candidate_source"]),
            Path(candidate),
            self.workspace,
            self.root / "backups" / ("adopt-" + uuid4().hex[:8]),
        )
        adopted = Path(self.settings["candidate_harness"])
        temporary = self.root / ("harness-" + uuid4().hex[:8])
        shutil.copytree(adopted, temporary)
        shutil.rmtree(self.harness)
        temporary.rename(self.harness)
        self.settings["candidate"] = None
        self.save()
        CONSOLE.print(
            f"Adopted {count} changed files and the selected harness into the chat workspace.",
            style="green",
        )

    def apply(self):
        repo = Path(self.settings["repo"])
        count = apply_changes(
            self.root / "original",
            self.workspace,
            repo,
            self.root / "backups" / ("apply-" + uuid4().hex[:8]),
        )
        # Refresh the comparison baseline only after the export succeeded.
        shutil.rmtree(self.root / "original")
        snapshot(self.workspace, self.root / "original")
        CONSOLE.print(f"Applied {count} changed files to {repo}", style="green")

    def status(self):
        table = Table(title="Coding chat")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", overflow="fold")
        values = {
            "Model": self.settings["model"],
            "Workspace": self.workspace,
            "Original project": self.settings["repo"],
            "Session": self.root,
            "Model turns": self.settings["max_steps"] or "unlimited",
            "Requests": f"{self.budget.calls} / {self.budget.limit or 'unlimited'}",
            "Output budget": self.settings["max_tokens"] or "provider default (no client cap)",
            "Thinking display": self.settings["thinking"],
            "Reasoning effort": self.settings.get("reasoning_effort") or "provider default",
            "Provider routing": self.settings.get("provider") or "auto (OpenRouter selects)",
            "Pending": self.settings.get("pending") or "none",
            "Known tokens": f"{self.budget.input_tokens} input / {self.budget.output_tokens} output",
        }
        for key, value in values.items():
            table.add_row(key, str(value))
        CONSOLE.print(table)

    async def handle(self, text):
        text = text.strip()
        if not text:
            return True
        if not text.startswith("/"):
            await self.chat(text)
            return True
        command, _, argument = text.partition(" ")
        argument = strip_outer_quotes(argument)
        if command in ("/quit", "/exit"):
            return False
        if command == "/help":
            CONSOLE.print(
                Panel(
                    "/improve <request>  Run Harness2 and adopt successful output in this workspace\n"
                    "/retry              Resume interrupted chat or improvement\n/model [id]         Show or change model\n/provider [name|auto] Pin a provider, disable fallback, or use automatic routing\n/thinking on|off    Toggle provider-exposed thinking text\n/effort <level>     Set reasoning effort: default, low, medium, high\n/tokens <n|0>       Set response token budget (0: provider default)\n/test [command]     Set and run validation\n/diff               Show changes against the original project\n/adopt              Adopt the last selected candidate even if validation failed\n/apply              Copy workspace changes to the original project (conflicts abort)\n/status             Show paths, model, limits and known usage\n/clear              Start a fresh conversation; retain all code\n/quit               Save and exit\n\nEnter sends; Alt+Enter inserts a newline. Ctrl+C interrupts work; Ctrl+D exits.",
                    title="Commands",
                )
            )
        elif command == "/status":
            self.status()
        elif command == "/model":
            if argument and argument != self.settings["model"]:
                if self.conversation.exists():
                    state = json.loads(self.conversation.read_text())
                    for message in state["messages"]:
                        extra = message.get("data", {}).get("additional_kwargs", {})
                        for key in ("reasoning", "reasoning_details", "_openrouter_deltas"):
                            extra.pop(key, None)
                    private_json(self.conversation, state)
                self.settings["model"] = argument
            CONSOLE.print("Model: " + self.settings["model"], style="cyan")
        elif command == "/thinking":
            if argument not in ("on", "off"):
                raise ValueError("Use /thinking on or /thinking off")
            self.settings["thinking"] = argument == "on"
        elif command == "/provider":
            if argument:
                self.settings["provider"] = argument
            CONSOLE.print(
                "Provider routing: " + (self.settings.get("provider") or "auto"), style="cyan"
            )
        elif command == "/effort":
            if argument not in ("default", "low", "medium", "high"):
                raise ValueError("Use /effort default, low, medium, or high")
            self.settings["reasoning_effort"] = argument
            CONSOLE.print(
                f"Reasoning effort: {argument}. This changes reasoning work, not streaming delivery.",
                style="cyan",
            )
        elif command == "/tokens":
            value = int(argument)
            if value < 0:
                raise ValueError("Token budget must be zero or positive")
            self.settings["max_tokens"] = value
        elif command == "/clear":
            if self.conversation.exists():
                self.conversation.rename(self.root / ("conversation-" + uuid4().hex[:8] + ".json"))
            self.settings["pending"] = None
            CONSOLE.print("Conversation cleared; workspace files retained.", style="cyan")
        elif command == "/improve":
            if not argument:
                raise ValueError("Use /improve followed by a coding request")
            await self.improve(argument)
        elif command == "/retry":
            pending = self.settings.get("pending")
            if pending == "chat":
                await self.chat(self.settings["last_request"], retry=True)
            elif pending == "improve":
                await self.improve(self.settings["last_request"], retry=True)
            else:
                CONSOLE.print("No interrupted request to retry.", style="cyan")
        elif command == "/diff":
            before = (
                Path(self.settings["candidate_source"])
                if self.settings.get("candidate")
                else self.root / "original"
            )
            after = (
                Path(self.settings["candidate"])
                if self.settings.get("candidate")
                else self.workspace
            )
            CONSOLE.print(Text(patch_between(before, after) or "No text changes.", style="magenta"))
            CONSOLE.print(
                "Changed files: " + ", ".join(name for name, _, _ in changes(before, after))
            )
        elif command == "/adopt":
            self.adopt()
        elif command == "/apply":
            self.apply()
        elif command == "/test":
            if argument:
                self.settings["test_command"] = argument
            test = self.settings.get("test_command")
            if not test:
                raise ValueError("Use /test followed by the project test/build command")
            trace = self.trace("test")
            self.current_trace = trace
            try:
                result = await async_command(
                    test,
                    self.workspace,
                    self.settings["tool_timeout"],
                    output_callback=lambda output: trace.stream("command", output),
                )
            finally:
                trace.end_stream()
            trace.emit(
                "quality", f"Validation exit={result['exit_code']}; timeout={result['timed_out']}"
            )
        else:
            raise ValueError("Unknown command. Use /help.")
        self.save()
        return True


async def terminal(session):
    bindings = KeyBindings()

    @bindings.add("enter")
    def send(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def newline(event):
        event.current_buffer.insert_text("\n")

    @bindings.add("c-j")
    def newline_ctrl(event):
        event.current_buffer.insert_text("\n")

    history = session.root / "history"
    history.touch(mode=0o600, exist_ok=True)
    prompt = PromptSession(
        history=FileHistory(str(history)),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=bindings,
        multiline=True,
    )
    CONSOLE.print(
        Panel(
            Text(
                "h2-code • streaming coding chat\n"
                f"Model: {session.settings['model']}\nWorkspace: {session.workspace}\n"
                "Type normally to chat/code. /improve <request> runs Harness2. /help lists commands.\n"
                "Original project changes only when you type /apply.\n"
                "Thinking text/summaries stream only when exposed by the provider.",
                style="cyan",
            ),
            title="Ready",
            border_style="cyan",
        )
    )
    if session.settings.get("pending"):
        CONSOLE.print(
            "An interrupted request is retained. Use /retry to continue it.", style="yellow"
        )
    loop = asyncio.get_running_loop()
    while True:
        try:
            message = await prompt.prompt_async(
                HTML("<ansicyan><b>you ❯ </b></ansicyan>"), prompt_continuation="    … "
            )
        except EOFError:
            break
        except KeyboardInterrupt:
            CONSOLE.print("Use /quit to exit; current code is saved.", style="yellow")
            continue
        session.cancel = threading.Event()
        session.active = asyncio.create_task(session.handle(message))

        def interrupt():
            session.cancel.set()
            if session.active:
                session.active.cancel()

        loop.add_signal_handler(signal.SIGINT, interrupt)
        try:
            if not await session.active:
                break
        except asyncio.CancelledError:
            CONSOLE.print(
                "\nInterrupted. Code and conversation checkpoints retained; /retry continues.",
                style="yellow",
            )
        except Exception as error:
            CONSOLE.print(Text(f"{type(error).__name__}: {error}", style="red"))
            CONSOLE.print(
                "Session remains open. Use /retry, change /model, or send a new message.",
                style="yellow",
            )
        finally:
            if session.current_trace:
                session.current_trace.end_stream()
            session.active = None
            session.save()
    CONSOLE.print(
        f'Saved session: {session.root}\nReopen: h2-code chat --session "{session.root}"',
        style="cyan",
    )
    return 0


def main_chat(args):
    import fcntl

    session = None if args.session else ChatSession.open(args)
    root = args.session.expanduser().resolve() if args.session else session.root
    with (root / "session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This chat session is already open in another terminal") from None
        try:
            session = session or ChatSession.open(args)
            return asyncio.run(terminal(session))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
