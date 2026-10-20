from __future__ import annotations

import asyncio
import difflib
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from .files import async_command, command, excluded, iter_files, safe_path
from .streaming import model_response
from .trace import Trace
from .transport import observe_response


class OpenRouterChatModel(ChatOpenAI):
    """Preserve OpenRouter's opaque continuation fields across tool turns.

    The pinned ChatOpenAI serializer drops these provider extensions. They must
    round-trip for models such as MiniMax; they are never printed in CLI traces.
    """

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, raw.get("choices", [])):
            for name in ("reasoning_details", "reasoning"):
                if choice.get("message", {}).get(name) is not None:
                    generation.message.additional_kwargs[name] = choice["message"][name]
        return result

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        converted = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or []
        if converted and chunk.get("provider"):
            converted.message.response_metadata["openrouter_provider"] = chunk["provider"]
        if converted and choices and choices[0].get("delta"):
            delta = choices[0]["delta"]
            extensions = {
                key: delta[key] for key in ("reasoning_details", "reasoning") if delta.get(key)
            }
            if extensions:
                # Keep individual deltas apart; generic dict merging corrupts opaque
                # signatures and repeated provider block type/format identifiers.
                converted.message.additional_kwargs["_openrouter_deltas"] = [extensions]
        return converted

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for original, serialized in zip(messages, payload.get("messages", [])):
            if isinstance(original, AIMessage):
                for name in ("reasoning_details", "reasoning"):
                    if name in original.additional_kwargs:
                        serialized[name] = original.additional_kwargs[name]
        return payload


class Budget:
    def __init__(self, limit=0):
        self.limit, self.calls, self.input_tokens, self.output_tokens = limit, 0, 0, 0
        self.lock = threading.Lock()

    def reserve(self):
        with self.lock:
            if self.limit and self.calls >= self.limit:
                raise RuntimeError(f"Model request budget exhausted ({self.limit})")
            self.calls += 1
            return self.calls

    def usage(self, metadata):
        with self.lock:
            self.input_tokens += metadata.get("input_tokens", 0)
            self.output_tokens += metadata.get("output_tokens", 0)


class AgentState(MessagesState):
    steps: int


def make_model(
    model,
    key,
    max_tokens,
    *,
    base_url="https://openrouter.ai/api/v1",
    stream_idle_timeout=0,
    reasoning_effort=None,
    provider=None,
):
    # Long reasoning pauses are valid; retain connection/write deadlines without
    # imposing a hidden response deadline when the user requested unlimited work.
    timeout = httpx.Timeout(connect=30, write=60, pool=30, read=stream_idle_timeout or None)
    reasoning = {"exclude": False}
    if reasoning_effort and reasoning_effort != "default":
        reasoning["effort"] = reasoning_effort
    extra_body = {"reasoning": reasoning}
    if provider and provider != "auto":
        extra_body["provider"] = {"only": [provider], "allow_fallbacks": False}
    return OpenRouterChatModel(
        model=model,
        api_key=key,
        base_url=base_url,
        http_client=httpx.Client(timeout=timeout),
        http_async_client=httpx.AsyncClient(
            timeout=timeout, event_hooks={"response": [observe_response]}
        ),
        max_tokens=max_tokens or None,
        stream_usage=True,
        extra_body=extra_body,
        timeout=timeout,
        stream_chunk_timeout=stream_idle_timeout or None,
        max_retries=0,
        use_responses_api=False,
    )


def local_tools(
    workspace: Path,
    harness: Path,
    trace: Trace,
    *,
    editor=False,
    writable=None,
    tool_timeout=60,
    stream=False,
):
    workspace, harness = workspace.resolve(), harness.resolve()

    def may_write(path):
        safe_path(workspace, path)
        if writable is not None and not writable(path):
            raise ValueError("This path is not an editable harness component")

    @tool
    def list_files(path: str = ".") -> str:
        """List files under a workspace directory, excluding credentials and build caches."""
        start = safe_path(workspace, path)
        if not start.is_dir():
            raise ValueError("Expected a directory")
        paths = []
        for file in iter_files(start):
            relative = file.relative_to(workspace)
            if not file.is_file() or file.is_symlink() or excluded(relative):
                continue
            paths.append(str(relative))
            if len(paths) == 1000:
                paths.append("… listing capped at 1000 files; narrow the path")
                break
        return "\n".join(paths)

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 250) -> str:
        """Read a UTF-8 file using one-based line numbers (at most 500 lines per call)."""
        if start_line < 1 or end_line < start_line or end_line - start_line > 499:
            raise ValueError("Use a valid range of at most 500 lines")
        file = safe_path(workspace, path)
        if file.stat().st_size > 2_000_000:
            raise ValueError("File exceeds the 2 MB read limit")
        lines = file.read_text().splitlines()
        return "\n".join(
            f"{i + 1}: {lines[i]}" for i in range(start_line - 1, min(end_line, len(lines)))
        )[:24000]

    @tool
    def search_files(query: str, path: str = ".") -> str:
        """Find literal text in workspace text files, returning paths and line numbers."""
        start = safe_path(workspace, path)
        found = []
        for file in iter_files(start):
            if not file.is_file() or file.is_symlink() or excluded(file.relative_to(workspace)):
                continue
            if file.stat().st_size > 1_000_000:
                continue
            try:
                for number, line in enumerate(file.read_text().splitlines(), 1):
                    if query in line:
                        found.append(f"{file.relative_to(workspace)}:{number}: {line[:300]}")
                        if len(found) >= 80:
                            return "\n".join(found) + "\n… search capped at 80 matches"
            except UnicodeError:
                continue
        return "\n".join(found) or "No matches"

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or replace a UTF-8 workspace file. Prefer replace_text for a small edit."""
        may_write(path)
        file = safe_path(workspace, path)
        before = file.read_text() if file.exists() else ""
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(True), content.splitlines(True), fromfile=path, tofile=path
            )
        )
        trace.emit("diff", f"Changed {path}", diff=diff)
        return f"Wrote {len(content)} characters to {path}"

    @tool
    def replace_text(path: str, old: str, new: str) -> str:
        """Replace exactly one occurrence of old text in a workspace file."""
        may_write(path)
        file = safe_path(workspace, path)
        content = file.read_text()
        if not old or content.count(old) != 1:
            raise ValueError("old must match exactly once; read the file again")
        return write_file.invoke({"path": path, "content": content.replace(old, new, 1)})

    @tool
    def load_skill(name: str) -> str:
        """Load an available procedural skill by name from the active harness."""
        if not name or "/" in name or "\\" in name or ".." in name:
            raise ValueError("Use a skill name, not a path")
        return safe_path(harness, f"skills/{name}/SKILL.md").read_text()

    tools = [list_files, read_file, search_files, write_file, replace_text]
    if not editor:

        @tool
        async def run_command(command_text: str) -> str:
            """Run a shell command in the workspace (for tests/builds). This is not an OS sandbox."""
            if stream:
                try:
                    result = await async_command(
                        command_text,
                        workspace,
                        tool_timeout,
                        output_callback=lambda text: trace.stream("command", text),
                    )
                finally:
                    trace.end_stream()
            else:
                result = await asyncio.to_thread(command, command_text, workspace, tool_timeout)
            return json.dumps(result)

        tools += [run_command, load_skill]
    return tools


async def _run_agent(
    *,
    workspace: Path,
    harness: Path,
    task: str,
    model,
    budget: Budget,
    trace: Trace,
    max_steps=0,
    tool_timeout=60,
    editor=False,
    writable=None,
    mcp_config=None,
    state_file=None,
    new_turn=False,
    interactive=False,
    stream=False,
    retries=3,
):
    saved = json.loads(state_file.read_text()) if state_file and state_file.exists() else None
    initial = (
        {"messages": messages_from_dict(saved["messages"]), "steps": saved["steps"]}
        if saved
        else {"messages": [HumanMessage(content=task)], "steps": 0}
    )

    def pending_tools(messages):
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AIMessage):
                completed = {
                    m.tool_call_id for m in messages[index + 1 :] if isinstance(m, ToolMessage)
                }
                return [call for call in message.tool_calls if call["id"] not in completed]
        return []

    if saved and new_turn:
        initial["messages"] += [
            ToolMessage(
                content="Interrupted before a result was committed. Inspect the workspace before repeating this action.",
                tool_call_id=call["id"],
            )
            for call in pending_tools(initial["messages"])
        ]
        initial["messages"].append(HumanMessage(content=task))
        initial["steps"] = 0
        saved["completed"] = False

    def continue_after_cutoff(response):
        # Truncated tool arguments are not safe to execute. Ask for a complete
        # continuation, preserving the provider's opaque reasoning fields.
        response.tool_calls = []
        response.invalid_tool_calls = []
        response.additional_kwargs.pop("tool_calls", None)
        return HumanMessage(
            content="Your previous response reached its output-token limit. "
            "Continue the original task from the current workspace. Any tool calls in the "
            "truncated response were not executed; issue complete structured calls if needed. "
            "Do not repeat file reads or analysis unnecessarily. Provide a final summary only after the work is complete."
        )

    if (
        saved
        and isinstance(initial["messages"][-1], AIMessage)
        and initial["messages"][-1].response_metadata.get("finish_reason") == "length"
    ):
        initial["messages"].append(continue_after_cutoff(initial["messages"][-1]))
        saved["completed"] = False
        trace.emit("stage", "Continuing a checkpoint whose last response was truncated")
    if saved and saved.get("completed"):
        content = initial["messages"][-1].content
        return content if isinstance(content, str) else json.dumps(content)

    def checkpoint(state, additions, *, completed=False):
        if not state_file:
            return
        payload = {
            "messages": messages_to_dict(state["messages"] + additions),
            "steps": state.get("steps", 0),
            "completed": completed,
        }
        temporary = state_file.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as file:
            json.dump(payload, file)
        os.replace(temporary, state_file)

    checkpoint(initial, [])
    if saved:
        trace.emit("stage", f"Continuing saved conversation after {initial['steps']} model turns")
    tools = local_tools(
        workspace,
        harness,
        trace,
        editor=editor,
        writable=writable,
        tool_timeout=tool_timeout,
        stream=stream,
    )
    if mcp_config and not editor:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(mcp_config)
        external = await client.get_tools()
        existing = {t.name for t in tools}
        if any(t.name in existing for t in external):
            raise ValueError("An MCP tool name conflicts with a built-in tool")
        tools += external
        trace.emit("stage", f"Connected {len(external)} MCP tools")
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    if editor:
        instructions = (
            "You improve the configuration of a LangGraph coding agent. Read the supplied context files "
            "using filesystem tools. In the mission, context/ refers to your workspace root. "
            "For edit missions, modify harness/ in place. Only four always-on Markdown files "
            "(systemprompt.md, guardrails.md, LongTermMEMORY.md, tool_descriptions/tool_guidance.md) "
            "and skills/<name>/SKILL.md are consumed. Do not create scripts, plugins or subagents. "
            "The executor is LangGraph, not OpenCode; any OpenCode naming in generic instructions "
            "describes the inherited harness file format. You cannot run code. "
            "Finish with the exact notes/choice/experience response format requested by the mission. "
            "Do not claim measured gains without observed evidence. Do not store task-specific solutions "
            "as general instructions. Final response must be text, not JSON."
        )
    else:
        parts = []
        for name in (
            "systemprompt.md",
            "guardrails.md",
            "LongTermMEMORY.md",
            "tool_descriptions/tool_guidance.md",
        ):
            file = harness / name
            if file.exists():
                parts.append(f"# {name}\n{file.read_text()}")
        skills = sorted(p.parent.name for p in (harness / "skills").glob("*/SKILL.md"))
        if skills:
            parts.append("Available skills (use load_skill): " + ", ".join(skills))
        if interactive:
            parts.append(
                "You are an interactive coding and chat assistant. Answer ordinary questions "
                "conversationally without unnecessary tool calls. Inspect or edit code when the user asks "
                "for coding work. Do not launch a new project or modify files in response to casual chat. "
                "The conversation persists across messages. You can use files, commands, skills and configured MCP tools."
            )
        parts.append(
            f"Current workspace: {workspace.resolve()}. Use relative paths within this workspace. "
            "Absolute paths in older conversation/tool results are historical; do not edit previous run folders or the original project."
        )
        parts.append(
            "You operate inside a copied project workspace. Use file and command tools to "
            "complete the coding task. Changes must be written to files; a proposed patch in "
            "chat alone does not complete the task. Summarize actual changes and validation. "
            "Never claim a command ran if it did not. Do not rewrite existing tests to hide failures. "
            "Make focused incremental edits with replace_text when possible, rather than regenerating "
            "large existing files in a single tool call. Complete and verify one coherent change at a time "
            "so successful work is checkpointed before moving on. "
            "For long-lived programs such as QEMU, never block on pipe.read(), read1(), or readline() "
            "while the child is running. Use bounded communicate(timeout=...), or capture stderr to "
            "a file and inspect it without waiting for EOF. Always terminate children in finally. "
            "A tool timeout means inspect and fix the command before retrying it unchanged."
        )
        instructions = "\n\n".join(parts)
    trace.emit(
        "stage",
        f"LangGraph started; {str(max_steps) + ' model turns maximum' if max_steps else 'no model-turn limit'}",
        model=getattr(model, "model_name", "test-model"),
    )

    async def llm(state):
        if max_steps and state.get("steps", 0) >= max_steps:
            raise RuntimeError(f"Agent exhausted {max_steps} turns before finishing")
        step = state.get("steps", 0) + 1
        turn_label = f"{step}/{max_steps}" if max_steps else str(step)
        trace.emit("model", f"Turn {turn_label}")
        start = time.monotonic()
        response = await model_response(
            bound,
            [SystemMessage(content=instructions)] + state["messages"],
            budget=budget,
            trace=trace,
            stream=stream,
            retries=retries,
            recovery_file=state_file.with_suffix(".stream.json") if state_file else None,
            recover_legacy=bool(saved and not new_turn),
        )
        usage = response.usage_metadata or {}
        budget.usage(usage)
        trace.emit(
            "model",
            f"Response in {time.monotonic() - start:.1f}s",
            usage=usage,
            tool_calls=len(response.tool_calls),
            finish_reason=response.response_metadata.get("finish_reason"),
        )
        if response.content:
            trace.emit(
                "assistant",
                "Assistant response",
                content=response.content,
                streamed=stream and hasattr(bound, "astream"),
            )
        truncated = response.response_metadata.get("finish_reason") == "length"
        additions = [response]
        if truncated:
            trace.emit("stage", "Response reached its output-token limit; requesting continuation")
            additions.append(continue_after_cutoff(response))
        checkpoint(
            {**state, "steps": step},
            additions,
            completed=not truncated and bool(response.content) and not response.tool_calls,
        )
        return {"messages": additions, "steps": step}

    async def dispatch(state):
        messages = []
        for call in pending_tools(state["messages"]):
            name, arguments = call["name"], call["args"]
            trace.emit("tool", name, args=arguments)
            start = time.monotonic()
            try:
                selected = by_name[name]
                result = await asyncio.wait_for(
                    selected.ainvoke(arguments), timeout=tool_timeout + 5
                )
                output = result if isinstance(result, str) else json.dumps(result, default=str)
                trace.emit(
                    "result", f"{name} finished in {time.monotonic() - start:.2f}s", output=output
                )
            except asyncio.CancelledError:
                output = (
                    f"Tool {name} was interrupted by the user. Its outcome is unknown and it may "
                    "have partially changed files or external state. Inspect the workspace before "
                    "deciding what to do next. Do not blindly repeat the interrupted command; "
                    "diagnose any blocking operation and use bounded subprocess cleanup. "
                    "Any remaining calls in this batch were not executed."
                )
                for abandoned in pending_tools(state["messages"] + messages):
                    messages.append(
                        ToolMessage(content=output, tool_call_id=abandoned["id"], status="error")
                    )
                checkpoint(state, messages)
                trace.emit(
                    "error",
                    f"Tool {name} interrupted; saved for model recovery instead of automatic replay",
                )
                raise
            except Exception as exc:
                output = f"{type(exc).__name__}: {exc}"
                trace.emit("error", f"Tool {name} failed", output=output)
            messages.append(ToolMessage(content=output[:24000], tool_call_id=call["id"]))
            # Commit each result before starting another tool in the batch.
            checkpoint(state, messages)
        return {"messages": messages}

    graph = StateGraph(AgentState)
    graph.add_node("model", llm)
    graph.add_node("tools", dispatch)
    graph.add_conditional_edges(
        START,
        lambda state: "tools" if pending_tools(state["messages"]) else "model",
    )
    graph.add_conditional_edges(
        "model",
        lambda state: (
            "model"
            if isinstance(state["messages"][-1], HumanMessage)
            else "tools"
            if state["messages"][-1].tool_calls
            else END
        ),
    )
    graph.add_edge("tools", "model")
    compiled = graph.compile()
    state = await compiled.ainvoke(
        initial,
        config={"recursion_limit": max_steps * 2 + 2 if max_steps else sys.maxsize},
    )
    content = state["messages"][-1].content
    answer = content if isinstance(content, str) else json.dumps(content)
    if not answer.strip():
        raise RuntimeError(
            "Agent ended without a final response; inspect finish_reason and token usage in trace.jsonl"
        )
    return answer


async def run_agent(*, model, cancel_event=None, **kwargs):
    # Harness2 invokes synchronous adapters from multiple threads. Each rollout owns
    # an event loop, so it must also own its async HTTP client/connection pool.
    owned = callable(model) and not hasattr(model, "bind_tools")
    active = model() if owned else model
    worker = asyncio.create_task(_run_agent(model=active, **kwargs))

    async def watch_cancel():
        while not worker.done():
            if cancel_event and cancel_event.is_set():
                worker.cancel()
                return
            await asyncio.sleep(0.1)

    watcher = asyncio.create_task(watch_cancel()) if cancel_event else None
    try:
        return await worker
    finally:
        if watcher:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if owned:
            client = getattr(active, "root_async_client", None)
            if client is not None:
                await client.close()
            client = getattr(active, "root_client", None)
            if client is not None:
                client.close()
