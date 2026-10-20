"""Private recovery for an interrupted model request, separate from committed tools."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict


def request_identity(bound, messages, *, include_model=True):
    model = getattr(bound, "bound", bound)
    values = messages_to_dict(messages)
    # LangGraph can assign fresh message IDs on reload; tool-call IDs must remain.
    for value in values:
        value.get("data", {}).pop("id", None)
    payload = {
        "messages": values,
        "model": getattr(model, "model_name", type(model).__name__),
        "max_tokens": getattr(model, "max_tokens", None),
        "extra_body": getattr(model, "extra_body", None),
        "tools": getattr(bound, "kwargs", {}),
    }
    if not include_model:
        for key in ("model", "max_tokens", "extra_body"):
            payload.pop(key)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class RequestRecovery:
    def __init__(self, path, identity, context_identity=None):
        self.path = Path(path) if path else None
        self.identity = identity
        self.data = {
            "identity": identity,
            "context_identity": context_identity,
            "notes": "",
            "continuation": None,
        }
        self.attempt_notes = ""
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text())
            if saved.get("identity") == identity:
                self.data = saved
            elif context_identity and saved.get("context_identity") == context_identity:
                # Text recovery is portable; opaque tokens are tied to provider/config.
                self.data["notes"] = saved.get("notes", "")

    def import_interrupted_trace(self, trace):
        """Recover visible notes from sessions created before stream checkpoints existed."""
        if not self.path or self.path.exists():
            return False
        turns = self.path.parent / "turns"
        if not turns.is_dir() or not trace.path.exists():
            return False

        def events(path):
            result = []
            for line in path.read_text().splitlines():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return result

        def last_turn(rows):
            matches = [
                re.match(r"Turn (\d+)", e.get("message", ""))
                for e in rows
                if e.get("event") == "model"
            ]
            return next((m.group(1) for m in reversed(matches) if m), None)

        current = events(trace.path)
        current_model = next((e.get("model") for e in current if e.get("model")), None)
        candidates = sorted(
            turns.glob("*/trace.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for path in candidates:
            if path == trace.path:
                continue
            rows = events(path)
            if not rows:
                continue
            previous_model = next((e.get("model") for e in rows if e.get("model")), None)
            if last_turn(rows) != last_turn(current) or previous_model != current_model:
                return False
            requests = [
                i
                for i, e in enumerate(rows)
                if e.get("event") == "model"
                and e.get("message", "").startswith("Request ")
                and "attempt" in e.get("message", "")
            ]
            if not requests:
                continue
            tail = rows[requests[-1] + 1 :]
            if any(
                e.get("message", "").startswith(("Response in ", "Stream completed:")) for e in tail
            ):
                return False
            notes = "".join(
                e.get("content", "")
                for e in tail
                if e.get("event") in ("thinking_delta", "assistant_delta")
            )
            if notes:
                self.data["notes"] = notes
                self.save()
                return True
            return False
        return False

    @property
    def mode(self):
        if self.data.get("continuation"):
            return "provider checkpoint"
        if self.data.get("notes"):
            return "partial output as context"
        return "last committed tool result"

    def messages(self, original):
        token = self.data.get("continuation")
        if token:
            # Gemini requires the original conversation unchanged, followed by
            # this assistant checkpoint. Do not add a user message here.
            return [
                *original,
                AIMessage(content="", additional_kwargs={"reasoning_details": [token]}),
            ]
        notes = self.data.get("notes")
        if notes:
            return [
                *original,
                HumanMessage(
                    content=(
                        "Recovery context from an interrupted model response follows. All completed tool "
                        "calls and their results remain in the conversation above and the current workspace. "
                        "Continue from that verified state. The partial response below is tentative planning "
                        "and output, not evidence of execution. No unfinished tool call was executed. "
                        "Use these notes to avoid repeating analysis, and inspect files if execution is uncertain.\n\n"
                        + notes
                    )
                ),
            ]
        return original

    def begin_attempt(self):
        self.attempt_notes = ""

    def record(self, text, thinking, deltas):
        changed = False
        if thinking or text:
            if thinking:
                self.attempt_notes += thinking
            if text:
                self.attempt_notes += text
            self.data["notes"] = self.attempt_notes
            changed = True
        for delta in deltas:
            for item in delta.get("reasoning_details", []):
                if (
                    item.get("type") == "reasoning.encrypted"
                    and item.get("id") == "gemini-continuation"
                    and item.get("format")
                    and item.get("data")
                ):
                    # Preserve a complete provider checkpoint frame byte for byte.
                    # A data-only continuation fragment is not a new checkpoint.
                    self.data["continuation"] = copy.deepcopy(item)
                    changed = True
        if changed:
            self.save()

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as file:
            json.dump(self.data, file)
        os.replace(temporary, self.path)

    def clear(self):
        if self.path:
            self.path.unlink(missing_ok=True)
