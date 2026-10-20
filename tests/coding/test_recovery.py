import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    messages_to_dict,
)

from harness2.coding.agent import Budget, make_model, run_agent
from harness2.coding.recovery import RequestRecovery
from harness2.coding.streaming import model_response
from harness2.coding.trace import Trace


class RecoveryTests(unittest.TestCase):
    def test_retry_resumes_latest_opaque_checkpoint_without_changing_prior_messages(self):
        token = {
            "type": "reasoning.encrypted",
            "id": "gemini-continuation",
            "format": "google-gemini-v1",
            "data": "opaque-checkpoint",
            "index": 0,
        }
        original = [HumanMessage(content="repair the OS")]

        class Failure(Exception):
            status_code = 504

        class Bound:
            model_name = "google/gemini-4-argon"
            calls = 0

            async def astream(self, messages):
                self.calls += 1
                if self.calls == 1:
                    yield AIMessageChunk(
                        content="",
                        additional_kwargs={
                            "_openrouter_deltas": [
                                {
                                    "reasoning_details": [token],
                                    "reasoning": "Investigated the cursor.",
                                }
                            ]
                        },
                    )
                    yield AIMessageChunk(content="uncheckpointed later text")
                    raise Failure()
                self.assert_resume(messages)
                yield AIMessageChunk(content="Completed")

            def assert_resume(self, messages):
                assert messages[:-1] == original
                assert messages[-1].content == ""
                assert messages[-1].additional_kwargs["reasoning_details"] == [token]

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = Trace(root / "trace.jsonl", "resume")
            with patch("harness2.coding.streaming.asyncio.sleep", return_value=None):
                answer = asyncio.run(
                    model_response(
                        Bound(),
                        original,
                        budget=Budget(),
                        trace=trace,
                        stream=True,
                        retries=1,
                        recovery_file=root / "pending.json",
                    )
                )
            self.assertEqual(answer.content, "Completed")
            self.assertFalse((root / "pending.json").exists())
            self.assertNotIn(token["data"], trace.path.read_text())

    def test_ctrl_c_progress_survives_reopen_without_turning_partial_tool_into_action(self):
        class Bound:
            model_name = "test/model"

            async def astream(self, messages):
                yield AIMessageChunk(
                    content="",
                    additional_kwargs={
                        "_openrouter_deltas": [
                            {"reasoning": "Found dirty-region redraw is needed."}
                        ]
                    },
                )
                yield AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "write_file",
                            "args": '{"path":"kernel.c",',
                            "id": "pending",
                            "index": 0,
                        }
                    ],
                )
                raise asyncio.CancelledError()

        class Reopened(Bound):
            async def astream(self, messages):
                assert "Found dirty-region redraw is needed." in messages[-1].content
                assert "not evidence of execution" in messages[-1].content
                assert all(not getattr(m, "tool_calls", []) for m in messages)
                yield AIMessageChunk(content="Resumed")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pending = root / "pending.json"
            args = dict(
                messages=[HumanMessage(content="repair")],
                budget=Budget(),
                trace=Trace(root / "trace.jsonl", "reopen"),
                stream=True,
                recovery_file=pending,
            )
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(model_response(Bound(), **args))
            self.assertEqual(pending.stat().st_mode & 0o777, 0o600)
            self.assertIn("Found dirty-region", json.loads(pending.read_text())["notes"])
            answer = asyncio.run(model_response(Reopened(), **args))
            self.assertEqual(answer.content, "Resumed")
            self.assertFalse(pending.exists())

    def test_provider_change_keeps_text_context_but_drops_provider_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pending.json"
            first = RequestRecovery(path, "provider-a", "same-conversation")
            first.record(
                "Found redraw problem.",
                "",
                [
                    {
                        "reasoning_details": [
                            {
                                "type": "reasoning.encrypted",
                                "id": "gemini-continuation",
                                "format": "google-gemini-v1",
                                "data": "provider-a-token",
                            }
                        ]
                    }
                ],
            )
            changed = RequestRecovery(path, "provider-b", "same-conversation")
            self.assertEqual(changed.mode, "partial output as context")
            self.assertIsNone(changed.data["continuation"])
            self.assertIn("Found redraw problem.", changed.messages([])[-1].content)
            different = RequestRecovery(path, "unrelated", "different-conversation")
            self.assertEqual(different.mode, "last committed tool result")

    def test_legacy_interrupted_trace_imports_notes_for_same_model_turn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "turns" / "chat-0001" / "trace.jsonl"
            old.parent.mkdir(parents=True)
            old.write_text(
                "\n".join(
                    json.dumps(e)
                    for e in [
                        {"event": "stage", "model": "google/gemini-4-argon"},
                        {"event": "model", "message": "Turn 23"},
                        {"event": "model", "message": "Request 5/unlimited; attempt 2"},
                        {"event": "thinking_delta", "content": "Cursor redraw plan retained."},
                    ]
                )
                + "\n"
            )
            trace = Trace(root / "turns" / "chat-0002" / "trace.jsonl", "new")
            trace.emit("stage", "Starting", model="google/gemini-4-argon")
            trace.emit("model", "Turn 23")
            recovery = RequestRecovery(root / "conversation.stream.json", "identity")
            self.assertTrue(recovery.import_interrupted_trace(trace))
            self.assertIn("Cursor redraw plan", recovery.data["notes"])

    def test_partially_completed_tool_batch_does_not_repeat_completed_call(self):
        class Model:
            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                results = [m for m in messages if isinstance(m, ToolMessage)]
                assert {m.tool_call_id for m in results} == {"first", "second"}
                return AIMessage(content="Completed")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "first.txt").write_text("already completed")
            state = root / "state.json"
            state.write_text(
                json.dumps(
                    {
                        "steps": 1,
                        "completed": False,
                        "messages": messages_to_dict(
                            [
                                HumanMessage(content="write both files"),
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "id": "first",
                                            "name": "write_file",
                                            "args": {
                                                "path": "first.txt",
                                                "content": "must not repeat",
                                            },
                                        },
                                        {
                                            "id": "second",
                                            "name": "write_file",
                                            "args": {
                                                "path": "second.txt",
                                                "content": "remaining work",
                                            },
                                        },
                                    ],
                                ),
                                ToolMessage(content="First file written", tool_call_id="first"),
                            ]
                        ),
                    }
                )
            )
            answer = asyncio.run(
                run_agent(
                    workspace=root,
                    harness=root,
                    task="continue",
                    model=Model(),
                    budget=Budget(),
                    trace=Trace(root / "trace.jsonl", "batch"),
                    state_file=state,
                )
            )
            self.assertEqual(answer, "Completed")
            self.assertEqual((root / "first.txt").read_text(), "already completed")
            self.assertEqual((root / "second.txt").read_text(), "remaining work")

    def test_interrupted_command_is_checkpointed_and_not_replayed_on_resume(self):
        class Model:
            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                if any(isinstance(m, ToolMessage) for m in messages):
                    results = [m for m in messages if isinstance(m, ToolMessage)]
                    assert len(results) == 2
                    assert all(m.status == "error" for m in results)
                    assert "outcome is unknown" in results[0].content
                    return AIMessage(content="Replanned after interruption")
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "blocked",
                            "name": "run_command",
                            "args": {
                                "command_text": "python3 -c 'import time; print(12345); time.sleep(30)'"
                            },
                        },
                        {
                            "id": "later",
                            "name": "write_file",
                            "args": {"path": "must-not-run.txt", "content": "dependent action"},
                        },
                    ],
                )

        async def exercise(root):
            trace = Trace(root / "trace.jsonl", "interrupt")
            ready = asyncio.Event()
            original_stream = trace.stream

            def output(channel, text):
                original_stream(channel, text)
                if channel == "command" and "12345" in text:
                    ready.set()

            trace.stream = output
            args = dict(
                workspace=root,
                harness=root,
                task="test interruption",
                model=Model(),
                budget=Budget(),
                trace=trace,
                state_file=root / "state.json",
                stream=True,
            )
            worker = asyncio.create_task(run_agent(**args))
            try:
                await asyncio.wait_for(ready.wait(), timeout=5)
            finally:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker
            saved = json.loads((root / "state.json").read_text())
            self.assertEqual([m["type"] for m in saved["messages"][-2:]], ["tool", "tool"])
            answer = await run_agent(**args)
            self.assertEqual(answer, "Replanned after interruption")
            self.assertFalse((root / "must-not-run.txt").exists())

        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(exercise(Path(temp)))

    def test_explicit_provider_pins_route_and_disables_fallback(self):
        model = make_model("google/gemini-4-argon", "test-key", 0, provider="Google AI Studio")
        try:
            self.assertEqual(
                model.extra_body["provider"],
                {"only": ["Google AI Studio"], "allow_fallbacks": False},
            )
        finally:
            model.http_client.close()
            asyncio.run(model.http_async_client.aclose())
