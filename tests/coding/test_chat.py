import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from harness2.coding.agent import Budget
from harness2.coding.chat import ChatSession, apply_changes
from harness2.coding.cli import parser_for_cli
from harness2.coding.files import async_command
from harness2.coding.streaming import model_response, restore_reasoning
from harness2.coding.trace import ReliableWriter, Trace


class StreamModel:
    def __init__(self):
        self.inputs = []

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.inputs.append(messages)
        user = next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        yield AIMessageChunk(content="Hello ")
        yield AIMessageChunk(
            content=user,
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


class EngineModel:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        system = messages[0].content
        user = next(m.content for m in messages if isinstance(m, HumanMessage))
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if "You improve the configuration" in system:
            if "# WHAT HAPPENED" in user:
                return AIMessage(content="NO_ENTRY")
            if not results:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "h",
                            "name": "write_file",
                            "args": {
                                "path": "harness/tool_descriptions/tool_guidance.md",
                                "content": "Run regression tests after editing.\n",
                            },
                        }
                    ],
                )
            return AIMessage(content="<notes>Verify behavior using tests.</notes>")
        if not results:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c",
                        "name": "write_file",
                        "args": {"path": "calc.py", "content": "def add(a,b):\n    return a+b\n"},
                    }
                ],
            )
        return AIMessage(content="Fixed addition.")


class ChatTests(unittest.TestCase):
    def session(self, root):
        repo = root / "repo"
        repo.mkdir()
        (repo / "calc.py").write_text("def add(a,b): return a-b\n")
        args = parser_for_cli().parse_args(
            ["chat", "--repo", str(repo), "--output", str(root / "chat"), "--model", "test/model"]
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            return ChatSession.open(args)

    def test_reasoning_effort_is_saved_and_used_by_model(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.session(Path(temp))
            asyncio.run(session.handle("/effort low"))
            self.assertEqual(
                json.loads((session.root / "session.json").read_text())["reasoning_effort"], "low"
            )
            with patch("harness2.coding.chat.make_model") as factory:
                session.model()
            self.assertEqual(factory.call_args.kwargs["reasoning_effort"], "low")
            with self.assertRaises(ValueError):
                asyncio.run(session.handle("/effort invalid"))

    def test_normal_chat_persists_history_without_harness2(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.session(Path(temp))
            model = StreamModel()
            with (
                patch.object(session, "model", new=lambda: model),
                patch("harness2.coding.chat.run_task") as engine,
            ):
                asyncio.run(session.handle("hello"))
                asyncio.run(session.handle("remember that"))
            engine.assert_not_called()
            self.assertEqual(len(model.inputs), 2)
            self.assertTrue(
                any(isinstance(m, HumanMessage) and m.content == "hello" for m in model.inputs[1])
            )
            self.assertEqual(session.budget.calls, 2)
            self.assertTrue(json.loads(session.conversation.read_text())["completed"])
            self.assertFalse((session.root / "improvements").exists())
            self.assertEqual(session.conversation.stat().st_mode & 0o777, 0o600)

    def test_improve_invokes_real_engine_and_keeps_original_project(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.session(Path(temp))
            with patch.object(session, "model", new=lambda: EngineModel()):
                asyncio.run(session.handle('/improve "Fix addition"'))
            self.assertIn("a+b", (session.workspace / "calc.py").read_text())
            self.assertIn("a-b", (Path(session.settings["repo"]) / "calc.py").read_text())
            self.assertTrue(list((session.root / "improvements").rglob("report.json")))
            self.assertEqual(session.settings["improvements"], 1)
            self.assertIsNone(session.settings["pending"])
            asyncio.run(session.handle("/apply"))
            self.assertIn("a+b", (Path(session.settings["repo"]) / "calc.py").read_text())

    def test_apply_checks_conflicts_before_writing_and_copies_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("before", "after", "target"):
                (root / name).mkdir()
            for name in ("before", "target"):
                (root / name / "code.py").write_text("old")
            (root / "after/code.py").write_text("new")
            (root / "after/boot.bin").write_bytes(b"\0\xff\0")
            (root / "target/code.py").write_text("external change")
            with self.assertRaisesRegex(ValueError, "Conflict"):
                apply_changes(root / "before", root / "after", root / "target", root / "backup")
            self.assertFalse((root / "target/boot.bin").exists())
            (root / "target/code.py").write_text("old")
            self.assertEqual(
                apply_changes(root / "before", root / "after", root / "target", root / "backup"), 2
            )
            self.assertEqual((root / "target/boot.bin").read_bytes(), b"\0\xff\0")

    def test_import_from_failed_run_keeps_code_and_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            origin = root / "old"
            source = origin / "source"
            source.mkdir(parents=True)
            (source / "code.py").write_text("old")
            workspace = origin / "executions/ns/task/base1/workspace"
            workspace.mkdir(parents=True)
            (workspace / "code.py").write_text("important edits")
            (workspace.parent / "harness").mkdir()
            (workspace.parent / "harness/systemprompt.md").write_text("Be helpful")
            (origin / "config.json").write_text(
                json.dumps({"repo": str(source), "model": "google/gemini-4-argon", "task": "code"})
            )
            args = parser_for_cli().parse_args(
                ["chat", "--from-run", str(origin), "--output", str(root / "chat")]
            )
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
                session = ChatSession.open(args)
            self.assertEqual(session.settings["model"], "google/gemini-4-argon")
            self.assertEqual((session.workspace / "code.py").read_text(), "important edits")

    def test_reasoning_fragments_preserve_opaque_protocol_fields(self):
        message = AIMessage(
            content="done",
            additional_kwargs={
                "_openrouter_deltas": [
                    {
                        "reasoning_details": [
                            {
                                "index": 0,
                                "type": "reasoning.encrypted",
                                "format": "vendor-v1",
                                "data": "abc",
                            }
                        ]
                    },
                    {
                        "reasoning_details": [
                            {
                                "index": 0,
                                "type": "reasoning.encrypted",
                                "format": "vendor-v1",
                                "data": "def",
                            }
                        ]
                    },
                ]
            },
        )
        result = restore_reasoning(message)
        self.assertEqual(
            result.additional_kwargs["reasoning_details"],
            [{"index": 0, "type": "reasoning.encrypted", "format": "vendor-v1", "data": "abcdef"}],
        )

    def test_stream_redacts_secret_split_across_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Trace(
                Path(temp) / "trace.jsonl", "stream", secrets=("secret-key",), thinking=True
            )
            trace.stream("assistant", "Hi!")
            # Even a very short chunk is written immediately when it cannot be a key.
            self.assertIn("Hi!", trace.path.read_text())
            trace.stream("assistant", "before secr")
            trace.stream("assistant", "et-key after")
            trace.end_stream()
            text = trace.path.read_text()
            self.assertNotIn("secret-key", text)
            joined = "".join(json.loads(line).get("content", "") for line in text.splitlines())
            self.assertIn("[REDACTED]", joined)

    def test_retry_does_not_repeat_a_committed_tool(self):
        class Transient(Exception):
            status_code = 504

        class Bound:
            attempts = 0

            async def ainvoke(self, messages):
                self.attempts += 1
                if self.attempts == 1:
                    raise Transient()
                return AIMessage(content="ok")

        async def exercise(root):
            budget = Budget(5)
            with patch("harness2.coding.streaming.asyncio.sleep", return_value=None):
                result = await model_response(
                    Bound(), [], budget=budget, trace=Trace(root / "trace.jsonl", "retry")
                )
            self.assertEqual(result.content, "ok")
            self.assertEqual(budget.calls, 2)

        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(exercise(Path(temp)))

    def test_retry_recognizes_error_in_an_http_200_sse_stream(self):
        import httpx
        from openai import APIError

        from harness2.coding.streaming import retryable

        error = APIError(
            "The operation was aborted",
            request=httpx.Request("POST", "https://example.com"),
            body={"code": 504, "message": "The operation was aborted"},
        )
        self.assertTrue(retryable(error))

    def test_streamed_command_reports_pipeline_failure_and_cancel(self):
        async def exercise(root):
            chunks = []
            result = await async_command(
                "echo first; false | cat", root, output_callback=chunks.append
            )
            self.assertNotEqual(result["exit_code"], 0)
            self.assertIn("first", "".join(chunks))
            task = asyncio.create_task(async_command("sleep 30", root))
            await asyncio.sleep(0.1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(exercise(Path(temp)))

    def test_reliable_writer_handles_nonblocking_stdout(self):
        class Output:
            encoding = "utf-8"

            def fileno(self):
                return 123

        with (
            patch("harness2.coding.trace.os.write", side_effect=[BlockingIOError(), 2, 3]) as write,
            patch("harness2.coding.trace.select.select"),
        ):
            self.assertEqual(ReliableWriter(Output()).write("hello"), 5)
            self.assertEqual(write.call_args_list[-1].args[1], b"llo")
