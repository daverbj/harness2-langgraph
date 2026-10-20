import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from harness2.coding.adapter import CodingAdapter, CodingProposer
from harness2.coding.agent import Budget, local_tools, make_model, run_agent
from harness2.coding.cli import main
from harness2.coding.files import command, safe_path, snapshot
from harness2.coding.report import build_report, evaluate
from harness2.coding.trace import Trace
from harness2.core.library import Library
from harness2.core.schedule import run_task


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return next(self.responses)


def call(name, arguments, identifier="1"):
    return AIMessage(content="", tool_calls=[{"id": identifier, "name": name, "args": arguments}])


class WorkflowModel:
    """A model double, not a replacement for LangGraph or Harness2."""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        system = messages[0].content
        mission = next(m.content for m in messages if isinstance(m, HumanMessage))
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if "You improve the configuration" in system:
            if "# WHAT HAPPENED" in mission:
                return AIMessage(content="NO_ENTRY")
            if not results:
                return call(
                    "write_file",
                    {
                        "path": "harness/tool_descriptions/tool_guidance.md",
                        "content": "Inspect the implementation, change only source code, and verify with tests.\n",
                    },
                )
            return AIMessage(
                content="<notes>Verify the modified implementation with executable tests.</notes>"
            )
        if not results:
            return call(
                "write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}
            )
        if len(results) == 1:
            return call("run_command", {"command_text": "python -m unittest discover -q"}, "2")
        return AIMessage(content="Fixed addition. Ran unittest successfully.")


class CodingTests(unittest.TestCase):
    def test_path_limits_and_editor_permissions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root.parent / "outside-no-write"
            with self.assertRaises(ValueError):
                safe_path(root, "../escape")
            with self.assertRaises(ValueError):
                safe_path(root, ".env")
            (root / "link").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_path(root, "link/outside-no-write")
            tools = {
                t.name: t
                for t in local_tools(
                    root,
                    root,
                    Trace(root / "trace.jsonl", "test"),
                    editor=True,
                    writable=lambda name: name.startswith("harness/"),
                )
            }
            self.assertNotIn("run_command", tools)
            with self.assertRaises(ValueError):
                tools["write_file"].invoke({"path": "source.py", "content": "bad"})
            self.assertFalse(outside.exists())

    def test_snapshot_excludes_secrets_symlinks_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "repo"
            source.mkdir()
            (source / "code.py").write_text("x=1\n")
            (source / ".env").write_text("secret")
            (source / "link.py").symlink_to(source / "code.py")
            output = source / "output"
            output.mkdir()
            snapshot(source, output / "source", excluded_roots=(output,))
            self.assertEqual([p.name for p in (output / "source").iterdir()], ["code.py"])

    def test_commands_strip_credentials_and_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "do-not-pass"}):
                result = command(
                    "python -c \"import os; print(os.getenv('OPENROUTER_API_KEY', 'absent'))\"",
                    root,
                )
            self.assertIn("absent", result["output"])
            result = command('python -c "import time; time.sleep(5)"', root, timeout=1)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)

    def test_graph_runs_tools_and_records_diffs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = Trace(root / "trace.jsonl", "graph", secrets=("secret123",))
            model = SequenceModel(
                [
                    call("write_file", {"path": "code.py", "content": "x=1\n"}),
                    AIMessage(content="Done secret123"),
                ]
            )
            answer = asyncio.run(
                run_agent(
                    workspace=root,
                    harness=root,
                    task="edit",
                    model=model,
                    budget=Budget(2),
                    trace=trace,
                    max_steps=3,
                )
            )
            self.assertEqual((root / "code.py").read_text(), "x=1\n")
            trace.emit("result", answer)
            raw = trace.path.read_text()
            self.assertNotIn("secret123", raw)
            self.assertIn('"event": "diff"', raw)

    def test_budget_exhaustion_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = SequenceModel([call("list_files", {})])
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                asyncio.run(
                    run_agent(
                        workspace=root,
                        harness=root,
                        task="inspect",
                        model=model,
                        budget=Budget(1),
                        trace=Trace(root / "trace.jsonl", "test"),
                        max_steps=3,
                    )
                )

    def test_full_harness2_langgraph_integration_and_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "calc.py").write_text("def add(a, b):\n    return a - b\n")
            (source / "test_calc.py").write_text(
                "import unittest\nfrom calc import add\nclass Tests(unittest.TestCase):\n"
                "    def test_add(self): self.assertEqual(add(2,3),5)\n"
            )
            output = root / "result"
            output.mkdir()
            trace = Trace(output / "trace.jsonl", "harness2")
            budget = Budget(50)
            adapter = CodingAdapter(
                root=output,
                source=source,
                task="Fix addition.",
                model=WorkflowModel(),
                model_id="test/model",
                budget=budget,
                trace=trace,
                max_steps=5,
            )
            lib = Library(output / "library" / "seq", "coding", adapter.substrate.surface())
            prop = CodingProposer(adapter)
            result = run_task(
                adapter.task_id,
                1,
                adapter=adapter,
                prop=prop,
                selector_prop=prop,
                runs_root=adapter.spec.runs_root,
                modes=("seq",),
                libraries={"seq": lib},
            )
            self.assertEqual(result.status, "ok", result.errors)
            self.assertEqual(adapter.execution_count, 4)
            self.assertEqual((source / "calc.py").read_text(), "def add(a, b):\n    return a - b\n")
            report = build_report(
                adapter,
                result=result.to_dict(),
                commands={"project_tests": "python -m unittest discover -q"},
                timeout=10,
                usage={"calls": budget.calls, "input_tokens": 0, "output_tokens": 0},
            )
            self.assertIn("No measured improvement", report["verdict"])
            self.assertEqual(report["selected_label"], "c1")
            self.assertFalse(report["rows"][0]["checks"]["project_tests"]["passed"])
            self.assertTrue(
                all(row["checks"]["project_tests"]["passed"] for row in report["rows"][1:])
            )
            self.assertTrue(any((output / "improver").rglob("harness.patch")))
            self.assertEqual(adapter.deliverable_files(output / "missing"), {})
            baseline = next((output / "executions").glob(f"*/{adapter.task_id}/base1"))
            calls_before = budget.calls
            adapter.execute(
                adapter.task_id, None, baseline.relative_to(output / "executions").as_posix()
            )
            self.assertEqual(budget.calls, calls_before)

    def test_held_out_tests_are_copied_only_for_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            (source / "value.py").write_text("VALUE=42\n")
            external = root / "hidden"
            external.mkdir()
            (external / "test_value.py").write_text(
                "import unittest\nfrom value import VALUE\nclass Check(unittest.TestCase):\n"
                "    def test_value(self): self.assertEqual(VALUE,42)\n"
            )
            result = evaluate(source, "python -m unittest discover -s .h2-eval -v", 10, external)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["test_counts"]["passed"], 1)
            self.assertFalse((source / ".h2-eval").exists())
            empty = evaluate(source, "python -m unittest discover -q", 10)
            self.assertFalse(empty["passed"])
            self.assertEqual(empty["validation_note"], "No tests were collected")

    def test_http_clients_survive_multiple_event_loops(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps(
                    {
                        "id": "test",
                        "object": "chat.completion",
                        "created": 1,
                        "model": "test/model",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "Done"},
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                budget = Budget(6)

                def factory():
                    return make_model(
                        "test/model",
                        "test-key",
                        100,
                        base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    )

                def invoke(number):
                    return asyncio.run(
                        run_agent(
                            workspace=root,
                            harness=root,
                            task="finish",
                            model=factory,
                            budget=budget,
                            trace=Trace(root / f"{number}.jsonl", str(number)),
                        )
                    )

                self.assertEqual(invoke(0), "Done")
                with ThreadPoolExecutor(max_workers=2) as pool:
                    self.assertEqual(list(pool.map(invoke, [1, 2])), ["Done", "Done"])
                self.assertEqual(invoke(3), "Done")
                self.assertEqual(budget.calls, 4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_mcp_tool_uses_the_same_graph_and_trace(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            server = root / "server.py"
            server.write_text(
                "from mcp.server.fastmcp import FastMCP\n"
                'mcp=FastMCP("local-test")\n'
                "@mcp.tool()\n"
                "def echo(text: str) -> str:\n"
                '    """Echo text for a local integration check."""\n'
                '    return "echoed: " + text\n'
                'mcp.run(transport="stdio")\n'
            )
            trace = Trace(root / "trace.jsonl", "mcp")
            model = SequenceModel([call("echo", {"text": "hello"}), AIMessage(content="Done")])
            answer = asyncio.run(
                run_agent(
                    workspace=root,
                    harness=root,
                    task="echo hello",
                    model=model,
                    budget=Budget(3),
                    trace=trace,
                    mcp_config={
                        "local": {
                            "transport": "stdio",
                            "command": sys.executable,
                            "args": [str(server)],
                        }
                    },
                )
            )
            self.assertEqual(answer, "Done")
            self.assertIn("echoed: hello", trace.path.read_text())

    def test_openrouter_continuation_fields_round_trip_without_trace_disclosure(self):
        async def exercise(root):
            model = make_model("minimax/minimax-m3", "test-key", 100)
            details = [
                {"type": "reasoning.text", "text": "opaque-continuation", "signature": "sig"}
            ]
            try:
                response = {
                    "id": "test",
                    "model": "minimax/minimax-m3",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_details": details,
                                "reasoning": "legacy-continuation",
                                "tool_calls": [
                                    {
                                        "id": "tool1",
                                        "type": "function",
                                        "function": {"name": "list_files", "arguments": "{}"},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                }
                from openai.types.chat import ChatCompletion

                response.update(object="chat.completion", created=1)
                typed = ChatCompletion.model_validate(response)
                message = model._create_chat_result(typed).generations[0].message
                payload = model._get_request_payload(
                    [
                        HumanMessage(content="inspect"),
                        message,
                        ToolMessage(content="code.py", tool_call_id="tool1"),
                    ]
                )
                self.assertEqual(payload["messages"][1]["reasoning_details"], details)
                self.assertEqual(payload["messages"][1]["reasoning"], "legacy-continuation")
                trace = Trace(root / "trace.jsonl", "test")
                trace.emit("model", "Response", usage=message.usage_metadata)
                self.assertNotIn("opaque-continuation", trace.path.read_text())
            finally:
                await model.root_async_client.close()
                model.root_client.close()

        with tempfile.TemporaryDirectory() as temp:
            asyncio.run(exercise(Path(temp)))

    def test_report_does_not_credit_resampling_as_harness_improvement(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source").mkdir()
            for label in ("base1", "c1"):
                folder = root / "executions" / "task" / label
                (folder / "workspace").mkdir(parents=True)
                (folder / "answer.txt").write_text("done")
                (folder / "metrics.json").write_text(json.dumps({"label": label, "wall_s": 1}))
            result = {
                "status": "ok",
                "selections": {"seq": {"chosen_label": "c1", "chosen_tree_is_base": True}},
            }

            def checked(passed):
                return {
                    "passed": passed,
                    "exit_code": 0 if passed else 1,
                    "timed_out": False,
                    "output": "",
                    "test_counts": {},
                }

            adapter = SimpleNamespace(
                root=root, source=root / "source", trace=Trace(root / "trace.jsonl", "test")
            )
            with patch(
                "harness2.coding.report.evaluate",
                side_effect=[checked(False), checked(False), checked(True)],
            ):
                report = build_report(
                    adapter,
                    result=result,
                    commands={"project_tests": "test"},
                    timeout=1,
                    usage={"calls": 0, "input_tokens": 0, "output_tokens": 0},
                )
            self.assertIn("not evidence of harness improvement", report["verdict"])

    def test_unlimited_mode_runs_past_old_turn_and_graph_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = SequenceModel(
                [call("list_files", {}, str(i)) for i in range(35)] + [AIMessage(content="Done")]
            )
            budget = Budget(0)
            answer = asyncio.run(
                run_agent(
                    workspace=root,
                    harness=root,
                    task="inspect",
                    model=model,
                    budget=budget,
                    trace=Trace(root / "trace.jsonl", "unlimited"),
                    max_steps=0,
                )
            )
            self.assertEqual(answer, "Done")
            self.assertEqual(budget.calls, 36)
            events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
            self.assertFalse(any(e["event"] == "error" for e in events))
            for _ in range(100):
                budget.reserve()
            self.assertEqual(budget.calls, 136)

    def test_checkpoint_resumes_without_repeating_completed_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state_file = root / "state.json"
            trace = Trace(root / "trace.jsonl", "checkpoint")
            model = SequenceModel([call("write_file", {"path": "code.py", "content": "x=42\n"})])
            with self.assertRaisesRegex(RuntimeError, "exhausted 1 turns"):
                asyncio.run(
                    run_agent(
                        workspace=root,
                        harness=root,
                        task="write code",
                        model=model,
                        budget=Budget(0),
                        trace=trace,
                        max_steps=1,
                        state_file=state_file,
                    )
                )
            self.assertEqual((root / "code.py").read_text(), "x=42\n")
            self.assertEqual(state_file.stat().st_mode & 0o777, 0o600)
            result = asyncio.run(
                run_agent(
                    workspace=root,
                    harness=root,
                    task="write code",
                    model=SequenceModel([AIMessage(content="Finished")]),
                    budget=Budget(0),
                    trace=trace,
                    max_steps=0,
                    state_file=state_file,
                )
            )
            self.assertEqual(result, "Finished")
            saved = json.loads(state_file.read_text())
            self.assertEqual(saved["steps"], 2)
            self.assertTrue(saved["completed"])
            events = [json.loads(line) for line in trace.path.read_text().splitlines()]
            self.assertEqual(sum(e["event"] == "diff" for e in events), 1)

    def test_continue_command_preserves_legacy_partial_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            output = root / "run"
            with (
                patch(
                    "harness2.coding.cli.make_model",
                    return_value=SequenceModel(
                        [
                            call(
                                "write_file",
                                {"path": "progress.py", "content": "important_progress=1\n"},
                            )
                        ]
                    ),
                ),
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            ):
                status = main(
                    [
                        "run",
                        "--repo",
                        str(source),
                        "--task",
                        "build",
                        "--output",
                        str(output),
                        "--max-steps",
                        "1",
                    ]
                )
            self.assertEqual(status, 1)
            checkpoint = next((output / "executions").rglob("state.json"))
            workspace = checkpoint.parent / "workspace"
            checkpoint.unlink()  # The user's original interrupted run predates checkpoints.
            with (
                patch(
                    "harness2.coding.cli.make_model",
                    return_value=SequenceModel([AIMessage(content="Finished")]),
                ),
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
            ):
                self.assertEqual(main(["continue", str(output)]), 0)
            self.assertEqual((workspace / "progress.py").read_text(), "important_progress=1\n")
            self.assertTrue(checkpoint.exists())
            self.assertEqual(json.loads((output / "config.json").read_text())["max_steps"], 0)

    def test_truncated_response_continues_instead_of_finishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            partial = AIMessage(
                content="Partial output",
                response_metadata={"finish_reason": "length"},
                tool_calls=[
                    {
                        "id": "partial",
                        "name": "write_file",
                        "args": {"path": "bad.py", "content": "partial"},
                    }
                ],
            )
            model = SequenceModel(
                [
                    partial,
                    call("write_file", {"path": "good.py", "content": "complete=1\n"}),
                    AIMessage(content="Finished"),
                ]
            )
            answer = asyncio.run(
                run_agent(
                    workspace=root,
                    harness=root,
                    task="finish",
                    model=model,
                    budget=Budget(5),
                    trace=Trace(root / "trace.jsonl", "cutoff"),
                    state_file=root / "state.json",
                )
            )
            self.assertEqual(answer, "Finished")
            self.assertFalse((root / "bad.py").exists())
            self.assertTrue((root / "good.py").exists())
            self.assertTrue(json.loads((root / "state.json").read_text())["completed"])

    def test_continue_can_increase_response_budget(self):
        from harness2.coding.cli import continue_run, parser_for_cli

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "action": "run",
                        "repo": str(root),
                        "task": "code",
                        "model": "test/model",
                        "max_tokens": 6000,
                        "mcp_enabled": False,
                    }
                )
            )
            args = parser_for_cli().parse_args(["continue", str(root), "--max-tokens", "24000"])
            with patch("harness2.coding.cli.main", return_value=0) as entry:
                self.assertEqual(continue_run(args), 0)
            options = entry.call_args.args[0]
            self.assertEqual(options[options.index("--max-tokens") + 1], "24000")
            self.assertEqual(options.count("--max-tokens"), 1)

    def test_cli_run_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "repo"
            source.mkdir()
            (source / "calc.py").write_text("def add(a,b): return a-b\n")
            output = root / "run"
            with (
                patch("harness2.coding.cli.make_model", return_value=WorkflowModel()),
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-not-a-real-key"}),
            ):
                code = main(
                    [
                        "run",
                        "--repo",
                        str(source),
                        "--task",
                        "fix addition",
                        "--output",
                        str(output),
                        "--test-command",
                        "python -m compileall -q .",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(main(["replay", str(output), "--run", "base1"]), 0)
            with (
                patch(
                    "harness2.coding.cli.make_model",
                    side_effect=AssertionError("cached run must not call a model"),
                ),
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-not-a-real-key"}),
            ):
                resumed = main(
                    [
                        "run",
                        "--repo",
                        str(source),
                        "--task",
                        "fix addition",
                        "--output",
                        str(output),
                        "--test-command",
                        "python -m compileall -q .",
                        "--resume",
                    ]
                )
            self.assertEqual(resumed, 0)
            self.assertTrue((output / "report.json").exists())
            self.assertEqual((source / "calc.py").read_text(), "def add(a,b): return a-b\n")


if __name__ == "__main__":
    unittest.main()
