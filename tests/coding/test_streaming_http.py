import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from harness2.coding.agent import Budget, make_model, run_agent
from harness2.coding.trace import Trace


class StreamingHTTPTests(unittest.TestCase):
    def test_response_timeouts_are_disabled_by_default_and_configurable(self):
        from langchain_openai.chat_models._client_utils import StreamChunkTimeoutError

        from harness2.coding.streaming import retryable

        for seconds in (0, 300):
            with self.subTest(seconds=seconds):
                model = make_model("test/model", "test-key", 0, stream_idle_timeout=seconds)
                try:
                    self.assertEqual(model.stream_chunk_timeout, seconds or None)
                    self.assertEqual(model.http_client.timeout.read, seconds or None)
                    self.assertEqual(model.http_async_client.timeout.read, seconds or None)
                    self.assertEqual(model.http_client.timeout.connect, 30)
                finally:
                    model.http_client.close()
                    asyncio.run(model.http_async_client.aclose())
        self.assertTrue(retryable(StreamChunkTimeoutError(300)))

    def test_real_sse_retries_504_streams_before_completion_and_preserves_reasoning(self):
        requests = []
        first_token = threading.Event()
        first_tool_argument = threading.Event()
        first_assistant_text = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                if len(requests) == 1:
                    error = json.dumps(
                        {
                            "error": {
                                "message": "temporary provider failure",
                                "type": "server_error",
                                "code": 504,
                            }
                        }
                    ).encode()
                    self.send_response(504)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(error)))
                    self.end_headers()
                    self.wfile.write(error)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b": OPENROUTER PROCESSING\n\n")
                self.wfile.flush()

                def send(delta, finish=None, usage=None):
                    chunk = {
                        "id": "test-stream",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "test/model",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    if usage:
                        chunk["usage"] = usage
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()

                if body["messages"][-1]["role"] != "tool":
                    send(
                        {
                            "role": "assistant",
                            "reasoning_details": [
                                {
                                    "type": "reasoning.text",
                                    "index": 0,
                                    "format": "provider-v1",
                                    "text": "Inspect ",
                                }
                            ],
                        }
                    )
                    if not first_token.wait(3):
                        raise AssertionError("Client did not stream an early token")
                    send(
                        {
                            "reasoning_details": [
                                {
                                    "type": "reasoning.text",
                                    "index": 0,
                                    "format": "provider-v1",
                                    "text": "then edit.",
                                }
                            ]
                        }
                    )
                    send(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "write-1",
                                    "type": "function",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": '{"path":"code.py",',
                                    },
                                }
                            ]
                        }
                    )
                    if not first_tool_argument.wait(3):
                        raise AssertionError("Tool arguments were buffered until completion")
                    send(
                        {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"content":"x=42\\n"}'}}
                            ]
                        }
                    )
                    send(
                        {},
                        "tool_calls",
                        {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                    )
                else:
                    send({"role": "assistant", "content": "Completed "})
                    if not first_assistant_text.wait(3):
                        raise AssertionError("Assistant text was buffered until completion")
                    send(
                        {"content": "the edit."},
                        "stop",
                        {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
                    )
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                trace = Trace(root / "trace.jsonl", "sse", thinking=True)
                original = trace.stream

                def stream(channel, text):
                    first_token.set()
                    original(channel, text)
                    if channel == "tool_arguments" and '{"path"' in text:
                        first_tool_argument.set()
                    if channel == "assistant" and text == "Completed ":
                        first_assistant_text.set()

                trace.stream = stream
                budget = Budget(5)

                def model():
                    return make_model(
                        "test/model",
                        "test-key",
                        100,
                        base_url=f"http://127.0.0.1:{server.server_port}/v1",
                        reasoning_effort="low",
                    )

                with patch("harness2.coding.streaming.asyncio.sleep", return_value=None):
                    answer = asyncio.run(
                        run_agent(
                            workspace=root,
                            harness=root,
                            task="edit",
                            model=model,
                            budget=budget,
                            trace=trace,
                            stream=True,
                        )
                    )
                self.assertEqual(answer, "Completed the edit.")
                self.assertEqual((root / "code.py").read_text(), "x=42\n")
                self.assertEqual(budget.calls, 3)
                self.assertEqual(requests[-1]["reasoning"], {"exclude": False, "effort": "low"})
                outgoing = requests[-1]["messages"][-2]
                self.assertEqual(
                    outgoing["reasoning_details"],
                    [
                        {
                            "type": "reasoning.text",
                            "index": 0,
                            "format": "provider-v1",
                            "text": "Inspect then edit.",
                        }
                    ],
                )
                events = [json.loads(line) for line in trace.path.read_text().splitlines()]
                arrivals = [e for e in events if "first_chunk_s" in e]
                self.assertEqual(len(arrivals), 2)
                self.assertTrue(all(e["wire_bytes"] > 0 for e in arrivals))
                self.assertTrue(any(e.get("wire_chunks", 0) > 0 for e in events))
                failures = [e for e in events if e.get("error_type")]
                self.assertEqual(failures[0]["status"], 504)
                self.assertEqual(failures[0]["chunks_received"], 0)
                self.assertTrue(any(e.get("conversation_bytes", 0) > 0 for e in events))
                self.assertEqual(sum(e["event"] == "diff" for e in events), 1)
                self.assertTrue(any(e["event"] == "retry" for e in events))
                self.assertTrue(any(e["event"] == "thinking_delta" for e in events))
                self.assertTrue(any(e["event"] == "assistant_delta" for e in events))
                self.assertTrue(first_tool_argument.is_set())
                self.assertTrue(first_assistant_text.is_set())
                self.assertTrue(any(e["event"] == "tool_arguments_delta" for e in events))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
