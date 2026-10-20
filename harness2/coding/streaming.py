"""Model streaming, provider continuation data, and retryable request failures."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from asyncio import sleep as pause

import httpx
from langchain_core.messages import message_chunk_to_message, messages_to_dict
from langchain_openai.chat_models._client_utils import StreamChunkTimeoutError
from openai import APIConnectionError, APITimeoutError

from .recovery import RequestRecovery, request_identity
from .transport import StreamObservation, active_observation


def error_status(error):
    value = getattr(error, "status_code", None) or getattr(error, "code", None)
    body = getattr(error, "body", None)
    if not value and isinstance(body, dict):
        detail = body.get("error", body)
        value = detail.get("code") if isinstance(detail, dict) else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def retryable(error):
    return error_status(error) in (408, 429, 500, 502, 503, 504) or isinstance(
        error, (APIConnectionError, APITimeoutError, httpx.TransportError, StreamChunkTimeoutError)
    )


def visible_reasoning(delta):
    if delta.get("reasoning"):
        return delta["reasoning"]
    return "".join(
        str(block.get("text") or block.get("summary") or "")
        for block in delta.get("reasoning_details", [])
        if block.get("type") in ("reasoning.text", "reasoning.summary")
    )


def restore_reasoning(message):
    deltas = message.additional_kwargs.pop("_openrouter_deltas", [])
    if not deltas:
        return message
    blocks, identities = [], {}
    reasoning = []
    for delta in deltas:
        if delta.get("reasoning"):
            reasoning.append(delta["reasoning"])
        for raw in delta.get("reasoning_details", []):
            block = copy.deepcopy(raw)
            identity = (block.get("index"), block.get("id"), block.get("type"))
            # Providers may omit type/id on continuation fragments of an indexed block.
            if block.get("index") is not None:
                identity = ("index", block["index"])
            if identity == (None, None, None) or (
                block.get("index") is None and not block.get("id")
            ):
                blocks.append(block)
                continue
            if identity not in identities:
                identities[identity] = len(blocks)
                blocks.append(block)
            else:
                prior = blocks[identities[identity]]
                for key, value in block.items():
                    if key in ("text", "summary", "data", "signature") and isinstance(value, str):
                        prior[key] = prior.get(key, "") + value
                    else:
                        prior[key] = value
    if blocks:
        message.additional_kwargs["reasoning_details"] = blocks
    if reasoning:
        message.additional_kwargs["reasoning"] = "".join(reasoning)
    return message


def content_text(content):
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") in ("text", "output_text")
    )


async def model_response(
    bound,
    messages,
    *,
    budget,
    trace,
    stream=False,
    retries=3,
    recovery_file=None,
    recover_legacy=False,
):
    """Retry API requests only; committed tool actions are never repeated here."""
    recovery = RequestRecovery(
        recovery_file,
        request_identity(bound, messages),
        request_identity(bound, messages, include_model=False),
    )
    if recover_legacy and recovery.import_interrupted_trace(trace):
        trace.emit("stage", "Recovered partial output from the previous interrupted chat trace")
    for attempt in range(retries + 1):
        number = budget.reserve()
        request_messages = recovery.messages(messages)
        recovery.begin_attempt()
        conversation_bytes = len(
            json.dumps(messages_to_dict(request_messages), ensure_ascii=False).encode()
        )
        if recovery.mode != "last committed tool result":
            trace.emit("stage", f"Recovering interrupted response: {recovery.mode}")
        trace.emit(
            "model", f"Request {number}/{budget.limit or 'unlimited'}; attempt {attempt + 1}"
        )
        started = last_activity = time.monotonic()
        observation = StreamObservation(started)
        observation_token = active_observation.set(observation)
        chunks_received = tool_argument_chars = 0
        announced_tools = set()
        announced_provider = None
        finish_reason = None
        trace.emit(
            "model",
            f"Sending {len(request_messages)} messages; conversation {conversation_bytes:,} bytes",
            message_count=len(request_messages),
            conversation_bytes=conversation_bytes,
        )

        async def heartbeat():
            while True:
                await pause(10)
                elapsed = time.monotonic() - last_activity
                if elapsed >= 10:
                    state = (
                        "awaiting HTTP response headers"
                        if observation.headers_at is None
                        else f"HTTP {observation.status}; {observation.wire_bytes:,} transport bytes"
                    )
                    phase = (
                        "Waiting for first model chunk"
                        if not chunks_received
                        else "Waiting for next model chunk"
                    )
                    trace.emit(
                        "model",
                        f"{phase} ({elapsed:.0f}s; {state}; "
                        f"{chunks_received} model chunks received). Ctrl+C interrupts",
                    )

        waiting = asyncio.create_task(heartbeat()) if stream else None
        try:
            if not stream or not hasattr(bound, "astream"):
                result = await bound.ainvoke(request_messages)
                recovery.clear()
                return result
            aggregate = None
            async for chunk in bound.astream(request_messages):
                last_activity = time.monotonic()
                chunks_received += 1
                if chunks_received == 1:
                    trace.emit(
                        "model",
                        f"First model chunk arrived after {last_activity - started:.1f}s",
                        first_chunk_s=round(last_activity - started, 3),
                        wire_bytes=observation.wire_bytes,
                    )
                if chunk.response_metadata.get("finish_reason"):
                    finish_reason = chunk.response_metadata["finish_reason"]
                provider = chunk.response_metadata.get("openrouter_provider")
                if provider and provider != announced_provider:
                    trace.emit("model", f"Serving provider: {provider}", provider=provider)
                    announced_provider = provider
                tool_argument_chars += sum(
                    len(part.get("args") or "") for part in getattr(chunk, "tool_call_chunks", [])
                )
                for part in getattr(chunk, "tool_call_chunks", []):
                    identity = part.get("index")
                    if part.get("name") and identity not in announced_tools:
                        trace.stream("tool_arguments", f"{part['name']} (pending execution)\n")
                        announced_tools.add(identity)
                    if part.get("args"):
                        trace.stream("tool_arguments", part["args"])
                text = content_text(chunk.content)
                if text:
                    trace.stream("assistant", text)
                deltas = chunk.additional_kwargs.get("_openrouter_deltas", [])
                thoughts = []
                for delta in deltas:
                    thinking = visible_reasoning(delta)
                    if thinking:
                        trace.stream("thinking", thinking)
                        thoughts.append(thinking)
                recovery.record(text, "".join(thoughts), deltas)
                aggregate = chunk if aggregate is None else aggregate + chunk
            if aggregate is None:
                raise RuntimeError("Provider returned an empty stream")
            trace.emit(
                "model",
                f"Stream completed: {chunks_received} model chunks received",
                chunks_received=chunks_received,
                wire_bytes=observation.wire_bytes,
                wire_chunks=observation.wire_chunks,
            )
            result = restore_reasoning(message_chunk_to_message(aggregate))
            # OpenRouter repeats terminal metadata on its usage frame. Generic
            # LangChain string merging would produce e.g. "tool_callstool_calls".
            if finish_reason:
                result.response_metadata["finish_reason"] = finish_reason
            if announced_provider:
                result.response_metadata["openrouter_provider"] = announced_provider
            recovery.clear()
            return result
        except Exception as error:
            trace.end_stream()
            body = getattr(error, "body", None)
            detail = body.get("error", body) if isinstance(body, dict) else {}
            metadata = (detail.get("metadata") or {}) if isinstance(detail, dict) else {}
            trace.emit(
                "model",
                f"Request failed after {time.monotonic() - started:.1f}s; "
                f"received {chunks_received} chunks and {tool_argument_chars:,} tool-argument characters",
                error_type=type(error).__name__,
                status=error_status(error),
                provider=metadata.get("provider_name") if isinstance(metadata, dict) else None,
                chunks_received=chunks_received,
                tool_argument_chars=tool_argument_chars,
                wire_bytes=observation.wire_bytes,
                wire_chunks=observation.wire_chunks,
            )
            if attempt == retries or not retryable(error):
                raise
            delay = min(2**attempt, 8)
            trace.emit(
                "retry",
                f"Temporary provider error ({type(error).__name__}, "
                f"status {error_status(error) or 'network'}); "
                f"retrying in {delay}s from {recovery.mode}. Completed tools are retained; "
                "unfinished tool calls are not executed.",
            )
            await asyncio.sleep(delay)
        finally:
            active_observation.reset(observation_token)
            if waiting:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
            trace.end_stream()
