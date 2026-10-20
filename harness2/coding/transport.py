"""Observe HTTP delivery without consuming, decoding, or buffering the stream."""

import time
from contextvars import ContextVar
from dataclasses import dataclass

import httpx


@dataclass
class StreamObservation:
    started: float
    headers_at: float | None = None
    first_byte_at: float | None = None
    last_byte_at: float | None = None
    wire_bytes: int = 0
    wire_chunks: int = 0
    status: int | None = None


active_observation: ContextVar[StreamObservation | None] = ContextVar(
    "coding_stream_observation", default=None
)


class ObservedStream(httpx.AsyncByteStream):
    def __init__(self, inner, observation):
        self.inner = inner
        self.observation = observation

    async def __aiter__(self):
        async for data in self.inner:
            now = time.monotonic()
            state = self.observation
            if data:
                if state.first_byte_at is None:
                    state.first_byte_at = now
                state.last_byte_at = now
                state.wire_bytes += len(data)
                state.wire_chunks += 1
            yield data

    async def aclose(self):
        await self.inner.aclose()


async def observe_response(response):
    state = active_observation.get()
    if state is not None:
        state.headers_at = time.monotonic()
        state.status = response.status_code
        response.stream = ObservedStream(response.stream, state)
