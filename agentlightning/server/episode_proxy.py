# Copyright (c) Microsoft. All rights reserved.
"""Opt-in, durable, bounded OpenCode episode forwarding inside Lightning."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from agentlightning.server.routes.events import record_event
from agentlightning.server.store import _rollouts


def rollout_token(secret: str, rollout_id: str, attempt_id: str) -> str:
    return hmac.new(secret.encode(), f"{rollout_id}:{attempt_id}".encode(), hashlib.sha256).hexdigest()


def journal(directory: str, rollout_id: str, event: dict):
    path = Path(directory) / (hashlib.sha256(rollout_id.encode()).hexdigest() + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sse_response(response: dict):
    """Transport conversion of a real response; tokens are never reconstructed."""
    common = {k: response[k] for k in ("id", "created", "model") if k in response}
    common["object"] = "chat.completion.chunk"
    choice = response["choices"][0]
    message = choice["message"]
    delta = {"role": "assistant"}
    for key in ("content", "reasoning_content", "reasoning"):
        if message.get(key) is not None:
            delta[key] = message[key]
    if "reasoning" in delta and "reasoning_content" not in delta:
        delta["reasoning_content"] = delta["reasoning"]
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **tool} for i, tool in enumerate(message["tool_calls"])]
    chunks = [
        {**common, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
        {**common, "choices": [], "usage": response.get("usage", {})},
    ]
    return "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"


async def forward_episode(request, rollout_id: str, attempt_id: str, upstream_path: str, body: dict):
    state = request.app.state
    rollout = _rollouts.get(rollout_id)
    if rollout is None:
        raise HTTPException(404, "unknown rollout")
    meta = rollout.metadata.model_dump()
    if attempt_id != str(meta.get("attempt", 0)) or rollout.status.state != "running":
        raise HTTPException(409, "inactive rollout attempt")
    if not hasattr(state, "episode_states"):
        state.episode_states = {}
    ep = state.episode_states.setdefault(
        (rollout_id, attempt_id),
        {
            "lock": asyncio.Lock(),
            "sequence": 0,
            "tokens": 0,
            "tools": 0,
            "deadline": time.monotonic() + meta["budgets"]["agent_seconds"],
        },
    )

    def emit(kind, data):
        event = {
            "event_type": kind,
            "rollout_id": rollout_id,
            "attempt_id": attempt_id,
            "timestamp": time.time(),
            "data": data,
        }
        journal(state.episode_journal, rollout_id, event)
        record_event(rollout_id, attempt_id, kind, data)

    credential = request.headers.get("authorization", "").removeprefix("Bearer ")
    if credential and credential in json.dumps(body):
        ep["failed"] = True
        emit("infrastructure", {"type": "credential_in_payload"})
        raise HTTPException(502, "credential appeared in payload; raw payload not recorded")

    if upstream_path == "events":
        kind = body.get("event_type")
        if kind not in {"tool_start", "tool_end", "compaction"}:
            raise HTTPException(403, "agents cannot write rewards or model events")
        if kind == "tool_start":
            ep["tools"] += 1
            if ep["tools"] > meta["budgets"]["tool_calls"]:
                emit("budget", {"reason": "tool_calls"})
                raise HTTPException(429, "tool budget exhausted")
        emit(kind, body.get("data", {}))
        return JSONResponse({"ok": True})
    if upstream_path != "chat/completions":
        raise HTTPException(404, "only Chat Completions supported")
    async with ep["lock"]:
        if ep.get("failed"):
            raise HTTPException(409, "attempt invalidated by an earlier infrastructure failure")
        pause = state.proxy_pause_state
        async with pause.lock:
            if pause.paused:
                raise HTTPException(409, "serving phase closed")
            pause.inflight += 1
        try:
            budgets = meta["budgets"]
            remaining = ep["deadline"] - time.monotonic()
            if remaining <= 0 or ep["sequence"] >= budgets["model_calls"] or ep["tokens"] >= budgets["output_tokens"]:
                emit("budget", {"reason": "model_budget"})
                raise HTTPException(429, "episode budget exhausted")
            server = state.proxy_router.select_server(state.proxy_router.model_name, rollout_id)
            if server.version != meta["policy_version"]:
                raise HTTPException(409, "checkpoint changed within rollout")
            kind = request.headers.get("x-cyber-call-kind", "unknown")
            if kind not in {"action", "title", "summary", "compaction"}:
                kind = "unknown"
            sequence = ep["sequence"]
            ep["sequence"] += 1
            prepared = {
                **body,
                "stream": False,
                "n": 1,
                "model": server.model,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "logprobs": True,
                "top_logprobs": 0,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "max_tokens": min(budgets["output_per_call"], budgets["output_tokens"] - ep["tokens"]),
                "seed": meta["seed"] + sequence,
                "chat_template_kwargs": {
                    "enable_thinking": meta.get("enable_thinking", True),
                    "preserve_thinking": True,
                },
            }
            for key in ("stream_options", "max_completion_tokens", "reasoning_effort"):
                prepared.pop(key, None)
            if credential and credential in json.dumps(prepared):
                raise ValueError("credential appeared in model input; raw payload not recorded")
            admission = {
                "sequence": sequence,
                "request_id": uuid.uuid4().hex,
                "kind": kind,
                "checkpoint_id": meta["checkpoint_id"],
                "request": prepared,
            }
            emit("model_admitted", admission)
            start = time.monotonic()
            response = await state.http_client.post(
                server.endpoint.rstrip("/") + "/chat/completions", json=prepared, timeout=remaining
            )
            if response.status_code == 400 and any(
                term in response.text.lower() for term in ("maximum context length", "max_model_len", "context length")
            ):
                ep["failed"] = True
                emit("budget", {"reason": "context_tokens"})
                raise HTTPException(429, "context budget exhausted")
            response.raise_for_status()
            response_body = response.json()
            if credential and credential in json.dumps(response_body):
                raise ValueError("credential appeared in model output; raw payload not recorded")
            choices = response_body.get("choices", [])
            if len(choices) != 1:
                raise HTTPException(502, "expected exactly one model completion")
            ep["tokens"] += len(choices[0].get("token_ids") or [])
            emit(
                "model_request",
                {
                    **admission,
                    "response": response_body,
                    "latency_ms": (time.monotonic() - start) * 1000,
                    "server": {"model": server.model, "version": server.version},
                },
            )
            if body.get("stream"):
                return StreamingResponse(iter([sse_response(response_body)]), media_type="text/event-stream")
            return JSONResponse(response_body)
        except HTTPException:
            raise
        except httpx.TimeoutException as exc:
            ep["failed"] = True
            emit("budget", {"reason": "agent_seconds"})
            raise HTTPException(429, "agent time budget exhausted") from exc
        except Exception as exc:
            ep["failed"] = True
            emit("infrastructure", {"type": type(exc).__name__})
            raise HTTPException(502, "upstream request failed; attempt ungraded") from exc
        finally:
            async with pause.lock:
                pause.inflight -= 1
