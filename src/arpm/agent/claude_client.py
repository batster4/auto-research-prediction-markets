"""Anthropic Claude via the official Python SDK (Messages API)."""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic

from arpm.domain.knowledge import get_domain_context
from arpm.strategies.base import StrategySpec


def _extract_json_list(text: str) -> list[Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not parse JSON array from model response.")


def _extract_text_blocks(msg: Any) -> str:
    """Concatenate assistant text blocks (skip thinking / tool metadata)."""
    parts: list[str] = []
    for block in getattr(msg, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(parts).strip()


def _messages_complete(
    client: anthropic.Anthropic,
    *,
    model: str,
    message_list: list[dict[str, Any]],
    max_tokens: int,
    thinking: dict[str, Any] | None,
    tools: list[dict[str, Any]] | None,
    max_pause_turns: int = 8,
) -> Any:
    """Run Messages API until not pause_turn (web search / server tools may pause).

    Uses **streaming** (`messages.stream`) — the Python SDK requires streaming for
    requests that may exceed the non-streaming ~10 minute path when using large
    budgets, thinking, and/or web search.
    """
    cur: list[dict[str, Any]] = list(message_list)
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
    }
    if thinking is not None:
        kwargs["thinking"] = thinking
    if tools:
        kwargs["tools"] = tools

    resp: Any = None
    for _ in range(max_pause_turns):
        kwargs["messages"] = cur
        with client.messages.stream(**kwargs) as stream:
            for _ in stream.text_stream:
                pass
            resp = stream.get_final_message()
        reason = getattr(resp, "stop_reason", None)
        if reason != "pause_turn":
            return resp
        cur.append({"role": "assistant", "content": resp.content})

    return resp


class ClaudeResearchClient:
    """Thin wrapper around `anthropic.Anthropic` for strategy proposals."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        thinking_budget_tokens: int = 16_000,
        max_output_tokens: int = 32_000,
        web_search_enabled: bool = True,
        web_search_max_uses: int = 5,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._thinking_budget = thinking_budget_tokens
        self._max_output_tokens = max_output_tokens
        self._web_search_enabled = web_search_enabled
        self._web_search_max_uses = web_search_max_uses

    def propose_strategies(
        self,
        task: str,
        prior_iterations: list[dict],
        max_strategies: int = 5,
    ) -> list[StrategySpec]:
        """Ask the model for a JSON array of strategy specs."""
        domain = get_domain_context()
        prior = json.dumps(prior_iterations, indent=2) if prior_iterations else "[]"
        user = f"""Research task:
{task}

Domain reference (follow these mechanics):
{domain}

Research memory — prior iterations (newest last). Each entry has iteration number, candidate strategy metrics, and best_in_iteration.
Use this to **avoid repeating** parameter sets that already failed or plateaued; refine or explore new regions. Prefer diversity when prior winners are clear.
{prior}

External knowledge: You have **web search** when enabled. Use it sparingly (a few queries per turn) to find **recent** approaches: academic papers, quant notes, prediction-market or binary-option research, GBM/digital formulations, and similar — then map findings into our JSON `threshold` / `momentum` / `hold` templates with realistic numeric params.

Respond with ONLY a JSON array (no markdown fences) of at most {max_strategies} objects.
Each object must match: {{"type": "threshold" | "momentum" | "hold", "params": {{...}}}}
Examples:
- {{"type": "threshold", "params": {{"buy_below": 0.35}}}}
- {{"type": "momentum", "params": {{"lookback": 3, "buy_if_rising": 1}}}}
- {{"type": "hold", "params": {{}}}}

Tune parameters using prior results; explore diverse candidates."""

        thinking: dict[str, Any] | None = None
        if self._thinking_budget > 0:
            thinking = {"type": "enabled", "budget_tokens": int(self._thinking_budget)}

        tools: list[dict[str, Any]] | None = None
        if self._web_search_enabled and self._web_search_max_uses > 0:
            tools = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": int(self._web_search_max_uses),
                }
            ]

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

        msg = _messages_complete(
            self._client,
            model=self._model,
            message_list=messages,
            max_tokens=int(self._max_output_tokens),
            thinking=thinking,
            tools=tools,
        )

        text = _extract_text_blocks(msg)
        if not text:
            raise ValueError("Empty text in model response (no JSON to parse).")

        raw_list = _extract_json_list(text)
        specs: list[StrategySpec] = []
        for item in raw_list:
            specs.append(StrategySpec.model_validate(item))
        return specs
