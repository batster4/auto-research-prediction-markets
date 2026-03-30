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
    # Strip optional markdown fences (model sometimes wraps JSON)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    # First [ ... ] span (non-greedy inner match can fail on nested strings; try last resort)
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    # Prefer a JSON array at the **end** of the reply (models often append it after prose).
    for i in range(len(text) - 1, -1, -1):
        if text[i] != "[":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            continue
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            continue
    preview = text[:800] + ("…" if len(text) > 800 else "")
    raise ValueError(f"Could not parse JSON array from model response. Text preview:\n{preview}")


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
    system: str | None = None,
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
    if system:
        kwargs["system"] = system
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
        *,
        experiment_root: str = "",
        research_line_label: str = "",
    ) -> list[StrategySpec]:
        """Ask the model for a JSON array of strategy specs."""
        domain = get_domain_context()
        prior = json.dumps(prior_iterations, indent=2) if prior_iterations else "[]"
        exp_id = experiment_root or "(single experiment)"
        line = research_line_label or "this research line"
        user = f"""Experiment isolation (mandatory):
- This run is **standalone**: experiment directory `{exp_id}`, research line `{line}`.
- Optimize **only** for the research task below. Do **not** align proposals with other parallel studies, other experiment folders, or any hypothetical "global best" PnL from elsewhere.
- The "prior iterations" JSON below is **only** from **this** experiment on **this** dataset — it never includes results from other tasks, processes, or research tracks.
- Do **not** pick a strategy because it might match outcomes from another unrelated study. Ground every proposal in **this** task text and **this** prior JSON only.
- Web search (if used) is for **methods and theory** tied to **this** task — not to copy conclusions from unrelated experiments.

Research task:
{task}

Domain reference (follow these mechanics):
{domain}

Research memory — prior iterations for **this experiment only** (newest last). Each entry has iteration number, candidate strategy metrics, and best_in_iteration.
Use this to **avoid repeating** parameter sets that already failed or plateaued **within this run**; refine or explore new regions. Prefer diversity when prior winners are clear.
{prior}

External knowledge: You have **web search** when enabled. Use it sparingly (a few queries per turn) to find **recent** approaches relevant to **this** task: academic papers, quant notes, prediction-market or binary-option research, GBM/digital formulations, and similar — then map findings into our JSON `threshold` / `momentum` / `hold` templates with realistic numeric params.

Respond with ONLY a JSON array (no markdown fences, no commentary) of at most {max_strategies} objects.
The **last** assistant text must be **only** valid JSON starting with `[` and ending with `]` — no words before or after.
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
        system = (
            "You output strategy proposals as machine-readable JSON only. "
            "The final user-visible assistant text must be a single JSON array and nothing else — "
            "no markdown, no headings, no bullet lists, no explanation outside the array. "
            "Each experiment is independent: do not assume or import strategy choices from other research lines."
        )

        last_err: ValueError | None = None
        for attempt in range(2):
            msg = _messages_complete(
                self._client,
                model=self._model,
                message_list=messages,
                max_tokens=int(self._max_output_tokens),
                thinking=thinking if attempt == 0 else None,
                tools=tools if attempt == 0 else None,
                system=system,
            )

            text = _extract_text_blocks(msg)
            if not text:
                last_err = ValueError("Empty text in model response (no JSON to parse).")
            else:
                try:
                    raw_list = _extract_json_list(text)
                    specs: list[StrategySpec] = []
                    for item in raw_list:
                        specs.append(StrategySpec.model_validate(item))
                    return specs
                except ValueError as e:
                    last_err = e

            if attempt == 0 and msg is not None:
                messages.append({"role": "assistant", "content": getattr(msg, "content", [])})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your reply was not a valid JSON array. "
                            "Reply with ONLY a JSON array of strategy objects — "
                            "the first character must be '[' and the last must be ']'. "
                            "No other text."
                        ),
                    }
                )

        assert last_err is not None
        # Third attempt: minimal prompt, no tools/thinking — turn prose + context into JSON only.
        recovery_user = f"""You must output ONLY a JSON array (no markdown, no explanation).

This experiment remains isolated (directory `{exp_id}`, line `{line}`). Do not mirror strategies from other parallel research runs.

Research task (reminder):
{task[:6000]}

Domain reference:
{domain[:8000]}

Prior iterations for this experiment only (JSON):
{prior[:20000]}

The model previously answered with prose or invalid JSON. Using **only** this task, domain, and this prior JSON,
propose up to {max_strategies} distinct strategy specs as ONE JSON array.
Each element: {{"type": "threshold" | "momentum" | "hold", "params": {{...}}}}
Examples: {{"type": "threshold", "params": {{"buy_below": 0.35}}}}, {{"type": "momentum", "params": {{"lookback": 3, "buy_if_rising": 1}}}}, {{"type": "hold", "params": {{}}}}.

First character of your reply MUST be '['. Last character MUST be ']'."""

        recovery_system = (
            "Emit a single JSON array only. No keys but the array root. "
            "No ``` fences. No text before or after the array."
        )
        rec_msg = _messages_complete(
            self._client,
            model=self._model,
            message_list=[{"role": "user", "content": recovery_user}],
            max_tokens=min(16_384, int(self._max_output_tokens)),
            thinking=None,
            tools=None,
            system=recovery_system,
        )
        rec_text = _extract_text_blocks(rec_msg)
        if not rec_text:
            raise last_err
        try:
            raw_list = _extract_json_list(rec_text)
            specs = [StrategySpec.model_validate(item) for item in raw_list]
            return specs
        except ValueError:
            raise last_err
