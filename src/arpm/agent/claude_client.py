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


class ClaudeResearchClient:
    """Thin wrapper around `anthropic.Anthropic` for strategy proposals."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

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

Previous iteration results (newest last):
{prior}

Respond with ONLY a JSON array (no markdown fences) of at most {max_strategies} objects.
Each object must match: {{"type": "threshold" | "momentum" | "hold", "params": {{...}}}}
Examples:
- {{"type": "threshold", "params": {{"buy_below": 0.35}}}}
- {{"type": "momentum", "params": {{"lookback": 3, "buy_if_rising": 1}}}}
- {{"type": "hold", "params": {{}}}}

Tune parameters using prior results; explore diverse candidates."""

        msg = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user}],
        )
        text = ""
        for block in msg.content:
            if block.type == "text":
                text += block.text
        raw_list = _extract_json_list(text)
        specs: list[StrategySpec] = []
        for item in raw_list:
            specs.append(StrategySpec.model_validate(item))
        return specs
