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
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
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
    """Run Messages API with streaming until stop (handles web-search pauses)."""
    cur: list[dict[str, Any]] = list(message_list)
    kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens}
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
        if getattr(resp, "stop_reason", None) != "pause_turn":
            return resp
        cur.append({"role": "assistant", "content": resp.content})
    return resp


_STRATEGY_CATALOG: dict[str, str] = {
    "threshold": """\
"threshold" — Buy YES first time price <= buy_below.
  params: {"buy_below": <float 0.01–0.95>}
  Economically: value-buy at fixed implied-probability ceiling.""",
    "momentum": """\
"momentum" — Buy when lookback-tick price change matches direction.
  params: {"lookback": <int 2–30>, "buy_if_rising": <bool>}
  Rising=true catches uptrends early; rising=false is contrarian.""",
    "early_threshold": """\
"early_threshold" — Threshold restricted to the first X% of the tradeable window.
  params: {"buy_below": <float>, "entry_window_pct": <float 0.1–0.9>}
  Forces early entry when information advantage is largest.""",
    "mean_reversion": """\
"mean_reversion" — Buy after a ≥drop_pct fall from the rolling lookback-high.
  params: {"drop_pct": <float 0.03–0.50>, "lookback": <int 3–20>}
  Catches overreactions / panic sells.""",
    "relative_value": """\
"relative_value" — Buy when price < (fair_value − edge_required).
  params: {"fair_value": <float 0.30–0.70>, "edge_required": <float 0.01–0.20>}
  Requires an explicit probability estimate — vary fair_value around the dataset base rate.""",
    "ma_crossover": """\
"ma_crossover" — Buy when price drops below its window-tick MA by discount.
  params: {"window": <int 3–30>, "discount": <float 0.01–0.20>}
  Technical: price below moving average signals undervaluation.""",
    "hold": """\
"hold" — No trade (flat baseline).
  params: {}""",
    "bs_fair_value": """\
"bs_fair_value" — Black-Scholes fair value: models the market as a binary call,
  infers moneyness from opening price, re-prices with BS at each tick.
  Buys when market price < fair_value − edge.
  params: {"vol_annual": <float 0.3–2.0>, "edge_required": <float 0.01–0.15>,
           "min_tte_pct": <float 0.1–0.8>, "max_tte_pct": <float 0.3–0.95>,
           "warmup_ticks": <int 3–15>}
  Key param: vol_annual is the annualised BTC volatility assumption (typical: 0.5–1.2).""",
    "bs_overreaction": """\
"bs_overreaction" — Buy when a price drop exceeds N standard deviations of
  BS-predicted per-tick volatility (i.e. the market overreacted to a small move).
  params: {"vol_annual": <float 0.3–2.0>, "z_threshold": <float 1.0–3.5>,
           "vol_window": <int 5–20>}
  Higher z_threshold = more selective but higher confidence.""",
    "gamma_scalp": """\
"gamma_scalp" — Buy near-ATM binary options (price ≈ 0.5 ± atm_band) after a dip.
  Binary gamma peaks near ATM → small underlying moves cause large option swings
  → market is prone to overreaction → buy the dip.
  params: {"atm_band": <float 0.05–0.25>, "dip_threshold": <float 0.02–0.15>,
           "max_tte_pct": <float 0.3–0.9>, "min_tte_pct": <float 0.1–0.6>,
           "lookback": <int 3–15>}""",
}

GENERAL_TYPES = {"threshold", "momentum", "early_threshold", "mean_reversion",
                 "relative_value", "ma_crossover", "hold"}
BS_TYPES = {"bs_fair_value", "bs_overreaction", "gamma_scalp"}


def _build_strategy_dsl(allowed_types: set[str] | None = None) -> str:
    """Build the strategy DSL section, filtered to *allowed_types* if given."""
    types_to_show = allowed_types or set(_STRATEGY_CATALOG.keys())
    lines = ["Available strategy types (use ONLY these in your JSON):\n"]
    idx = 1
    for name in _STRATEGY_CATALOG:
        if name not in types_to_show:
            continue
        lines.append(f"{idx}. {_STRATEGY_CATALOG[name]}\n")
        idx += 1
    if allowed_types:
        lines.append(
            f"You MUST ONLY use the {len(types_to_show)} types listed above. "
            f"Any other type will be rejected.\n"
        )
    return "\n".join(lines)

_ENGINE_NOTES = """\
IMPORTANT — backtest engine realism (all applied automatically):
• Resolution cutoff: the last 60 seconds of each market are INVISIBLE to strategies.
  You cannot buy at price=0 near expiry — the engine removes those rows.
• Slippage: +$0.005 is added to your entry price (models execution cost).
• Taker fees: Polymarket fee schedule is applied per share on entry.
• Strategies see (timestamp, price_yes, time_to_expiry_s) — never the outcome.
  time_to_expiry_s = seconds until market resolution (parsed from market_id).
• Metrics shown are from the TRAIN split (70% of markets by time).
  A held-out TEST split (30%) is evaluated separately — you NEVER see test numbers.
• buy_below=0.0 will enter ~zero markets and earn nothing.
  Use realistic thresholds (0.10–0.50) to actually trade.

ROBUSTNESS FIELD (in each iteration record):
  Each past iteration includes a "robustness" label: strong / moderate / weak / poor.
  This tells you how well the best strategy generalised to unseen test data:
    strong   = generalises very well — refine this direction
    moderate = somewhat positive on test — promising, keep exploring
    weak     = marginal or slightly negative on test — likely overfitting, pivot
    poor     = significantly negative on test — overfitting confirmed, STOP refining, change approach
  If recent iterations show "weak"/"poor", you MUST try fundamentally different strategies.
"""


class ClaudeResearchClient:
    """Thin wrapper around ``anthropic.Anthropic`` for strategy proposals."""

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
        stagnation_warning: str | None = None,
        banned_types: set[str] | None = None,
        allowed_types: set[str] | None = None,
    ) -> list[StrategySpec]:
        """Ask the model for a JSON array of strategy specs."""
        domain = get_domain_context()
        prior = json.dumps(prior_iterations, indent=2) if prior_iterations else "[]"
        exp_id = experiment_root or "(single experiment)"
        line = research_line_label or "this research line"

        strategy_dsl = _build_strategy_dsl(allowed_types)

        # Effective allowed set for diversity requirement
        effective_n_types = len(allowed_types) if allowed_types else len(_STRATEGY_CATALOG)
        min_diverse = min(3, effective_n_types)

        # Build dynamic blocks
        stagnation_block = ""
        if stagnation_warning:
            stagnation_block = f"\n⚠️ {stagnation_warning}\n"

        banned_block = ""
        if banned_types:
            visible_banned = banned_types - (set() if allowed_types is None else (banned_types - allowed_types))
            if visible_banned:
                banned_block = (
                    f"\nBANNED STRATEGY TYPES (do NOT use): {', '.join(sorted(visible_banned))}.\n"
                    f"Any candidate using a banned type will be automatically discarded.\n"
                )

        user = f"""\
Experiment isolation (mandatory):
- Standalone experiment: directory `{exp_id}`, research line `{line}`.
- Optimise ONLY for the task below.  Do NOT align with other parallel experiments.
- Prior iterations JSON is from THIS experiment only.
{stagnation_block}{banned_block}
Research task:
{task}

Domain reference:
{domain}

{_ENGINE_NOTES}

{strategy_dsl}

Research memory — prior iterations (newest last):
{prior}

Guidelines:
- Study prior results carefully.  If a strategy type/param range plateaued, pivot to a DIFFERENT type or region.
- DIVERSITY REQUIREMENT: your {max_strategies} candidates MUST include at least {min_diverse} different strategy types.
  Submitting 5 variations of the same type is wasteful — explore the strategy space broadly.
- Pay attention to the "robustness" field in prior iterations:
  "weak"/"poor" = the strategy overfits training data. Do NOT keep refining it. Try something different.
  "strong"/"moderate" = promising direction, but still diversify your other candidates.
- Avoid "buy_below" near 0 — the resolution cutoff makes ultra-low thresholds ineffective.
- Think about WHEN to enter (early vs mid-market), not just at what price.
- Consider the dataset base rate (~50% YES) when setting fair_value / thresholds.

Respond with ONLY a JSON array (no markdown fences, no commentary) of at most {max_strategies} objects.
Each object: {{"type": "<type>", "params": {{...}}}}
First character must be '[', last must be ']'."""

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
            "The final assistant text must be a single JSON array and nothing else — "
            "no markdown, no headings, no explanation outside the array. "
            "Each experiment is independent."
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
                last_err = ValueError("Empty text in model response.")
            else:
                try:
                    raw_list = _extract_json_list(text)
                    return [StrategySpec.model_validate(item) for item in raw_list]
                except ValueError as e:
                    last_err = e

            if attempt == 0 and msg is not None:
                messages.append({"role": "assistant", "content": getattr(msg, "content", [])})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your reply was not a valid JSON array. "
                        "Reply with ONLY a JSON array of strategy objects — "
                        "first char '[', last char ']'. No other text."
                    ),
                })

        assert last_err is not None

        recovery_user = f"""\
You must output ONLY a JSON array (no markdown, no explanation).
Experiment: directory `{exp_id}`, line `{line}`.

{_STRATEGY_DSL}

{_ENGINE_NOTES}

Research task (reminder):
{task[:6000]}

Prior iterations (JSON):
{prior[:20000]}

Propose up to {max_strategies} distinct strategy specs as ONE JSON array.
First character '['. Last character ']'."""

        rec_msg = _messages_complete(
            self._client,
            model=self._model,
            message_list=[{"role": "user", "content": recovery_user}],
            max_tokens=min(16_384, int(self._max_output_tokens)),
            thinking=None,
            tools=None,
            system="Emit a single JSON array only. No fences. No text before or after.",
        )
        rec_text = _extract_text_blocks(rec_msg)
        if not rec_text:
            raise last_err
        try:
            raw_list = _extract_json_list(rec_text)
            return [StrategySpec.model_validate(item) for item in raw_list]
        except ValueError:
            raise last_err
