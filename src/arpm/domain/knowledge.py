"""Base domain knowledge for prediction markets (agent prompts and human reference).

This module encodes mechanics so the research agent can reason correctly about
strategies, positions, settlement, and PnL.
"""

from __future__ import annotations

PREDICTION_MARKETS_KNOWLEDGE = """# Prediction markets — mechanics for strategy research

## What a prediction market is

A **prediction market** is a market where participants trade contracts whose
payoffs depend on the outcome of a future, verifiable event (often binary:
YES/NO). Prices aggregate beliefs about probabilities; trading transfers risk
and can reward accurate forecasting.

## How a market functions

- A **market** (or **contract**) references one or more **outcomes** (e.g. “Team A wins”).
- Participants place **orders** to buy or sell outcome shares. A **central limit order book**
  matches bids and asks; some platforms use **automated market makers (AMMs)** instead.
- **Liquidity** and **fees** vary by venue; always model fees explicitly when backtesting.

## The `market` field (IDs and structure)

In datasets, **`market_id`** (or equivalent) is a **stable identifier** for one contract
or one event instance. All rows with the same `market_id` belong to the same market.
Do not assume unrelated markets share timing or resolution rules.

## Shares and positions

- **YES shares** pay **$1 per share** if the event resolves YES, else **$0**.
- **NO shares** pay **$1 per share** if the event resolves NO, else **$0**.
- YES and NO are complementary views of the same binary event; at fair prices,
  implied YES price + implied NO price ≈ 1 (before fees).
- **Position size** is measured in shares (and optionally notional exposure).

## Prices, probabilities, and implied odds

- Traded **price** for YES is often interpreted as **implied probability** of YES
  (e.g. 0.35 ≈ 35% implied chance), ignoring fees and risk premia.
- **Implied odds** can be expressed as decimal odds: e.g. implied probability `p`
  corresponds to decimal odds `1/p` for a simple YES bet (before fees).

## Payouts and settlement

- At **resolution**, the oracle/venue declares the outcome for the event.
- Each YES share pays **1** if outcome is YES, **0** otherwise; NO shares pay **1** if NO, **0** otherwise.
- **Settlement** converts final share holdings into cash; open orders may be cancelled
  according to venue rules before resolution.

## Resolution

- **Resolution** is the process of determining the event outcome using predefined rules.
- Resolution may be **delayed** or **disputed** on real venues; research datasets should
  state whether outcomes are **final** in the data.

## Profits and losses (PnL)

For a **long YES** position of `n` shares entered at average price `p` per share (ignoring fees):

- If outcome is YES: payoff per share is `(1 - p)`; total `n * (1 - p)` vs entry cost already paid.
- Equivalently: **PnL** = `n * (outcome - p)` where `outcome ∈ {0, 1}` for YES shares.

Include **fees** as adjustments to effective entry/exit prices or as explicit deductions.

## Research implications

- Backtests must define **what each row means** (quote vs trade, whose side, fee structure).
- Strategies should separate **signal** from **execution** and **risk** (position sizing).
"""


def get_domain_context() -> str:
    """Return domain text suitable for system/developer prompts."""
    return PREDICTION_MARKETS_KNOWLEDGE.strip()
