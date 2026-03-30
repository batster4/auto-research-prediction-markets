#!/usr/bin/env python3
"""Build HTML report: three experiments, iteration PnL curves, equity curves, metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arpm.backtest.engine import run_backtest  # noqa: E402
from arpm.data.loaders import load_trades_table  # noqa: E402
from arpm.evaluation.metrics import evaluate_returns  # noqa: E402
from arpm.strategies.base import StrategySpec, strategy_from_spec  # noqa: E402

EXP_ROOT = ROOT / "experiments"

LABELS = [
    ("BS — binary / vol mispricing", "batch1_bs", "bs"),
    ("GBM — last-second edge", "batch1_gbm", "gbm"),
    ("OPEN — opening inefficiency", "batch1_open", "open"),
]


def _profit_factor(pm: list[float]) -> float | None:
    wins = sum(x for x in pm if x > 0)
    losses = sum(x for x in pm if x < 0)
    if losses == 0:
        return None if wins == 0 else float("inf")
    return wins / abs(losses)


def _explain_strategy(spec: StrategySpec) -> str:
    if spec.type == "threshold":
        b = float(spec.params.get("buy_below", 0.4))
        return (
            f"Порог (threshold): покупаем YES при первом котике, где цена YES ≤ {b}; "
            "держим до резолва. Размер — 1 шар на рынок; PnL на рынок = outcome(0/1) − цена входа."
        )
    if spec.type == "momentum":
        lb = int(spec.params.get("lookback", 3))
        br = spec.params.get("buy_if_rising", True)
        return (
            f"Моментум: на каждом шаге сравниваем цену с ценой {lb} тиков назад; "
            f"покупаем YES один раз, если движение «{'вверх' if br else 'вниз'}»; 1 шар на рынок."
        )
    if spec.type == "hold":
        return "Hold: не входим в рынок, PnL = 0."
    return str(spec.model_dump())


def _analyze_batch(title: str, batch: str, short: str) -> tuple[str, dict]:
    jsonls = sorted((EXP_ROOT / batch).glob("*/iterations.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsonls:
        return f"<h2>{title}</h2><p>Нет iterations.jsonl</p>", {}

    jpath = jsonls[0]
    manifest = json.loads((jpath.parent / "manifest.json").read_text(encoding="utf-8"))
    task = manifest.get("task") or ""
    ds_path = Path(manifest["dataset_path"])

    lines = [json.loads(l) for l in jpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    iters = [r["iteration"] for r in lines]
    pnls = [float((r.get("best_in_iteration") or {}).get("total_pnl") or 0) for r in lines]

    best_pnl = None
    for r in lines:
        tp = (r.get("best_in_iteration") or {}).get("total_pnl")
        if tp is None:
            continue
        if best_pnl is None or tp > best_pnl:
            best_pnl = tp

    last_best = (lines[-1].get("best_in_iteration") or {}) if lines else {}
    strat_dict = last_best.get("strategy")
    if not strat_dict:
        body = f"<h2>{title}</h2><p>Нет стратегии в последней итерации.</p>"
        return body, {"batch": batch, "short": short, "iterations": iters, "pnls": pnls}

    spec = StrategySpec.model_validate(strat_dict)
    trades = load_trades_table(ds_path)
    strat = strategy_from_spec(spec)
    bt = run_backtest(trades, strat)
    pm = bt.per_market_pnl
    ev = evaluate_returns(pm)
    cum = np.cumsum(np.array(pm, dtype=float)).tolist()
    wins = [x for x in pm if x > 0]
    losses = [x for x in pm if x < 0]
    pf = _profit_factor(pm)

    table = f"""
<table border="1" cellpadding="8" style="border-collapse:collapse;margin:12px 0;">
<tr><td><b>total_pnl</b></td><td>{ev.total_pnl:.6f}</td></tr>
<tr><td>mean_pnl / market</td><td>{ev.mean_pnl_per_market:.6f}</td></tr>
<tr><td>std pnl / market</td><td>{ev.std_pnl_per_market:.6f}</td></tr>
<tr><td>Sharpe (mean/std)</td><td>{ev.sharpe_per_market if ev.sharpe_per_market is not None else "—"}</td></tr>
<tr><td><b>Max drawdown</b> (по кумулятивной сумме per-market)</td><td>{ev.max_drawdown:.6f}</td></tr>
<tr><td>Hit rate</td><td>{ev.hit_rate:.6f}</td></tr>
<tr><td>N markets</td><td>{ev.n_markets}</td></tr>
<tr><td>Profit factor</td><td>{("∞" if pf == float("inf") else f"{pf:.4f}") if pf is not None else "—"}</td></tr>
<tr><td>Средний win (если есть)</td><td>{float(np.mean(wins)) if wins else "—"}</td></tr>
<tr><td>Средний loss (если есть)</td><td>{float(np.mean(losses)) if losses else "—"}</td></tr>
</table>"""

    strat_json = json.dumps(strat_dict, indent=2, ensure_ascii=False)
    body = f"""
<section id="sec-{short}">
<h2>{title}</h2>
<p><b>Каталог эксперимента:</b> <code>{jpath.parent.name}</code></p>
<p><b>CSV:</b> <code>{ds_path}</code></p>
<p><b>Итераций в логе:</b> {len(lines)}</p>
<details><summary>Задача (начало текста)</summary><pre>{task[:2500]}</pre></details>
<h3>Смысл финальной стратегии (последняя итерация)</h3>
<p>{_explain_strategy(spec)}</p>
<pre>{strat_json}</pre>
<p><b>Максимум total_pnl среди итераций:</b> {best_pnl}</p>
<h3>Метрики бэктеста финальной стратегии</h3>
{table}
<h3>Кривая кумулятивного PnL по рынкам (порядок как в движке)</h3>
<canvas id="chart-eq-{short}" height="120"></canvas>
</section>
"""
    chart_payload = {
        "batch": batch,
        "short": short,
        "iterations": iters,
        "pnls": pnls,
        "equity": cum,
        "label": batch,
    }
    return body, chart_payload


def main() -> int:
    sections: list[str] = []
    payloads: list[dict] = []
    for title, batch, short in LABELS:
        sec, pl = _analyze_batch(title, batch, short)
        sections.append(sec)
        payloads.append(pl)

    iter_data = json.dumps(
        [{"label": p["label"], "x": p["iterations"], "y": p["pnls"]} for p in payloads if p],
        ensure_ascii=False,
    )
    eq_data = json.dumps({p["short"]: p["equity"] for p in payloads if p and "equity" in p}, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<title>ARPM — отчёт по трём исследованиям</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 1000px; line-height: 1.45; }}
pre {{ background: #f4f4f6; padding: 12px; overflow: auto; font-size: 12px; }}
h1 {{ border-bottom: 2px solid #333; }}
canvas {{ max-height: 320px; margin: 16px 0; }}
table td:first-child {{ white-space: nowrap; }}
</style>
</head>
<body>
<h1>Три research-прогона: результаты и метрики</h1>
<p><b>Позиция в симуляторе:</b> 1 YES-шар на рынок при входе (как в коде). Max drawdown — по кумулятивной сумме per-market PnL (как в <code>evaluate_returns</code>).</p>
<h2>Best total_pnl по номеру итерации (каждая линия — свой эксперимент)</h2>
<canvas id="iterAll"></canvas>
{"".join(sections)}
<script>
const iterRaw = {iter_data};
const eqData = {eq_data};
const colors = ['#2563eb','#16a34a','#dc2626'];
const iterData = {{
  datasets: iterRaw.map((d, i) => ({{
    label: d.label,
    data: d.x.map((xi, j) => ({{ x: xi, y: d.y[j] }})),
    borderColor: colors[i % colors.length],
    tension: 0.15,
    pointRadius: 0,
    borderWidth: 2
  }}))
}};
new Chart(document.getElementById('iterAll'), {{
  type: 'line',
  data: iterData,
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Лучший total_pnl в итерации' }} }},
    scales: {{
      x: {{ type: 'linear', title: {{ display: true, text: 'Итерация' }} }},
      y: {{ title: {{ display: true, text: 'total_pnl' }} }}
    }}
  }}
}});
Object.keys(eqData).forEach((short) => {{
  const y = eqData[short];
  const canvas = document.getElementById('chart-eq-' + short);
  if (!canvas || !y.length) return;
  const x = y.map((_, i) => i + 1);
  new Chart(canvas, {{
    type: 'line',
    data: {{
      labels: x,
      datasets: {{ label: 'Cumulative PnL', data: y, borderColor: '#111', borderWidth: 1, pointRadius: 0 }}
    }},
    options: {{
      responsive: true,
      plugins: {{ title: {{ display: true, text: 'Кумулятивный PnL по рынкам (индекс рынка в порядке бэктеста)' }} }},
      scales: {{ y: {{ title: {{ display: true, text: 'PnL' }} }} }}
    }}
  }});
}});
</script>
</body>
</html>
"""

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research_report.html"
    out.write_text(html, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
