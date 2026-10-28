"""Recommendation engine (pure Python, no pandas). Rules of notebook 04, Step 6.

1. capability: model has a score for the task; vision needs image input; optional tools / JSON
2. requirements: min context, open-weight only, include / exclude labs, min score, released within N months
3. budget: cheapest-provider monthly cost <= budget
4. rank: rank_by="score" (best score within budget, cheaper first on ties) or
         rank_by="value" (score / cost, the proposal's rule tested in RQ4)
5. trend: falling x1.25, rising x0.75 on value; rising models using >= 90% of the budget are dropped
6. return the top N with tier and trend
"""
import json
from datetime import date

TREND_ALPHA = 0.25
RISING_DROP_SHARE = 0.90
VALUE_WARNING = ("rank_by='value' is the proposal's score / cost rule. In the RQ4 backtest it behaved like "
                 "'pick the cheapest' and did not beat the baselines (H4 not supported).")
SCORE_NOTE = ("rank_by='score' (best score within budget) beat both baselines in the RQ4 backtest, "
              "but this was an exploratory finding that is not yet confirmed.")


def load_snapshot(path):
    with open(path) as f:
        return json.load(f)


def monthly_cost(model, input_m, output_m):
    """(cost, provider) at the cheapest provider for this usage; (None, None) if no offer."""
    best = None
    for o in model["offers"]:
        c = input_m * o["input_price_per_1m"] + output_m * o["output_price_per_1m"]
        if best is None or c < best[0]:
            best = (c, o["provider"])
    return best or (None, None)


def _months_before(iso_day, months):
    y, m, d = map(int, iso_day.split("-"))
    m -= months
    while m <= 0:
        m += 12; y -= 1
    return date(y, m, min(d, 28)).isoformat()


def recommend(snapshot, task, input_m, output_m, budget, requirements=None,
              rank_by="score", use_trend=True, top_n=3):
    """Return a dict with the top-n recommendations and funnel counts."""
    if task not in snapshot["meta"]["task_benchmarks"]:
        raise ValueError(f"unknown task {task!r}")
    if rank_by not in ("score", "value"):
        raise ValueError("rank_by must be 'score' or 'value'")
    req = requirements or {}
    as_of = snapshot["meta"]["as_of_week"]
    oldest = _months_before(as_of, req["released_within_months"]) if req.get("released_within_months") else None
    include = {l.lower() for l in req.get("include_labs") or []}
    exclude = {l.lower() for l in req.get("exclude_labs") or []}

    capable, meets, fits = [], [], []
    for m in snapshot["models"]:
        score = m["scores"].get(task)
        if score is None or (task == "vision" and not m["input_image"]):
            continue
        if (req.get("needs_tools") and not m["tool_call"]) or (req.get("needs_json") and not m["json_ok"]):
            continue
        capable.append(m)                                                  # step 1
        if req.get("open_only") and not m["open_weights"]:
            continue
        if req.get("min_context") and (m["context_window"] or 0) < req["min_context"]:
            continue
        if include and m["lab"].lower() not in include:
            continue
        if m["lab"].lower() in exclude:
            continue
        if req.get("min_score") is not None and score < req["min_score"]:
            continue
        if oldest and m["release_date"] < oldest:
            continue
        meets.append(m)                                                    # step 2
        cost, provider = monthly_cost(m, input_m, output_m)
        if cost is None or cost > budget:
            continue                                                       # step 3
        value = score / cost if cost > 0 else float("inf")                 # step 4
        adj = value
        if use_trend:                                                      # step 5
            if m["trend"] == "rising" and cost >= RISING_DROP_SHARE * budget:
                continue
            adj *= {"falling": 1 + TREND_ALPHA, "rising": 1 - TREND_ALPHA}.get(m["trend"], 1.0)
        fits.append((m, score, cost, provider, value, adj))

    if rank_by == "value":
        fits.sort(key=lambda x: (-x[5], -x[1]))
    else:
        fits.sort(key=lambda x: (-x[1], x[2]))
    recs = [{"rank": i + 1, "model_id": m["model_id"], "model_name": m["model_name"], "lab": m["lab"],
             "tier": m["tier"], "trend": m["trend"], "task_score": round(s, 4),
             "monthly_cost_usd": round(c, 4), "cheapest_provider": p, "budget_used": round(c / budget, 4),
             "value": round(v, 6)}
            for i, (m, s, c, p, v, _) in enumerate(fits[:top_n])]                       # step 6
    return {"as_of_week": as_of, "task": task, "rank_by": rank_by, "use_trend": use_trend,
            "trend_source": snapshot["meta"]["trend_source"],
            "note": VALUE_WARNING if rank_by == "value" else SCORE_NOTE,
            "funnel": {"capable": len(capable), "meet_requirements": len(meets), "within_budget": len(fits)},
            "recommendations": recs}
