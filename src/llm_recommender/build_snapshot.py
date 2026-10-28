"""Build the recommender snapshot (JSON) from data/processed.

Same rules as notebooks/04_RQ4_model_selection.ipynb:
- benchmark models are linked to price-history series by normalised name
- prices come from every provider with a non-zero price in the snapshot week
- task score = mean within-test percentile (tests covering >= 5 models in the pool)
- trend = rq3_trends.csv if present, else the 12-week momentum placeholder

Usage:
    uv run --with pandas python src/llm_recommender/build_snapshot.py [--week 2026-09-28]
"""
import argparse
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "processed")
OUT_FILE = os.path.join(DATA_DIR, "recommender_snapshot.json")

TASK_BENCH = {
    "coding": ["SWE-Bench Pro", "SWE-Bench Verified", "Aider Polyglot", "Terminal-Bench",
               "Terminal-Bench Hard", "Terminal Bench 2.0", "SciCode", "LiveCodeBench"],
    "reasoning": ["Humanity's Last Exam", "GPQA Diamond", "AIME"],
    "vision": ["MMMU Pro", "MathVision"],
}
MIN_MODELS_PER_TEST = 5
TREND_LOOKBACK_WEEKS = 12
TREND_THRESHOLD = 0.05
ALIASES = {}   # {benchmark model_id: price key}; see notebook 04 Step 2


def norm_key(s):
    s = str(s).lower().split("/")[-1]
    s = re.sub(r"^(apac|us|eu|au|jp|global)\.", "", s)
    s = re.sub(r"^(anthropic|meta|mistral|amazon|cohere|ai21|deepseek|qwen|openai|google|"
               r"moonshotai|minimax|zai|xai)\.", "", s)
    s = re.sub(r":.*$", "", s); s = re.sub(r"-v\d+$", "", s); s = re.sub(r"-20\d{6}$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def load():
    read = lambda f: pd.read_csv(os.path.join(DATA_DIR, f))
    return (read("clean_models.csv"), read("clean_benchmarks.csv"),
            read("clean_price_history.csv"), read("clean_catalog.csv"), read("rq1_tiers.csv"))


def link_models(models, bench, hist):
    """Return hist with a bm_model_id column (benchmark-side model id or NaN)."""
    cand = models[models.has_valid_price & models.model_id.isin(bench.model_id)]
    hist = hist.copy()
    hist["key_id"], hist["key_name"] = hist.model_id.map(norm_key), hist.model_name.map(norm_key)
    hist_keys = set(hist.key_id) | set(hist.key_name)
    key2model = {}
    for r in cand.itertuples():
        for k in {ALIASES.get(r.model_id, norm_key(r.model_id)), norm_key(r.model_name)}:
            if k in hist_keys:
                key2model.setdefault(k, set()).add(r.model_id)
    clash = {k: v for k, v in key2model.items() if len(v) > 1}
    assert not clash, f"one price key maps to several models: {clash}"
    key2model = {k: next(iter(v)) for k, v in key2model.items()}
    hist["bm_model_id"] = hist.key_id.map(key2model).fillna(hist.key_name.map(key2model))
    return hist[hist.bm_model_id.notna()], key2model, len(cand)


def blended_min(h, week):
    w = h[h.week_start == week]
    return ((3 * w.input_price_per_1m_tokens + w.output_price_per_1m_tokens) / 4).groupby(w.bm_model_id).min()


def task_scores(bench, pool, task, week):
    b = bench[bench.higher_is_better & bench.benchmark_clean.isin(TASK_BENCH[task]) & bench.model_id.isin(pool)].copy()
    b["test"] = b.benchmark_clean + " [" + b.metric_clean + "]"
    cover = b.groupby("test").model_id.nunique()
    b = b[b.test.isin(cover[cover >= MIN_MODELS_PER_TEST].index)]
    if b.empty:
        return pd.Series(dtype=float)
    r = b.groupby("test").score.rank(method="average"); n = b.groupby("test").score.transform("size")
    return ((r - 1) / (n - 1)).groupby(b.model_id).mean()


def trends(h, week):
    path = os.path.join(DATA_DIR, "rq3_trends.csv")
    if os.path.exists(path):
        t = pd.read_csv(path)
        return t[t.as_of_week == week].set_index("model_id").trend, "RQ3 classifier (rq3_trends.csv)"
    past = (pd.Timestamp(week) - pd.Timedelta(weeks=TREND_LOOKBACK_WEEKS)).strftime("%Y-%m-%d")
    now, before = blended_min(h, week), blended_min(h, past)
    chg = (now / before.reindex(now.index) - 1).fillna(0.0)
    lab = chg.map(lambda c: "falling" if c <= -TREND_THRESHOLD else "rising" if c >= TREND_THRESHOLD else "stable")
    return lab, "placeholder: 12-week price momentum (RQ3 not available yet)"


def build(week=None):
    models, bench, hist, catalog, tiers = load()
    week = week or hist.week_start.max()
    assert week in set(hist.week_start), f"week {week} not in price history"
    h, key2model, n_cand = link_models(models, bench, hist)
    h = h[(h.input_price_per_1m_tokens > 0) & (h.output_price_per_1m_tokens > 0)]

    info = models.set_index("model_id")
    available = set(h.loc[h.week_start == week, "bm_model_id"])
    pool = set(info.index[info.release_date <= week]) & available
    scores = {t: task_scores(bench, pool, t, week) for t in TASK_BENCH}
    trend, trend_source = trends(h, week)

    cat = catalog.assign(bm=catalog.model_id.map(norm_key).map(key2model))
    json_cat = cat.groupby("bm").supports_response_schema.max()
    tier_of = tiers.set_index("model_id").tier
    offers = (h[h.week_start == week]
              .groupby(["bm_model_id", "provider"])[["input_price_per_1m_tokens", "output_price_per_1m_tokens"]]
              .min().reset_index())

    out = []
    for mid in sorted(pool):
        sc = {t: round(float(s[mid]), 4) for t, s in scores.items() if mid in s.index}
        if not sc:
            continue                      # no task score → cannot be ranked
        r = info.loc[mid]
        off = offers[offers.bm_model_id == mid]
        out.append({
            "model_id": mid, "model_name": r.model_name, "lab": r.lab,
            "open_weights": bool(r.open_weights), "release_date": r.release_date,
            "context_window": None if pd.isna(r.context_window) else int(r.context_window),
            "tool_call": bool(r.tool_call), "input_image": bool(r.input_image),
            "json_ok": bool((r.structured_output == 1) or json_cat.get(mid, 0) == 1),
            "tier": tier_of.get(mid, "untiered"), "trend": trend.get(mid, "stable"),
            "scores": sc,
            "offers": [{"provider": o.provider, "input_price_per_1m": float(o.input_price_per_1m_tokens),
                        "output_price_per_1m": float(o.output_price_per_1m_tokens)} for o in off.itertuples()],
        })
    meta = {
        "snapshot_version": week, "as_of_week": week,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "data/processed (notebooks 00, 01; rules of notebook 04)",
        "price_rule": "cheapest provider per request; free / zero-price offers excluded; USD per 1M tokens",
        "score_rule": "mean within-test percentile (0 = worst, 1 = best) among models available at as_of_week",
        "task_benchmarks": TASK_BENCH, "trend_source": trend_source,
        "n_models": len(out), "n_linked_of_candidates": f"{len(set(key2model.values()))}/{n_cand}",
    }
    return {"meta": meta, "models": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--week", help="snapshot week (YYYY-MM-DD Monday); default = latest")
    ap.add_argument("--out", default=OUT_FILE)
    a = ap.parse_args()
    snap = build(a.week)
    with open(a.out, "w") as f:
        json.dump(snap, f, indent=1)
    m = snap["meta"]
    print(f"wrote {a.out}: {m['n_models']} models as of {m['as_of_week']} | trend: {m['trend_source']}")
