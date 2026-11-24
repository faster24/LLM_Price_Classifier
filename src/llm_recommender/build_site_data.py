"""Build the data file for the project story page (static/story.html).

Reads only finished outputs: data/raw/ai_models.db (raw counts) and data/processed
(notebooks 00, 01, 02, 04). Writes data/processed/site_data.json.

Usage:
    uv run --with pandas python src/llm_recommender/build_site_data.py
"""
import json
import math
import os
import sqlite3

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "processed")
RAW_DB = os.path.join(ROOT, "data", "raw", "ai_models.db")
OUT_FILE = os.path.join(DATA_DIR, "site_data.json")
PRIMARY_CUTOFF, PRIMARY_MODE = "2025-12-29", "lenient"   # notebook 04 primary backtest


def read(name, **kw):
    return pd.read_csv(os.path.join(DATA_DIR, name), **kw)


def records(df):
    return json.loads(df.round(4).to_json(orient="records"))


def raw_counts():
    with sqlite3.connect(f"file:{RAW_DB}?mode=ro", uri=True) as con:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("catalog_models", "price_history", "models", "benchmark_results")}


def placeholders(catalog, tiers):
    # [Y]: share of text models whose price (input or output) changed at least once since 2025,
    # on observed weeks only (forward-filled weeks dropped; one-week spikes were already reverted).
    ph = read("clean_price_history.csv", usecols=["week_start", "provider", "model_id", "model_name", "imputed",
                                                  "input_price_per_1m_tokens", "output_price_per_1m_tokens"])
    ph = ph[(ph.week_start >= "2025-01-01") & ~ph.imputed.astype(bool)]
    ph = ph.assign(name=ph.model_name.str.lower())   # same unit as [N]: one model across providers
    changed = (ph.groupby(["provider", "model_id", "name"])[["input_price_per_1m_tokens", "output_price_per_1m_tokens"]]
                 .nunique().max(axis=1) > 1).groupby(level="name").any()
    # [X]: largest max/min blended-price ratio among tiered models in the same performance decile.
    dec = pd.qcut(tiers.performance.rank(method="first"), 10, labels=False)
    ratio = tiers.groupby(dec).blended_price.agg(lambda p: p.max() / p.min())
    return {
        "N_text_models": int(catalog.model_name.str.lower().nunique()),
        "M_providers": int(catalog.provider.nunique()),
        "X_price_gap": round(float(ratio.max()), 1),
        "Y_pct_price_changed": round(100 * float(changed.mean()), 1),
        "Y_n_models": int(len(changed)),
    }


def eda():
    """Small aggregates for the Data Understanding / Preparation charts."""
    m = read("clean_models.csv")
    p = m[m.has_valid_price].copy()
    p["blended"] = (3 * p.median_input_price_per_1m + p.median_output_price_per_1m) / 4
    p = p[p.blended > 0]
    edges = [x / 4 for x in range(-12, 11)]                  # log10 price bins, $0.001 .. $316
    p["bin"] = pd.cut(p.blended.apply(math.log10), edges, labels=edges[:-1])
    hist = p.groupby(["bin", "open_weights"], observed=False).size().unstack(fill_value=0)

    ph = read("clean_price_history.csv", usecols=["week_start", "input_price_per_1m_tokens",
                                                  "output_price_per_1m_tokens", "spike_fixed", "extreme_jump",
                                                  "imputed", "provider", "model_id", "eligible_for_trend"])
    ph["blended"] = (3 * ph.input_price_per_1m_tokens + ph.output_price_per_1m_tokens) / 4
    weekly = ph.groupby("week_start").blended.agg(["median", "size"]).reset_index()
    series = ph.groupby(["provider", "model_id"]).eligible_for_trend.first()

    b = read("clean_benchmarks.csv")
    b = b[b.higher_is_better & b.model_id.isin(p.model_id)]
    b = b.assign(test=b.benchmark_clean + " [" + b.metric_clean + "]")
    cover = b.groupby("test").model_id.nunique()
    good = b[b.test.isin(cover[cover >= 5].index)]           # same rule as notebook 01
    fill = len(good.drop_duplicates(["model_id", "test"])) / (good.model_id.nunique() * good.test.nunique())

    q = pd.to_datetime(m.release_date).dt.to_period("Q").astype(str)
    rel = m.groupby([q, "open_weights"]).size().unstack(fill_value=0).sort_index()
    return {
        "price_hist": {"log10_lo": [float(x) for x in hist.index], "closed": hist.get(0, 0).tolist(),
                       "open": hist.get(1, 0).tolist()},
        "price_io": records(p[["model_name", "open_weights", "median_input_price_per_1m",
                               "median_output_price_per_1m"]].rename(columns={
            "median_input_price_per_1m": "inp", "median_output_price_per_1m": "out"})),
        "weekly": records(weekly.rename(columns={"size": "n_series"})),
        "coverage": records(cover.sort_values(ascending=False).head(15).rename("n_models").reset_index()),
        "coverage_stats": {"tests": int(good.test.nunique()), "models": int(good.model_id.nunique()),
                           "fill": round(fill, 4), "all_tests": int(len(cover))},
        "releases": {"quarter": rel.index.tolist(), "closed": rel.get(0, 0).tolist(), "open": rel.get(1, 0).tolist()},
        "labs": records(m.lab.value_counts().head(10).rename("n").rename_axis("lab").reset_index()),
        "fixes": {"spike_fixed": int(ph.spike_fixed.sum()), "extreme_jump": int(ph.extreme_jump.sum()),
                  "imputed": int(ph.imputed.sum()), "rows": len(ph),
                  "eligible": int(series.sum()), "series": len(series)},
    }


def main():
    catalog = read("clean_catalog.csv", usecols=["model_name", "provider"])
    models = read("clean_models.csv", usecols=["model_id", "has_valid_price"])
    tiers = read("rq1_tiers.csv")
    profiles = read("rq1_tier_profiles.csv").sort_values("tier_id")
    cv = read("rq2_cv_results.csv")
    imp = read("rq2_feature_importance.csv").sort_values("shap_rank")
    summ = read("rq4_summary.csv")
    mapping = read("rq4_mapping.csv")
    val = read("validation_report.csv")
    bt = read("rq4_backtest_results.csv", usecols=["cutoff", "mode", "budget", "method", "achieved"])

    assert len(tiers) == 144 and tiers.tier_id.nunique() == 7, "RQ1 output changed; re-check story text"
    rq3 = summ[(summ.cutoff == PRIMARY_CUTOFF) & (summ["mode"] == PRIMARY_MODE)]
    assert len(rq3), "primary backtest setting missing from rq4_summary.csv"

    data = {
        "counts": {
            "raw": raw_counts() if os.path.exists(RAW_DB) else None,
            "clean_models": len(models),
            "priced_models": int(models.has_valid_price.sum()),
            "clean_benchmarks": len(read("clean_benchmarks.csv", usecols=[0])),
            "clean_catalog": len(catalog),
            "clean_price_history": len(read("clean_price_history.csv", usecols=[0])),
            "tiered_models": len(tiers),
            "validation_checks": len(val),
            "validation_failed": int((~val.passed.astype(bool)).sum()),
            "linked": int(mapping.matched.astype(bool).sum()),
            "link_pool": len(mapping),
        },
        "placeholders": placeholders(catalog, tiers),
        "tiers": records(tiers[["model_id", "model_name", "lab", "open_weights", "blended_price",
                                "performance", "n_tests", "tier_id", "tier"]]),
        "tier_profiles": records(profiles[["tier_id", "tier", "n", "median_price", "price_q1", "price_q3",
                                           "median_perf", "perf_q1", "perf_q3",
                                           "open_weights_share", "top_labs", "examples"]]),
        "rq2_cv": records(cv[["run", "section", "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd",
                              "beats_majority_folds"]]),
        "rq2_importance": records(imp[["feature", "group", "mean_abs_shap", "shap_rank",
                                       "perm_f1_drop", "perm_rank"]]),
        "rq3_backtest": {"cutoff": PRIMARY_CUTOFF, "mode": PRIMARY_MODE,
                         "rows": records(rq3[["task", "method", "n", "mean_achieved", "fail_rate",
                                              "over_budget", "unavailable"]])},
        "eda": eda(),
        "validation": records(val),
        "rq1_k": records(read("rq1_k_selection.csv")),
        "rq1_kruskal": records(read("rq1_kruskal.csv")),
        "rq2_confusion": records(read("rq2_confusion.csv")),
        "rq2_per_tier": records(read("rq2_per_tier.csv")),
        "rq2_perm": records(read("rq2_permutation.csv")),
        "rq3_all": records(summ[["cutoff", "mode", "task", "method", "n", "mean_achieved"]]),
        "rq3_budget": records(bt[(bt.cutoff == PRIMARY_CUTOFF) & (bt["mode"] == PRIMARY_MODE)]
                              .groupby(["budget", "method"]).achieved.mean().reset_index()),
    }
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1024:.0f} KB)", data["placeholders"])


if __name__ == "__main__":
    main()
