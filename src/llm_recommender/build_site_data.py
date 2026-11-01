"""Build the data file for the project story page (static/story.html).

Reads only finished outputs: data/raw/ai_models.db (raw counts) and data/processed
(notebooks 00, 01, 02, 04). Writes data/processed/site_data.json.

Usage:
    uv run --with pandas python src/llm_recommender/build_site_data.py
"""
import json
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
        "tier_profiles": records(profiles[["tier_id", "tier", "n", "median_price", "median_perf",
                                           "open_weights_share", "top_labs", "examples"]]),
        "rq2_cv": records(cv[["run", "section", "accuracy_mean", "accuracy_sd", "macro_f1_mean", "macro_f1_sd"]]),
        "rq2_importance": records(imp[["feature", "group", "mean_abs_shap", "shap_rank",
                                       "perm_f1_drop", "perm_rank"]]),
        "rq3_backtest": {"cutoff": PRIMARY_CUTOFF, "mode": PRIMARY_MODE,
                         "rows": records(rq3[["task", "method", "n", "mean_achieved", "fail_rate"]])},
    }
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1024:.0f} KB)", data["placeholders"])


if __name__ == "__main__":
    main()
