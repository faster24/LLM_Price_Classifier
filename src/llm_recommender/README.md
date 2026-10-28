# LLM Model Recommender (RQ4 engine as a REST API)

Recommends LLM API models for a task, requirements and monthly budget. It uses the decision rules of `notebooks/04_RQ4_model_selection.ipynb`.

| File | Role |
|---|---|
| `build_snapshot.py` | Builds `data/processed/recommender_snapshot.json` from the clean data (pandas). Re-run it whenever the data or `rq3_trends.csv` changes. |
| `engine.py` | Filter-then-rank rules in plain Python, with no pandas needed at runtime. |
| `api.py` | FastAPI service. |

Live: https://llm-price-classifier.vercel.app

## Run

```bash
# 1. build the snapshot (default: latest week in the price history)
uv run --with pandas python src/llm_recommender/build_snapshot.py

# 2. start the API
uv run --with fastapi,uvicorn uvicorn llm_recommender.api:app --app-dir src --port 8000
# demo page: http://localhost:8000/   interactive docs: http://localhost:8000/docs
```

The demo page is `static/index.html`, served at `GET /`. For Vercel, the root `app.py` imports this app (see the main README). To deploy elsewhere, copy `src/llm_recommender/` and the snapshot JSON, then set `LLM_RECOMMENDER_SNAPSHOT=/path/to/recommender_snapshot.json`.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/` | demo web page |
| GET | `/health` | status and snapshot version |
| GET | `/meta` | snapshot week, price/score rules, task → benchmark mapping, trend source |
| GET | `/models?task=coding` | models in the snapshot (without per-provider offers) |
| GET | `/models/{model_id}` | one model, including every provider's prices |
| POST | `/recommend` | top-N recommendations |

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' -d '{
  "task": "coding",
  "monthly_input_tokens_m": 100, "monthly_output_tokens_m": 20,
  "monthly_budget_usd": 200,
  "requirements": {"open_only": true},
  "rank_by": "score"
}'
```

Request fields:
- `task`: `coding`, `reasoning` or `vision`
- token volumes in **millions per month**, and the budget in **USD per month**
- `requirements`: `needs_tools`, `needs_json`, `open_only`, `min_context`, `include_labs`, `exclude_labs`, `min_score` (0–1), `released_within_months`
- `rank_by`, `use_trend`, `top_n` (1–10), `offset` (skip that many ranked models; the response's `has_more` says whether more remain)

The response lists each pick's task score, monthly cost, cheapest provider, share of budget used, RQ1 tier and price trend. It also includes a `funnel` showing how many models passed each filter.

## Ranking modes: read before using

- **`score` (default):** the best task score that fits the budget, with the cheaper model first on ties. In the RQ4 backtest this beat "cheapest" and "top benchmark", but that was an **exploratory** finding that is not yet confirmed by a pre-registered test.
- **`value`:** score ÷ cost, the proposal's rule. In the backtest it behaved like "pick the cheapest" and **did not** beat the baselines (H4 not supported). Use it for comparison.

## Data caveats

- Scores are percentiles **within the snapshot's model pool** (0 = worst, 1 = best on that task). They are not raw benchmark numbers.
- Prices come from the cheapest provider. Free tiers are excluded.
- Trends currently come from a 12-week momentum **placeholder**. Once notebook 03 writes `rq3_trends.csv`, rebuild the snapshot to use RQ3's classifier.
- The snapshot is a point-in-time copy. Results get stale as prices change, so rebuild it when the data is refreshed.
