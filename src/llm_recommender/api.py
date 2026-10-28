"""REST API for the LLM model recommender.

Run (from the project root):
    uv run --with fastapi,uvicorn uvicorn llm_recommender.api:app --app-dir src --port 8000
Then open http://localhost:8000/docs for the interactive docs.

The snapshot file defaults to data/processed/recommender_snapshot.json; set
LLM_RECOMMENDER_SNAPSHOT to use another one (e.g. when deploying elsewhere).
"""
import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

try:
    from . import engine
except ImportError:                     # run as a plain module
    import engine

DEFAULT_SNAPSHOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed",
                                "recommender_snapshot.json")
SNAPSHOT_PATH = os.environ.get("LLM_RECOMMENDER_SNAPSHOT", DEFAULT_SNAPSHOT)
SNAPSHOT = engine.load_snapshot(SNAPSHOT_PATH)
TASKS = tuple(SNAPSHOT["meta"]["task_benchmarks"])
INDEX_HTML = os.path.join(os.path.dirname(__file__), "static", "index.html")

app = FastAPI(
    title="LLM Model Recommender",
    version=SNAPSHOT["meta"]["snapshot_version"],
    description="Recommends LLM API models for a task, requirements and monthly budget "
                "(ICT324 Team Straw Hats, RQ4). Prices: cheapest provider, USD per 1M tokens. "
                "Scores: within-test percentiles (0–1). See GET /meta for the data snapshot.",
)


class Requirements(BaseModel):
    needs_tools: bool = False
    needs_json: bool = False
    open_only: bool = Field(False, description="open-weight models only")
    min_context: Optional[int] = Field(None, ge=0, description="minimum context window (tokens)")
    include_labs: Optional[List[str]] = Field(None, description="only these labs, e.g. ['google']")
    exclude_labs: Optional[List[str]] = Field(None, description="never these labs")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="minimum task score (0–1)")
    released_within_months: Optional[int] = Field(None, ge=1)


class RecommendRequest(BaseModel):
    task: Literal["coding", "reasoning", "vision"]
    monthly_input_tokens_m: float = Field(..., gt=0, description="input tokens per month, in millions")
    monthly_output_tokens_m: float = Field(..., ge=0, description="output tokens per month, in millions")
    monthly_budget_usd: float = Field(..., gt=0)
    requirements: Requirements = Requirements()
    rank_by: Literal["score", "value"] = Field(
        "score", description="'score' = best score within budget (default); "
                             "'value' = score / cost, the proposal's rule (did not beat baselines in RQ4)")
    use_trend: bool = True
    top_n: int = Field(3, ge=1, le=10)

    model_config = {"json_schema_extra": {"examples": [{
        "task": "coding", "monthly_input_tokens_m": 100, "monthly_output_tokens_m": 20,
        "monthly_budget_usd": 200, "requirements": {"open_only": True}, "rank_by": "score"}]}}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {"status": "ok", "snapshot": SNAPSHOT["meta"]["snapshot_version"]}


@app.get("/meta")
def meta():
    return SNAPSHOT["meta"]


@app.get("/models")
def list_models(task: Optional[Literal["coding", "reasoning", "vision"]] = Query(None)):
    rows = [m for m in SNAPSHOT["models"] if task is None or task in m["scores"]]
    return [{k: v for k, v in m.items() if k != "offers"} | {"n_offers": len(m["offers"])} for m in rows]


@app.get("/models/{model_id:path}")
def get_model(model_id: str):
    for m in SNAPSHOT["models"]:
        if m["model_id"] == model_id:
            return m
    raise HTTPException(404, f"model {model_id!r} not in snapshot {SNAPSHOT['meta']['snapshot_version']}")


@app.post("/recommend")
def recommend(req: RecommendRequest):
    return engine.recommend(
        SNAPSHOT, req.task, req.monthly_input_tokens_m, req.monthly_output_tokens_m,
        req.monthly_budget_usd, req.requirements.model_dump(), req.rank_by, req.use_trend, req.top_n)
