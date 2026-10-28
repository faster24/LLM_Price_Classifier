"""Vercel entrypoint: exposes the recommender FastAPI app from src/llm_recommender."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llm_recommender.api import app  # noqa: E402,F401
