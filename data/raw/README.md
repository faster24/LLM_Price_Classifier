# Raw data

Raw files are not committed. Download both Kaggle datasets (same author) and place the files here:

| File | Dataset |
|---|---|
| `ai_models_2026.csv`, `ai_model_price_history.csv` | [AI Models Pricing & Price History 2023–2026](https://www.kaggle.com/datasets/abirambharathisa/ai-models-and-api-pricing-2026-price-history) |
| `llm_models.csv`, `llm_benchmarks.csv`, `llm_provider_offerings.csv` | [LLM Benchmarks & Specs 2026](https://www.kaggle.com/datasets/abirambharathisa/llm-benchmarks-and-specs-2026-448-ai-models) |
| `ai_models.db` | SQLite database holding all five CSVs as tables (`catalog_models`, `price_history`, `models`, `benchmark_results`, `provider_offerings`, plus `labs`/`providers` lookups) |

These files are read-only inputs. `notebooks/00_data_validation_cleaning.ipynb` reads `ai_models.db` (and the CSVs for a parity check) and writes the clean files to `data/processed/`.
