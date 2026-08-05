# Backend

FastAPI service that serves next-word predictions from the tied-weight LSTM model. Deployed on Railway.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

By default this loads the model and tokenizer from local files (`../models/best_model_tied.keras`, `../data/processed/tokenizer_word_index.json`, relative to the repo root) — run `dvc repro` or `dvc pull` first so those files exist.

## Configuration

All settings are environment variables with an `NWP_` prefix (see `app/core/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `NWP_MODEL_PATH` | `models/best_model_tied.keras` | Local model path (used when `NWP_USE_HF_HUB=false`) |
| `NWP_TOKENIZER_PATH` | `data/processed/tokenizer_word_index.json` | Local tokenizer path (used when `NWP_USE_HF_HUB=false`) |
| `NWP_SEQUENCE_LENGTH` | `50` | Context window length the model expects |
| `NWP_DEFAULT_TOP_K` | `5` | Default number of predictions returned |
| `NWP_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed origins |
| `NWP_USE_HF_HUB` | `false` | If `true`, downloads model + tokenizer from Hugging Face Hub at startup instead of reading local files |
| `NWP_HF_REPO_ID` | `dev-Ahmad450/next-word-predictor` | HF Hub repo to download from |
| `NWP_HF_REVISION` | `v1.0` | HF Hub revision/tag to download. **Production (Railway) overrides this** to whichever tag currently holds the `champion` alias in the MLflow registry — see the root README's retrain/promote pipeline. |

Production (Railway) runs with `NWP_USE_HF_HUB=true` and a real `NWP_HF_REVISION` tag.

## API

### `GET /health`
Returns service status, whether the model loaded successfully, and vocab size.
```json
{ "status": "ok", "model_loaded": true, "vocab_size": 33280 }
```

### `POST /predict`
```json
{ "text": "the game was released in", "top_k": 5 }
```
`top_k` is optional (default 5, max 20). Returns the top-k next-word predictions with probabilities:
```json
{
  "predictions": [
    { "word": "the", "probability": 0.12 },
    { "word": "1996", "probability": 0.08 }
  ]
}
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

Tests load the model/tokenizer from local files, so run `dvc pull` (or have them present from a prior `dvc repro`) before testing. CI (`backend-ci.yml`) fetches these directly from HF Hub instead of using DVC — see the root README.

## Deployment

Built via the root `Dockerfile` (not a Dockerfile inside this directory — the image needs both `backend/` and `src/`). Listens on `$PORT` (Railway sets this) or `7860` by default.
