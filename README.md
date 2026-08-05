---
title: Next Word Predictor
emoji: 📝
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Next Word Predictor

A next-word prediction language model trained on WikiText-2, served via FastAPI and a React frontend.

## Architecture

- **Model**: 2-layer tied-weight LSTM (weight tying between the embedding and output layers), trained on WikiText-2. Test perplexity ≈ 124.24, top-1 accuracy ≈ 24.6%, top-5 accuracy ≈ 45.2%.
- **Data pipeline**: DVC-managed, five stages (`preprocess` → `tokenize` → `build_sequences` → `train` → `evaluate`) — see `dvc.yaml` and `params.yaml`.
- **Backend**: FastAPI (`backend/`) — serves `/predict` and `/health`. Model + tokenizer are versioned on [Hugging Face Hub](https://huggingface.co/dev-Ahmad450/next-word-predictor) and downloaded at startup in production.
- **Frontend**: React + Vite (`frontend/`).

## Project structure

```
src/ core ML code (preprocessing, model definition) — shared by the pipeline and the backend
backend/ FastAPI app
frontend/ React + Vite app
data/, models/ DVC-tracked (not in git directly — see .dvc files)
params.yaml all pipeline hyperparameters
dvc.yaml pipeline stage definitions
```

## Running locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Reproducing the pipeline

```bash
dvc repro
```

See `README` inside `backend/` and `frontend/` for more (if present), and `dvc.yaml` for the full stage graph.

Testing GitHub Actions CI pipeline.