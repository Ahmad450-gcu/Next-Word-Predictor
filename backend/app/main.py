from fastapi import FastAPI
from app.api import health, predict

app = FastAPI(title="Next Word Predictor API")
app.include_router(health.router)
app.include_router(predict.router)