from fastapi import FastAPI
from app.api import health

app = FastAPI(title="Next Word Predictor API")
app.include_router(health.router)