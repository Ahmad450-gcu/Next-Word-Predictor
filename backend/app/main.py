from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, predict
from app.core.config import settings

app = FastAPI(title="Next Word Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://next-word-predictor-iota.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)