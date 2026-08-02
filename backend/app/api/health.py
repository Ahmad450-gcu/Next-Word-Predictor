from fastapi import APIRouter
from app.services.inference import MODEL, WORD_INDEX

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "vocab_size": len(WORD_INDEX),
    }