from fastapi import APIRouter
from app.schemas.predict import PredictRequest, PredictResponse
from app.services.inference import predict_next_words

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    predictions = predict_next_words(request.text, top_k=request.top_k)
    return PredictResponse(predictions=predictions)