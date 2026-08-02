from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to predict the next word for")
    top_k: int = Field(5, ge=1, le=20, description="Number of predictions to return")

class Prediction(BaseModel):
    word: str
    probability: float

class PredictResponse(BaseModel):
    predictions: list[Prediction]