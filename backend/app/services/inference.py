import json
import numpy as np
import tensorflow as tf
from app.core.config import settings
from src.model.tied_lstm import TiedLSTMLanguageModel
from src.preprocessing.clean_text import normalize_unicode, normalize_artifacts, mask_numbers


print("Loading tokenizer...")
with open(settings.tokenizer_path) as f:
    WORD_INDEX = json.load(f)
INDEX_WORD = {i: w for w, i in WORD_INDEX.items()}
OOV_INDEX = WORD_INDEX["<UNK>"]

print("Loading model...")
MODEL = tf.keras.models.load_model(
    settings.model_path,
    custom_objects={"TiedLSTMLanguageModel": TiedLSTMLanguageModel},
)
print("Model loaded.")

def _normalize_input(text: str) -> str:
    text = normalize_unicode(text)
    text = normalize_artifacts(text)
    text = mask_numbers(text)
    return text.lower()


def predict_next_words(text: str, top_k: int = 5):
    normalized = _normalize_input(text)
    tokens = normalized.split()
    if not tokens:
        return []
    context = tokens[-settings.sequence_length:]
    token_ids = [WORD_INDEX.get(tok, OOV_INDEX) for tok in context]
    input_tensor = tf.constant([token_ids], dtype=tf.int32)
    probs = MODEL(input_tensor, training=False)[0].numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = [
        {"word": INDEX_WORD.get(int(idx), "<UNK>"), "probability": float(probs[idx])}
        for idx in top_indices
    ]
    return predictions