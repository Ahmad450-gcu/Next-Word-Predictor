import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import tensorflow as tf
from app.core.config import settings
from src.model.tied_lstm import TiedLSTMLanguageModel
from src.preprocessing.clean_text import normalize_unicode, normalize_artifacts, mask_numbers

def _resolve_paths():
    if settings.use_hf_hub:
        from huggingface_hub import hf_hub_download
        print(f"Downloading model + tokenizer from {settings.hf_repo_id}@{settings.hf_revision}...")
        model_path = hf_hub_download(repo_id=settings.hf_repo_id, filename="best_model_tied.keras", revision=settings.hf_revision)
        tokenizer_path = hf_hub_download(repo_id=settings.hf_repo_id, filename="tokenizer_word_index.json", revision=settings.hf_revision)
        return model_path, tokenizer_path
    else:
        return settings.model_path, settings.tokenizer_path

model_path, tokenizer_path = _resolve_paths()

print("Loading tokenizer...")
with open(tokenizer_path) as f:
    WORD_INDEX = json.load(f)
INDEX_WORD = {i: w for w, i in WORD_INDEX.items()}
OOV_INDEX = WORD_INDEX["<UNK>"]
SPECIAL_TOKENS = {"<UNK>", "<unk>", "<num>", "<NUM>"}
SPECIAL_INDICES = np.array(
    sorted({WORD_INDEX[t] for t in SPECIAL_TOKENS if t in WORD_INDEX}),
    dtype=np.int64,
)

print("Loading model...")
MODEL = tf.keras.models.load_model(
    model_path,
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

    if SPECIAL_INDICES.size:
        probs = probs.copy()
        probs[SPECIAL_INDICES] = -np.inf

    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = [
        {"word": INDEX_WORD.get(int(idx), "<UNK>"), "probability": float(probs[idx])}
        for idx in top_indices
    ]
    return predictions