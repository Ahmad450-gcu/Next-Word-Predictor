from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[3]
    model_path: Path = project_root / "models" / "best_model_tied.keras"
    tokenizer_path: Path = project_root / "data" / "processed" / "tokenizer_word_index.json"
    sequence_length: int = 50
    default_top_k: int = 5

    model_config = SettingsConfigDict(env_prefix="NWP_")

settings = Settings()