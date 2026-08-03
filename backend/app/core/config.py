from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parents[3]
    model_path: Path = project_root / "models" / "best_model_tied.keras"
    tokenizer_path: Path = project_root / "data" / "processed" / "tokenizer_word_index.json"
    sequence_length: int = 50
    default_top_k: int = 5
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    use_hf_hub: bool = False
    hf_repo_id: str = "dev-Ahmad450/next-word-predictor"
    hf_revision: str = "v1.0"

    model_config = SettingsConfigDict(env_prefix="NWP_")

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

settings = Settings()