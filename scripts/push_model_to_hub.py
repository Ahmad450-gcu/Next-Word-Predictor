from huggingface_hub import HfApi

api = HfApi()
repo_id = "dev-Ahmad450/next-word-predictor"

api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

api.upload_file(
    path_or_fileobj="models/best_model_tied.keras",
    path_in_repo="best_model_tied.keras",
    repo_id=repo_id,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="data/processed/tokenizer_word_index.json",
    path_in_repo="tokenizer_word_index.json",
    repo_id=repo_id,
    repo_type="model",
)