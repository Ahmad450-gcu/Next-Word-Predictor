import os
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from huggingface_hub import HfApi

MODEL_NAME = "next-word-predictor"
PROD_ALIAS = "champion"
HF_REPO_ID = "dev-Ahmad450/next-word-predictor"

def main():
    version = os.environ["CANDIDATE_VERSION"]
    client = MlflowClient()
    mv = client.get_model_version(MODEL_NAME, version)
    local_dir = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="model", dst_path="promoted_model"
    )
    model_path = Path(local_dir) / "best_model_tied.keras"
    tokenizer_path = Path(local_dir) / "tokenizer_word_index.json"

    api = HfApi(token=os.environ["HF_TOKEN"])
    tag = f"v{version}"
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo="best_model_tied.keras", repo_id=HF_REPO_ID, repo_type="model")
    api.upload_file(path_or_fileobj=str(tokenizer_path), path_in_repo="tokenizer_word_index.json", repo_id=HF_REPO_ID, repo_type="model")
    api.create_tag(repo_id=HF_REPO_ID, tag=tag, repo_type="model")

    client.set_registered_model_alias(MODEL_NAME, PROD_ALIAS, version)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"hf_revision={tag}\n")

    print(f"Promoted v{version} to '{PROD_ALIAS}', tagged HF Hub as {tag}")

if __name__ == "__main__":
    main()