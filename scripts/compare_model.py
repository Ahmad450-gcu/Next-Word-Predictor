import os
import sys
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "next-word-predictor"
PROD_ALIAS = "champion"

def get_metrics(client, version):
    return client.get_run(version.run_id).data.metrics

def main():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    candidate = max(versions, key=lambda v: int(v.version))

    try:
        production = client.get_model_version_by_alias(MODEL_NAME, PROD_ALIAS)
    except MlflowException:
        print(f"No '{PROD_ALIAS}' alias set yet — nothing to compare against.")
        sys.exit(1)

    if candidate.version == production.version:
        print(f"Candidate v{candidate.version} is already '{PROD_ALIAS}' — nothing to compare.")
        sys.exit(0)

    c = get_metrics(client, candidate)
    p = get_metrics(client, production)
    print(f"Candidate v{candidate.version}: {c}")
    print(f"{PROD_ALIAS} (v{production.version}): {p}")

    better = (
        c["test_perplexity"] < p["test_perplexity"]
        and c["test_top1_accuracy"] >= p["test_top1_accuracy"]
        and c["test_top5_accuracy"] >= p["test_top5_accuracy"]
    )

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"promote={'true' if better else 'false'}\n")
            f.write(f"candidate_version={candidate.version}\n")

    if not better:
        print(f"v{candidate.version} did not beat {PROD_ALIAS} — leaving unpromoted.")
        sys.exit(1)

    print(f"v{candidate.version} beats {PROD_ALIAS} — eligible for promotion.")

if __name__ == "__main__":
    main()