import argparse
import json
from pathlib import Path
import numpy as np
import yaml

def generate_sequences(sequences, sequence_length):
    X, y = [], []
    for article in sequences:
        if len(article) <= sequence_length:
            continue
        for i in range(sequence_length, len(article)):
            X.append(article[i - sequence_length:i])
            y.append(article[i])
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--in-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)
    sequence_length = params["sequences"]["sequence_length"]

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"sequence_length": sequence_length}
    for split in ["train", "valid", "test"]:
        with open(in_dir / f"{split}_sequences.json") as f:
            sequences = json.load(f)
        X, y = generate_sequences(sequences, sequence_length)
        X = np.array(X, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        np.save(out_dir / f"{split}_X.npy", X)
        np.save(out_dir / f"{split}_y.npy", y)
        stats[f"{split}_samples"] = int(X.shape[0])
        print(f"{split}: X{X.shape} y{y.shape}")

    with open(out_dir / "sequence_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()