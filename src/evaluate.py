import argparse
import json
from pathlib import Path
import numpy as np
import yaml
import tensorflow as tf
from src.model.tied_lstm import TiedLSTMLanguageModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-path", default="models/best_model_tied.keras")
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)
    batch_size = params["evaluate"]["batch_size"]

    data_dir = Path(args.data_dir)
    X_test = np.load(data_dir / "test_X.npy")
    y_test = np.load(data_dir / "test_y.npy")

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = tf.keras.models.load_model(
        args.model_path, custom_objects={"TiedLSTMLanguageModel": TiedLSTMLanguageModel}
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    test_loss, test_top1, test_top5 = model.evaluate(test_ds)
    metrics = {
        "test_loss": float(test_loss),
        "test_perplexity": float(np.exp(test_loss)),
        "test_top1_accuracy": float(test_top1),
        "test_top5_accuracy": float(test_top5),
    }
    print(json.dumps(metrics, indent=2))

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()