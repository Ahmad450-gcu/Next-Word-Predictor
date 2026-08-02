import argparse
import json
import os
from pathlib import Path
import numpy as np
import yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.model.tied_lstm import TiedLSTMLanguageModel

def load_dataset(x_path, y_path, batch_size, shuffle, seed):
    X = np.load(x_path)
    y = np.load(y_path)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, X.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--warm-start-embedding-from", default=None,
                         help="Optional path to a different-architecture checkpoint to copy "
                              "the embedding matrix from (e.g. an earlier untied baseline).")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    seed = params.get("seed", 42)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    mixed_precision.set_global_policy(params["train"]["precision_policy"])

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(model_dir / "best_model_tied.keras")

    with open(data_dir / "tokenizer_word_index.json") as f:
        vocab_size = len(json.load(f)) + 1

    sequence_length = params["sequences"]["sequence_length"]
    model_cfg = params["model"]
    train_cfg = params["train"]

    train_ds, n_train = load_dataset(
        data_dir / "train_X.npy", data_dir / "train_y.npy",
        train_cfg["batch_size"], shuffle=True, seed=seed,
    )
    valid_ds, n_valid = load_dataset(
        data_dir / "valid_X.npy", data_dir / "valid_y.npy",
        train_cfg["batch_size"], shuffle=False, seed=seed,
    )
    print(f"train samples: {n_train:,} | valid samples: {n_valid:,}")

    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path} - resuming from it.")
        model = tf.keras.models.load_model(
            checkpoint_path, custom_objects={"TiedLSTMLanguageModel": TiedLSTMLanguageModel}
        )
    else:
        print("No existing checkpoint - building a fresh model.")
        model = TiedLSTMLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=model_cfg["embedding_dim"],
            lstm_units=model_cfg["lstm_units"],
            dropout_rate=model_cfg["dropout_rate"],
            sequence_length=sequence_length,
            name="tied_lstm_lm",
        )
        dummy_input = tf.zeros((1, sequence_length), dtype=tf.int32)
        _ = model(dummy_input)  

        if args.warm_start_embedding_from and os.path.exists(args.warm_start_embedding_from):
            print(f"Warm-starting embedding from {args.warm_start_embedding_from}")
            old_model = tf.keras.models.load_model(args.warm_start_embedding_from)
            model.embedding.set_weights(old_model.get_layer("embedding").get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"], clipnorm=train_cfg["clipnorm"])
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=train_cfg["patience"], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=train_cfg["reduce_lr_patience"], min_lr=1e-6, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = model.fit(train_ds, validation_data=valid_ds, epochs=train_cfg["epochs"], callbacks=callbacks, verbose=1,)

    with open(model_dir / "train_history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    print(f"Best model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()