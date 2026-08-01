import argparse
import json
from pathlib import Path
import yaml
from tensorflow.keras.preprocessing.text import Tokenizer

def build_tokenizer(train_texts, min_frequency, oov_token="<UNK>"):
    tokenizer = Tokenizer(lower=False, filters="", oov_token=oov_token)
    tokenizer.fit_on_texts(train_texts)

    original_vocab_size = len(tokenizer.word_index) + 1

    kept_words = [
        w for w, c in sorted(tokenizer.word_counts.items(), key=lambda x: (-x[1], x[0]))
        if c >= min_frequency
    ]
    new_word_index = {oov_token: 1}
    for i, w in enumerate(kept_words, start=2):
        new_word_index[w] = i

    tokenizer.word_index = new_word_index
    tokenizer.index_word = {i: w for w, i in new_word_index.items()}

    filtered_vocab_size = len(tokenizer.word_index) + 1
    stats = {
        "original_vocab_size": original_vocab_size,
        "filtered_vocab_size": filtered_vocab_size,
        "words_removed_by_min_freq": original_vocab_size - filtered_vocab_size,
    }
    return tokenizer, stats

def oov_stats(sequences, oov_index):
    total = sum(len(seq) for seq in sequences)
    oov = sum(tok == oov_index for seq in sequences for tok in seq)
    return {"total_tokens": total, "oov_tokens": oov, "oov_percent": (oov / total * 100) if total else 0.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--in-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)
    cfg = params["tokenizer"]

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_dir / "train_texts.json") as f:
        train_texts = json.load(f)

    tokenizer, vocab_stats = build_tokenizer(train_texts, min_frequency=cfg["min_frequency"], oov_token=cfg["oov_token"])

    with open(out_dir / "tokenizer_word_index.json", "w") as f:
        json.dump(tokenizer.word_index, f, indent=2)

    oov_index = tokenizer.word_index[tokenizer.oov_token]
    all_oov_stats = {}
    for split in ["train", "valid", "test"]:
        with open(in_dir / f"{split}_texts.json") as f:
            texts = json.load(f)
        sequences = tokenizer.texts_to_sequences(texts)
        with open(out_dir / f"{split}_sequences.json", "w") as f:
            json.dump(sequences, f)
        all_oov_stats[split] = oov_stats(sequences, oov_index)
        print(f"{split}: {len(sequences):,} article sequences -> {out_dir / f'{split}_sequences.json'}")

    vocab_stats["oov"] = all_oov_stats
    with open(out_dir / "vocab_stats.json", "w") as f:
        json.dump(vocab_stats, f, indent=2)
    print(json.dumps(vocab_stats, indent=2))

if __name__ == "__main__":
    main()