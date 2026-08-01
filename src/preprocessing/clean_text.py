import argparse
import json
from pathlib import Path
import yaml
import re
import unicodedata

_HEADING_RE = re.compile(r"^=+.*=+$")
_ARTIFACT_MAP = {" @-@ ": "-", " @,@ ": ",", " @.@ ": "."}
_NUMBER_RE = re.compile(r"(?<!\S)[+-]?\d[\d,]*\.?\d*(?!\S)")


def split_articles(lines, remove_headings=False):
    articles = []
    current_article = []
    for line in lines:
        line = line.strip()
        if line == "":
            if current_article:
                articles.append(current_article)
                current_article = []
        else:
            if remove_headings and _HEADING_RE.match(line):
                continue
            current_article.append(line)
    if current_article:
        articles.append(current_article)
    return articles


def normalize_artifacts(text):
    for artifact, replacement in _ARTIFACT_MAP.items():
        text = text.replace(artifact, replacement)
    return text


def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)


def mask_numbers(text):
    return _NUMBER_RE.sub("<NUM>", text)


def articles_to_text(articles, normalize_wikitext=False, lowercase=False,
                      unicode_normalize=False, normalize_numbers=False):
    article_texts = []
    for article in articles:
        text = " ".join(article)
        if unicode_normalize:
            text = normalize_unicode(text)
        if normalize_wikitext:
            text = normalize_artifacts(text)
        if normalize_numbers:
            text = mask_numbers(text)
        if lowercase:
            text = text.lower()
        article_texts.append(text)
    return article_texts


def process_split(raw_path, cfg):
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    articles = split_articles(lines, remove_headings=cfg["remove_headings"])
    texts = articles_to_text(
        articles,
        normalize_wikitext=cfg["normalize_wikitext"],
        lowercase=cfg["lowercase"],
        unicode_normalize=cfg["unicode_normalize"],
        normalize_numbers=cfg["normalize_numbers"],
    )
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)
    cfg = params["preprocessing"]

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, filename in [("train", "wiki.train.raw"), ("valid", "wiki.valid.raw"), ("test", "wiki.test.raw")]:
        texts = process_split(raw_dir / filename, cfg)
        out_path = out_dir / f"{split}_texts.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(texts, f)
        print(f"{split}: {len(texts):,} articles -> {out_path}")


if __name__ == "__main__":
    main()