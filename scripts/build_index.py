"""Read the documents and build the search index.

    python -m scripts.build_index [--docs FOLDER] [--out FOLDER] [--no-models]

--no-models skips images and dense vectors: useful on a machine without the models.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag.ingest import ingest_folder, to_dicts
from rag.search import Index

ROOT = Path(__file__).resolve().parent.parent


def build(docs: Path, out: Path, vision=None, embed=None) -> dict:
    chunks, report = ingest_folder(docs, vision, out / "vision_cache.json")
    Index.build(to_dicts(chunks), embed).save(out)
    summary = asdict(report)
    (out / "ingest_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=ROOT / "data/sample/documents")
    parser.add_argument("--out", type=Path, default=ROOT / "data/index")
    parser.add_argument("--no-models", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vision = embed = None
    if not args.no_models:
        from rag.models import Embedder, load_chat
        vision, embed = load_chat().read_image, Embedder()
    print(json.dumps(build(args.docs, args.out, vision, embed), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
