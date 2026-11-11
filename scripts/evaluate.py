"""Score the whole system on eval/questions.jsonl and write results/eval.md.

    python -m scripts.evaluate [--split dev|test] [--floor X]

Tune the rerank floor with --split dev only; then run --split test once with that value.
An answer is correct when all strings of any one group in `answer` appear in it.
A citation is correct when a cited source contains the question's `source` text.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from rag.database import Database
from rag.models import Embedder, Reranker, load_chat
from rag.pipeline import RERANK_FLOOR, Assistant
from rag.search import Index
from rag.text import unify

ROOT = Path(__file__).resolve().parent.parent
FLOORS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def plain(text: str) -> str:
    return unify(text).replace("‌", " ").replace(",", "")


def correct_answer(question: dict, text: str) -> bool:
    return any(all(plain(s) in plain(text) for s in group) for group in question["answer"])


def correct_source(question: dict, sources) -> bool:
    return any(plain(question["source"]) in plain(s.citation) for s in sources)


def tune_floor(index: Index, questions: list[dict]) -> tuple[float, list[dict]]:
    """Pick the floor that best separates answerable document questions from unanswerable ones."""
    rows = []
    for q in questions:
        if q["type"] == "database":
            continue
        hits = index.search(q["question"], 1)
        rows.append((q["type"] == "none", hits[0].rerank if hits else 0.0))
    table = []
    for floor in FLOORS:
        answerable = [s >= floor for none, s in rows if not none]
        refused = [s < floor for none, s in rows if none]
        balanced = (sum(answerable) / max(len(answerable), 1) + sum(refused) / max(len(refused), 1)) / 2
        table.append({"floor": floor, "balanced_accuracy": round(balanced, 3)})
    return max(table, key=lambda r: r["balanced_accuracy"])["floor"], table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--floor", type=float, default=RERANK_FLOOR)
    parser.add_argument("--index", type=Path, default=ROOT / "data/index")
    parser.add_argument("--db", type=Path, default=ROOT / "data/sample/company.db")
    args = parser.parse_args()

    lines = (ROOT / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [q for q in map(json.loads, filter(None, lines)) if q["split"] == args.split]
    index = Index.load(args.index, Embedder(), Reranker())
    floor = args.floor
    if args.split == "dev":
        floor, sweep = tune_floor(index, questions)
        print("floor sweep:", sweep, "chosen:", floor)
    assistant = Assistant(load_chat(), index, Database(args.db), rerank_floor=floor)

    records = []
    for q in questions:
        started = time.monotonic()
        a = assistant.answer(q["question"])
        none = q["type"] == "none"
        records.append({
            **{k: q[k] for k in ("id", "type", "question")}, "reply": a.text, "found": a.found,
            "ok": (not a.found) if none else a.found and correct_answer(q, a.text),
            "source_ok": None if none else a.found and correct_source(q, a.sources),
            "citations": [s.citation for s in a.sources], "sql": a.sql, "notes": a.notes,
            "seconds": round(time.monotonic() - started, 1),
        })
        print(records[-1]["id"], records[-1]["ok"], records[-1]["source_ok"], records[-1]["seconds"], flush=True)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"eval_{args.split}.json").write_text(json.dumps({"floor": floor, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    groups = defaultdict(list)
    for r in records:
        groups[r["type"]].append(r)
    report = [f"# Evaluation: {args.split} (rerank floor {floor})", "", "| type | n | answer ok | source ok | mean s |", "|---|---|---|---|---|"]
    for kind, rows in [*sorted(groups.items()), ("ALL", records)]:
        sources = [r["source_ok"] for r in rows if r["source_ok"] is not None]
        report.append(f"| {kind} | {len(rows)} | {sum(r['ok'] for r in rows)} | "
                      f"{sum(sources)}/{len(sources)} | {sum(r['seconds'] for r in rows) / len(rows):.1f} |")
    (out / f"eval_{args.split}.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
