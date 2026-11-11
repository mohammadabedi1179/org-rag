"""Web interface: ask a Persian question, get an answer with its sources.

    python app.py [--index data/index] [--db data/sample/company.db] [--port 7861]
"""
import argparse
import json
from pathlib import Path

import gradio as gr

from rag.database import Database
from rag.models import Embedder, Reranker, load_chat
from rag.pipeline import RERANK_FLOOR, Assistant
from rag.search import Index

ROOT = Path(__file__).resolve().parent


def saved_floor() -> float:
    path = ROOT / "results/eval_dev.json"
    return json.loads(path.read_text(encoding="utf-8"))["floor"] if path.exists() else RERANK_FLOOR


def build_ui(assistant: Assistant) -> gr.Blocks:
    def respond(question: str):
        if not question.strip():
            return "پرسشی وارد نشده است.", "", ""
        a = assistant.answer(question.strip())
        sources = "\n\n".join(f"**[{i}] {s.citation}**\n\n{s.text}" for i, s in enumerate(a.sources, start=1))
        details = "\n".join(([f"SQL: {a.sql}"] if a.sql else []) + a.notes)
        return a.text, sources or "منبعی برای این پاسخ وجود ندارد.", details

    with gr.Blocks(title="پرسش از اسناد و پایگاه داده") as ui:
        gr.Markdown("## پرسش از اسناد و پایگاه داده سازمان")
        question = gr.Textbox(label="پرسش", lines=2, rtl=True)
        ask = gr.Button("پرسیدن", variant="primary")
        answer = gr.Textbox(label="پاسخ", lines=5, rtl=True, interactive=False)
        sources = gr.Markdown(label="منابع", rtl=True)
        with gr.Accordion("جزئیات اجرا", open=False):
            details = gr.Textbox(show_label=False, lines=4, interactive=False)
        ask.click(respond, question, [answer, sources, details])
        question.submit(respond, question, [answer, sources, details])
    return ui


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=ROOT / "data/index")
    parser.add_argument("--db", type=Path, default=ROOT / "data/sample/company.db")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    index = Index.load(args.index, Embedder(), Reranker())
    assistant = Assistant(load_chat(), index, Database(args.db), rerank_floor=saved_floor())
    build_ui(assistant).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
