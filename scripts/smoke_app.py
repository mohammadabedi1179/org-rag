"""Start the web interface, ask one answerable and one unanswerable question through its API, stop.

    python -m scripts.smoke_app
"""
import app
from gradio_client import Client

from rag.database import Database
from rag.models import Embedder, Reranker, load_chat
from rag.pipeline import NOT_FOUND, Assistant
from rag.search import Index

PORT = 7862
ANSWERABLE = "حق مسکن گروه شغلی سرپرست چقدر است؟"
UNANSWERABLE = "پایتخت فرانسه کجاست؟"


def main() -> None:
    index = Index.load(app.ROOT / "data/index", Embedder(), Reranker())
    assistant = Assistant(load_chat(), index, Database(app.ROOT / "data/sample/company.db"), app.saved_floor())
    ui = app.build_ui(assistant)
    ui.launch(server_name="127.0.0.1", server_port=PORT, prevent_thread_lock=True, quiet=True)
    try:
        client = Client(f"http://127.0.0.1:{PORT}", verbose=False)
        reply, sources, _ = client.predict(ANSWERABLE, api_name="/respond")
        print("ANSWER:", reply, "\nSOURCES:", sources[:200])
        assert "160" in reply.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")) and "صفحه 2" in sources
        reply, sources, _ = client.predict(UNANSWERABLE, api_name="/respond")
        print("REFUSAL:", reply)
        assert reply == NOT_FOUND
        print("SMOKE_OK")
    finally:
        ui.close()


if __name__ == "__main__":
    main()
