"""Model backends. Heavy libraries are imported only by the backend that needs them.

RAG_LLM_URL set   → any OpenAI-compatible server (vLLM, llama.cpp, or a paid API).
RAG_LLM_URL unset → Qwen3-VL through transformers, in this process.
"""
import base64
import json
import os
import urllib.request

import numpy as np

CHAT_MODEL = os.environ.get("RAG_LLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
SEARCH_DEVICE = os.environ.get("RAG_SEARCH_DEVICE", "cpu")  # leaves the GPU to the language model
MAX_NEW_TOKENS = 400
BATCH = 16

VISION_PROMPT = ("همه متن و عددهای این تصویر را دقیقاً و به همان زبان بازنویسی کن. اگر جدول است، هر ردیف را در یک خط و "
                 "خانه‌ها را با | جدا کن. اگر نمودار یا شکل است، مقدارها و برچسب‌هایش را بنویس. چیزی از خودت اضافه نکن.")


def _image_message(image: bytes) -> list[dict]:
    uri = "data:image/png;base64," + base64.b64encode(image).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                         {"type": "text", "text": VISION_PROMPT}]}]


class RemoteChat:
    """OpenAI-compatible /chat/completions. The key comes only from RAG_LLM_KEY."""

    def __init__(self, url: str, model: str = CHAT_MODEL):
        self.url, self.model = url.rstrip("/") + "/chat/completions", model

    def __call__(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": MAX_NEW_TOKENS}
        headers = {"Content-Type": "application/json"}
        if key := os.environ.get("RAG_LLM_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)["choices"][0]["message"]["content"]

    def read_image(self, image: bytes) -> str:
        return self(_image_message(image))


class LocalQwen:
    """Text and images through one local model; CPU unless the GPU is explicitly visible."""

    def __init__(self, model: str = CHAT_MODEL):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model)
        options = {"dtype": torch.float32}
        if torch.cuda.is_available():  # 4-bit: the 8B model does not fit a 12 GB card otherwise
            from transformers import BitsAndBytesConfig
            options = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)}
        self.model = AutoModelForImageTextToText.from_pretrained(model, device_map="auto", **options).eval()

    def _generate(self, messages: list[dict]) -> str:
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def __call__(self, messages: list[dict]) -> str:
        wrapped = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        return self._generate(wrapped)

    def read_image(self, image: bytes) -> str:
        import io

        from PIL import Image
        picture = Image.open(io.BytesIO(image)).convert("RGB")
        return self._generate([{"role": "user", "content": [{"type": "image", "image": picture},
                                                            {"type": "text", "text": VISION_PROMPT}]}])


def load_chat():
    url = os.environ.get("RAG_LLM_URL")
    return RemoteChat(url) if url else LocalQwen()


class Embedder:
    def __init__(self, model: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=SEARCH_DEVICE)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=BATCH, normalize_embeddings=True))


class Reranker:
    def __init__(self, model: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model, device=SEARCH_DEVICE)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        scores = self.model.predict([(query, t) for t in texts], batch_size=BATCH)
        return [float(s) for s in 1 / (1 + np.exp(-np.asarray(scores)))] if np.min(scores) < 0 or np.max(scores) > 1 \
            else [float(s) for s in scores]
