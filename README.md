# سامانه بازیابی اطلاعات و پاسخ‌گویی هوشمند سازمانی

پاسخ به پرسش‌های فارسی از روی فایل‌های PDF، Word و Excel و یک پایگاه داده SQLite، همراه با ارجاع به منبع. اگر پاسخ در منابع نباشد، سامانه آن را اعلام می‌کند. همه مدل‌ها به‌صورت محلی اجرا می‌شوند.

اسناد و پایگاه داده پوشه `data/sample` ساختگی‌اند و فقط برای نمایش و ارزیابی ساخته شده‌اند.

## ساختار

```
app.py                  رابط وب
rag/ingest.py           خواندن PDF، Word و Excel، شامل جدول و تصویر
rag/search.py           جست‌وجوی ترکیبی BM25 و بردار، و بازرتبه‌بندی
rag/database.py         اجرای فقط‌خواندنی SQL نوشته‌شده توسط مدل
rag/pipeline.py         گردآوری شواهد، تولید پاسخ و بررسی ارجاع‌ها
rag/models.py           بارگذاری مدل زبانی، بردارساز و بازرتبه‌ساز
rag/text.py             یکسان‌سازی متن فارسی
scripts/build_index.py  ساخت نمایه
scripts/evaluate.py     ارزیابی روی eval/questions.jsonl
scripts/make_samples.py ساخت داده نمونه
scripts/smoke_app.py    آزمون رابط از طریق API
tests/                  آزمون‌های بدون نیاز به مدل
```

## نصب

پیش‌نیازها: لینوکس، `uv`، Python 3.12 و کارت NVIDIA با حدود ۱۲ گیگابایت حافظه.

```bash
bash setup.sh
.venv/bin/python -m pytest tests -q
```

مدل‌ها در نخستین اجرا از Hugging Face دریافت می‌شوند: `Qwen/Qwen3-VL-8B-Instruct`، `BAAI/bge-m3` و `BAAI/bge-reranker-v2-m3`.

## اجرا

```bash
.venv/bin/python -m scripts.build_index           # خواندن اسناد و ساخت نمایه
.venv/bin/python app.py                           # http://127.0.0.1:7861
```

برای اسناد خود سازمان:

```bash
.venv/bin/python -m scripts.build_index --docs /path/to/documents --out data/my_index
.venv/bin/python app.py --index data/my_index --db /path/to/company.db
```

توضیح فارسی جدول‌ها و ستون‌ها را در فایل `company.schema.json` کنار پایگاه داده قرار دهید؛ نمونه آن در `data/sample` است. این توضیح‌ها دقت پرسش‌های پایگاه داده را به‌طور محسوس بهتر کرد.

برای استفاده از یک سرویس سازگار با OpenAI به جای مدل محلی، `RAG_LLM_URL` و در صورت نیاز `RAG_LLM_KEY` و `RAG_LLM_MODEL` را تنظیم کنید. در این حالت متن اسناد به آن سرویس ارسال می‌شود.

## ارزیابی

```bash
.venv/bin/python -m scripts.evaluate --split dev
.venv/bin/python -m scripts.evaluate --split test --floor 0.05
```

نتیجه روی ۲۴ پرسش آزمون: ۲۱ پاسخ درست، ۱۴ از ۱۹ ارجاع درست، و رد صحیح هر ۵ پرسش بی‌پاسخ (`results/eval_test.md`). جزئیات و محدودیت‌ها در `report/report.pdf` آمده است.
