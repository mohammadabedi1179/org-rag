"""Checks that need no language model. The model is replaced by a scripted function."""
import sqlite3
from pathlib import Path

import pytest

from rag.database import Database, QueryError, clean_sql
from rag.ingest import ingest_folder, split_text, to_dicts
from rag.pipeline import NOT_FOUND, Assistant, Evidence, database_evidence, problem_with
from rag.search import Index
from rag.text import citations, numbers, tokens, unify
from scripts import make_samples

SAMPLE = make_samples.ROOT


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    if not (SAMPLE / "company.db").exists():  # building them needs a Persian font; see make_samples
        make_samples.main()
    vision = lambda image: "متن خوانده‌شده از تصویر"
    return ingest_folder(SAMPLE / "documents", vision, tmp_path_factory.mktemp("c") / "vision.json")


@pytest.fixture(scope="session")
def index(corpus):
    return Index.build(to_dicts(corpus[0]))


@pytest.fixture(scope="session")
def db(corpus):
    return Database(SAMPLE / "company.db")


def find(chunks, doc_part, kind):
    return [c for c in chunks if doc_part in c.doc and c.kind == kind]


def test_text_helpers():
    assert unify("كيفيت ۱۲") == "کیفیت 12"
    assert tokens("كتاب‌ها") == tokens("کتاب ها") or "کتاب" in tokens("کتاب‌ها")
    assert numbers("مبلغ ۳٬۵۰۰٫۵۰ در [۲]\n1. مورد") == {"3500.5"}
    assert citations("الف [1] ب [۲، 3] ج [1]") == [1, 2, 3]
    assert all(len(p) < 900 for p in split_text("جمله‌ای کوتاه است. " * 200))


def test_all_files_read(corpus):
    chunks, report = corpus
    assert report.files == 3 and not report.errors and report.images_read == 2 and report.images_skipped == 0


def test_pdf_text_keeps_page_and_logical_order(corpus):
    text = find(corpus[0], ".pdf", "text")
    page1 = " ".join(c.text for c in text if c.page == 1)
    assert "مرخصی" in unify(page1) and "26" in unify(page1), page1[:300]


def test_pdf_table_rows_stay_with_header(corpus):
    tables = find(corpus[0], ".pdf", "table")
    assert len(tables) == 1 and tables[0].page == 2
    lines = [unify(l) for l in tables[0].text.splitlines()]
    assert "حق مسکن" in lines[0]
    manager = next(l for l in lines if "مدیر واحد" in l)
    assert [p.strip() for p in manager.split("|")][1:] == ["210", "80", "60"]


def test_image_chunks_carry_location(corpus):
    pdf = find(corpus[0], ".pdf", "image")
    docx = find(corpus[0], ".docx", "image")
    assert pdf[0].page == 3 and "تصویر" in pdf[0].locator
    assert "پرداخت" in docx[0].locator


def test_missing_vision_is_reported(tmp_path):
    chunks, report = ingest_folder(SAMPLE / "documents", None, tmp_path / "v.json")
    assert report.images_skipped == 2 and not [c for c in chunks if c.kind == "image"]


def test_docx_table_and_section(corpus):
    table = find(corpus[0], ".docx", "table")[0]
    assert "سقف اختیارات خرید" in table.locator and "800" in table.text


def test_xlsx_citation_uses_real_row_numbers(corpus):
    sheet = [c for c in find(corpus[0], ".xlsx", "table") if "بودجه واحدها" in c.locator][0]
    assert "ردیف 2 تا 5" in sheet.locator and "18500" in sheet.text


def test_corrupt_file_does_not_stop_the_rest(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    (tmp_path / "bad2.docx").write_bytes(b"nope")
    chunks, report = ingest_folder(tmp_path, None, tmp_path / "v.json")
    assert report.files == 0 and len(report.errors) == 2


@pytest.mark.parametrize("query,doc", [
    ("چند روز مرخصی استحقاقی داریم", ".pdf"),
    ("سقف خرید معاون مالی", ".docx"),
    ("بودجه مصوب واحد بازرگانی", ".xlsx"),
    ("طرف قرارداد ق-۱۰۲", ".xlsx"),
])
def test_bm25_finds_the_right_file(index, query, doc):
    assert any(h.chunk["doc"].endswith(doc) for h in index.search(query, 3))


def test_arabic_letters_and_digits_still_match(index):
    assert index.search("مرخصي", 1)[0].chunk["doc"].endswith(".pdf")


def test_no_terms_no_hits(index):
    assert index.search("زرافه کهکشان", 5) == []


def test_select_works_and_reports_tables(db):
    r = db.run("```sql\nSELECT id, full_name FROM employees WHERE department_id = 1;\n```")
    assert r.tables == ["employees"] and len(r.rows) == 3


@pytest.mark.parametrize("sql", [
    "DROP TABLE employees",
    "DELETE FROM employees",
    "UPDATE employees SET hire_year = 1",
    "INSERT INTO departments VALUES (9,'a','b','c')",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA table_info(employees)",
    "SELECT * FROM sqlite_master",
    "SELECT 1; DROP TABLE employees",
    "SELECT load_extension('x')",
])
def test_dangerous_sql_refused(db, sql):
    with pytest.raises(QueryError):
        db.run(sql)
    with sqlite3.connect(db.path) as c:
        assert c.execute("SELECT count(*) FROM employees").fetchone()[0] == 8


def test_runaway_query_is_stopped(db):
    with pytest.raises(QueryError):
        db.run("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c")


def test_clean_sql():
    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert clean_sql("SELECT 1;\n\n-- توضیح اضافه\n-- و باز هم") == "SELECT 1"


def test_schema_notes_reach_prompt_and_evidence(db):
    assert "نام مدیر واحد" in db.schema()
    evidence = database_evidence(db.run("SELECT id, manager FROM departments WHERE id = 2"), db)
    assert "مدیر" in evidence.text and "تعداد ردیف‌های نتیجه: 1" in evidence.text


EVIDENCE = [Evidence("document", "الف.pdf، صفحه 1", "سالانه ۲۶ روز مرخصی"), Evidence("database", "جدول x", "a\n5")]


@pytest.mark.parametrize("draft,ok", [
    ("۲۶ روز است [1]", True),
    ("۲۶ روز است", False),
    ("۲۶ روز است [3]", False),
    ("۳۰ روز است [1]", False),
    ("۵ روز است [1]", False),  # 5 exists only in evidence 2, which is not cited
    ("۵ مورد [2]", True),
])
def test_answer_checks(draft, ok):
    assert (problem_with(draft, EVIDENCE, "چند روز؟") is None) == ok


def scripted(*answers):
    """A stand-in model: SQL prompts get NONE unless given, answers come from the list."""
    queue = list(answers)

    def llm(messages):
        if messages[0]["role"] != "system":
            return "NONE"
        return queue.pop(0)
    return llm


def test_answer_with_valid_citation(index, db):
    a = Assistant(scripted("۲۶ روز در سال [1]"), index, db).answer("چند روز مرخصی استحقاقی داریم")
    assert a.found and a.sources and a.sources[0].citation.endswith(tuple("0123456789")) and "صفحه" in a.sources[0].citation


def test_invented_number_is_retried_then_refused(index, db):
    a = Assistant(scripted("۹۹۹۹ روز [1]", "۸۸۸۸ روز [1]"), index, db).answer("چند روز مرخصی استحقاقی داریم")
    assert not a.found and a.text == NOT_FOUND


def test_model_refusal_is_respected(index, db):
    a = Assistant(scripted("NOT_FOUND"), index, db).answer("چند روز مرخصی استحقاقی داریم")
    assert not a.found and a.text == NOT_FOUND


def test_no_evidence_skips_the_answer_model(index):
    def llm(messages):
        raise AssertionError("must not be called")
    a = Assistant(llm, index).answer("زرافه کهکشان")
    assert not a.found and a.text == NOT_FOUND


def test_database_evidence_cites_records(index, db):
    def llm(messages):
        if messages[0]["role"] != "system":
            return "SELECT id, full_name FROM employees WHERE id = 2"
        return "مریم اکبری [1]"
    a = Assistant(llm, index, db).answer("کارمند شماره ۲ کیست")
    assert a.found and a.sources[0].kind == "database" and "رکورد" in a.sources[0].citation and "2" in a.sources[0].citation


def test_bad_sql_is_repaired_once(index, db):
    replies = iter(["SELECT nope FROM employees", "SELECT id FROM employees WHERE id = 1"])

    def llm(messages):
        return next(replies) if messages[0]["role"] != "system" else "NOT_FOUND"
    a = Assistant(llm, index, db).answer("مرخصی کارمند ۱")
    assert a.sql and "nope" not in a.sql and any("خطای SQL" in n for n in a.notes)
