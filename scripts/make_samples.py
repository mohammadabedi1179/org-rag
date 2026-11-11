"""Build a small fictional organisation: PDF, Word and Excel files plus a SQLite database.

The assessment supplies no documents. Every name and number here is invented.
Needs a font with Persian glyphs: FONT_DIR/FONT_FILE (Tahoma on Windows).
"""
import json
import os
import sqlite3
from pathlib import Path

import pymupdf as fitz
import openpyxl
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm

ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"
FONT_DIR = os.environ.get("RAG_FONT_DIR", "C:/Windows/Fonts")
FONT_FILE = os.environ.get("RAG_FONT_FILE", "tahoma.ttf")
CSS = f"""@font-face {{font-family: fa; src: url({FONT_FILE});}}
* {{font-family: fa; direction: rtl; text-align: right; font-size: 12px; line-height: 1.7;}}
h1 {{font-size: 18px;}} h2 {{font-size: 14px;}}
table {{border-collapse: collapse; width: 100%;}} td, th {{border: 1px solid black; padding: 4px;}}"""
PAGE = fitz.paper_rect("a4")
BOX = PAGE + (50, 50, -50, -50)


def add_page(pdf: fitz.Document, html: str) -> fitz.Page:
    page = pdf.new_page(width=PAGE.width, height=PAGE.height)
    page.insert_htmlbox(BOX, html, css=CSS, archive=fitz.Archive(FONT_DIR))
    return page


def picture(html: str, width: int = 460, height: int = 170) -> bytes:
    """Persian text as a pure image, so only a vision model can read it."""
    pdf = fitz.open()
    page = pdf.new_page(width=width, height=height)
    page.insert_htmlbox(page.rect + (10, 10, -10, -10), html, css=CSS, archive=fitz.Archive(FONT_DIR))
    return page.get_pixmap(dpi=150).tobytes("png")


def make_pdf(path: Path) -> None:
    pdf = fitz.open()
    add_page(pdf, """<h1>آیین‌نامه مرخصی و مزایای کارکنان شرکت آرمان‌پویا</h1>
<h2>ماده ۱: مرخصی استحقاقی</h2>
<p>هر کارمند تمام‌وقت در هر سال ۲۶ روز مرخصی استحقاقی دارد. حداکثر ۹ روز از مرخصی استفاده‌نشده به سال بعد منتقل می‌شود و مانده بیش از آن بازخرید نمی‌شود.</p>
<h2>ماده ۲: مرخصی استعلاجی</h2>
<p>برای مرخصی استعلاجی بیش از ۳ روز، ارائه گواهی پزشک معتمد شرکت الزامی است. گواهی باید حداکثر تا ۵ روز کاری پس از بازگشت به امور اداری تحویل داده شود.</p>
<h2>ماده ۳: دورکاری</h2>
<p>کارکنانی که دست‌کم ۱۲ ماه سابقه کار دارند می‌توانند با تأیید مدیر واحد، حداکثر ۸ روز در ماه دورکاری کنند.</p>""")
    add_page(pdf, """<h2>ماده ۴: جدول مزایای ماهانه</h2>
<p>مبالغ جدول زیر به میلیون ریال است و از ابتدای فروردین ۱۴۰۴ اجرا می‌شود.</p>
<table><tr><th>گروه شغلی</th><th>حق مسکن</th><th>بن خواربار</th><th>حق ایاب و ذهاب</th></tr>
<tr><td>کارشناس</td><td>90</td><td>45</td><td>30</td></tr>
<tr><td>کارشناس ارشد</td><td>120</td><td>55</td><td>35</td></tr>
<tr><td>سرپرست</td><td>160</td><td>65</td><td>42</td></tr>
<tr><td>مدیر واحد</td><td>210</td><td>80</td><td>60</td></tr></table>""")
    page = add_page(pdf, "<h2>ماده ۵: هزینه مأموریت</h2><p>سقف هزینه روزانه مأموریت در تصویر زیر آمده است.</p>")
    page.insert_image(fitz.Rect(70, 150, 525, 320), stream=picture(
        """<h2>سقف هزینه روزانه مأموریت</h2><p>داخل استان: 3500000 ریال</p><p>خارج استان: 5200000 ریال</p>"""))
    pdf.save(path)
    pdf.close()


def make_docx(path: Path, image: Path) -> None:
    document = Document()
    def paragraph(text):
        p = document.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    document.add_heading("دستورالعمل خرید کالا و خدمات", 1)
    paragraph("این دستورالعمل برای همه واحدهای شرکت آرمان‌پویا لازم‌الاجرا است.")
    document.add_heading("سقف اختیارات خرید", 2)
    rows = [("سمت", "سقف هر خرید (میلیون ریال)"), ("سرپرست", "150"), ("مدیر واحد", "800"), ("معاون مالی", "2000")]
    table = document.add_table(len(rows), 2, style="Table Grid")
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.cell(i, j).text = value
    document.add_heading("مناقصه", 2)
    paragraph("خرید بیش از 2000 میلیون ریال فقط از راه برگزاری مناقصه عمومی انجام می‌شود.")
    paragraph("برای هر خرید بیش از 300 میلیون ریال دریافت دست‌کم سه پیش‌فاکتور الزامی است.")
    document.add_heading("پرداخت", 2)
    document.add_picture(str(image), width=Cm(12))
    document.save(path)


def make_xlsx(path: Path) -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "بودجه واحدها"
    sheet.append(["واحد", "بودجه مصوب ۱۴۰۴ (میلیون ریال)", "هزینه‌شده تا شهریور (میلیون ریال)"])
    for row in [("فناوری اطلاعات", 18500, 9200), ("بازرگانی", 24000, 15100), ("منابع انسانی", 7600, 3400), ("مالی", 5200, 2650)]:
        sheet.append(row)
    sheet = book.create_sheet("قراردادها")
    sheet.append(["شماره قرارداد", "طرف قرارداد", "موضوع", "تاریخ پایان"])
    for row in [("ق-۱۰۱", "شرکت داده‌پرداز سپهر", "پشتیبانی شبکه", "1404/12/29"),
                ("ق-۱۰۲", "مؤسسه بهین‌کار", "آموزش کارکنان", "1405/03/31"),
                ("ق-۱۰۳", "شرکت نوین‌ابزار", "تأمین تجهیزات", "1404/10/15")]:
        sheet.append(row)
    book.save(path)
    book.close()


SCHEMA_NOTES = {
    "departments": {"_": "واحدهای سازمانی شرکت", "name": "نام واحد", "manager": "نام مدیر واحد", "city": "شهر محل واحد"},
    "employees": {"_": "کارکنان شرکت", "full_name": "نام و نام خانوادگی کارمند", "department_id": "شناسه واحد",
                  "job_group": "گروه شغلی", "hire_year": "سال استخدام"},
    "projects": {"_": "پروژه‌های شرکت", "title": "عنوان پروژه", "department_id": "شناسه واحد مجری", "status": "وضعیت پروژه",
                 "budget_million_rial": "بودجه پروژه به میلیون ریال"},
}


def make_database(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_suffix(".schema.json").write_text(json.dumps(SCHEMA_NOTES, ensure_ascii=False, indent=1), encoding="utf-8")
    with sqlite3.connect(path) as c:
        c.executescript("""
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL, manager TEXT NOT NULL, city TEXT NOT NULL);
        CREATE TABLE employees (id INTEGER PRIMARY KEY, full_name TEXT NOT NULL,
            department_id INTEGER NOT NULL REFERENCES departments(id), job_group TEXT NOT NULL, hire_year INTEGER NOT NULL);
        CREATE TABLE projects (id INTEGER PRIMARY KEY, title TEXT NOT NULL,
            department_id INTEGER NOT NULL REFERENCES departments(id), status TEXT NOT NULL, budget_million_rial INTEGER NOT NULL);
        INSERT INTO departments VALUES (1,'فناوری اطلاعات','نگین رحمانی','تهران'),(2,'بازرگانی','کامران فلاح','اصفهان'),
            (3,'منابع انسانی','سارا کیانی','تهران'),(4,'مالی','بهرام نادری','تهران');
        INSERT INTO employees VALUES (1,'علی صادقی',1,'کارشناس',1398),(2,'مریم اکبری',1,'کارشناس ارشد',1395),
            (3,'رضا مرادی',1,'سرپرست',1392),(4,'الهام نوری',2,'کارشناس',1401),(5,'حسین زارع',2,'سرپرست',1394),
            (6,'زهرا امینی',3,'کارشناس ارشد',1399),(7,'پویا شریفی',4,'کارشناس',1402),(8,'لیلا حاتمی',4,'سرپرست',1393);
        INSERT INTO projects VALUES (1,'نوسازی مرکز داده',1,'در حال اجرا',6400),(2,'سامانه فروش برخط',2,'در حال اجرا',3900),
            (3,'ارزیابی عملکرد',3,'پایان‌یافته',850),(4,'انبار هوشمند',2,'متوقف',2750),(5,'مهاجرت ایمیل',1,'پایان‌یافته',1200);
        """)


def main() -> None:
    docs = ROOT / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    make_pdf(docs / "آیین‌نامه مرخصی و مزایا.pdf")
    image = ROOT / "payment_rule.png"
    image.write_bytes(picture("<h2>مهلت پرداخت</h2><p>فاکتور تأییدشده حداکثر ظرف 14 روز کاری پرداخت می‌شود.</p>"))
    make_docx(docs / "دستورالعمل خرید.docx", image)
    image.unlink()
    make_xlsx(docs / "بودجه و قراردادها ۱۴۰۴.xlsx")
    make_database(ROOT / "company.db")
    print("wrote", ROOT)


if __name__ == "__main__":
    main()
