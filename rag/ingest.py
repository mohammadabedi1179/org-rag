"""Read PDF, Word and Excel files into chunks that remember where they came from.

Tables are cut by rows and every piece repeats the header, so a retrieved cell
still has its column names. Images go to a vision model when one is given;
without one they are counted in the report, never silently dropped.
"""
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pymupdf as fitz
import openpyxl
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

CHUNK_CHARS = 700
OVERLAP_CHARS = 100
TABLE_ROWS = 15
MIN_IMAGE_PX = 80
MIN_PAGE_LETTERS = 30

Vision = Callable[[bytes], str]
_SENTENCE = re.compile(r"(?<=[.؟!؛:])\s+|\n+")
_LETTER = re.compile(r"[^\W\d_]")
_PRESENTATION = re.compile("[\uFB50-\uFDFF\uFE70-\uFEFC]+")


@dataclass
class Chunk:
    doc: str
    kind: str  # text | table | image
    locator: str  # human-readable place: page, sheet and rows, or section
    text: str
    page: int | None = None


@dataclass
class Report:
    files: int = 0
    chunks: int = 0
    tables: int = 0
    images_read: int = 0
    images_skipped: int = 0
    pages_without_text: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class VisionCache:
    """Remember image readings by content hash, because the vision model is slow."""

    def __init__(self, vision: Vision | None, path: Path):
        self.vision, self.path = vision, path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def read(self, image: bytes) -> str | None:
        key = hashlib.sha256(image).hexdigest()
        if key not in self.data:
            if self.vision is None:
                return None
            self.data[key] = self.vision(image).strip()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0), encoding="utf-8")
        return self.data[key]


def split_text(text: str) -> list[str]:
    """Group sentences into overlapping pieces of about CHUNK_CHARS."""
    pieces, current = [], ""
    for sentence in (s.strip() for s in _SENTENCE.split(text)):
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > CHUNK_CHARS:
            pieces.append(current)
            current = current[-OVERLAP_CHARS:].split(" ", 1)[-1]
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def clean_extracted(text: str) -> str:
    """PDFs often store shaped letter forms; search and models need the plain letters."""
    return _PRESENTATION.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text).replace("\xa0", " ")


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return " ".join(str(value).split())


def table_chunks(rows: list[list[str]], doc: str, locator: Callable[[int, int], str],
                 page: int | None = None, title: str = "") -> list[Chunk]:
    """Cut a table into row groups; each group starts with the header row."""
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    header, body = rows[0], rows[1:] or rows[:1]
    prefix = f"{title}\n" if title else ""
    chunks = []
    for start in range(0, len(body), TABLE_ROWS):
        group = body[start:start + TABLE_ROWS]
        lines = [" | ".join(header)] + [" | ".join(r) for r in group]
        chunks.append(Chunk(doc, "table", locator(start + 1, start + len(group)), prefix + "\n".join(lines), page))
    return chunks


def _image_chunk(image: bytes, doc: str, locator: str, page: int | None, cache: VisionCache,
                 report: Report) -> list[Chunk]:
    reading = cache.read(image)
    if reading is None:
        report.images_skipped += 1
        return []
    report.images_read += 1
    return [Chunk(doc, "image", locator, reading, page)]


def read_pdf(path: Path, cache: VisionCache, report: Report) -> list[Chunk]:
    chunks: list[Chunk] = []
    with fitz.open(path) as pdf:
        for number, page in enumerate(pdf, start=1):
            where = f"صفحه {number}"
            found = page.find_tables().tables
            boxes = [fitz.Rect(t.bbox) for t in found]
            for i, table in enumerate(found, start=1):
                # Cell by cell: the table extractor returns right-to-left text in reversed character order.
                rows = [[_cell(clean_extracted(page.get_text("text", clip=fitz.Rect(box)))) if box else ""
                         for box in row.cells] for row in table.rows]
                report.tables += 1
                chunks += table_chunks(rows, path.name, lambda a, b: f"{where}، جدول {i}، ردیف {a} تا {b}", number)

            blocks = sorted((b for b in page.get_text("blocks") if b[6] == 0), key=lambda b: (b[1], -b[2]))
            text = "\n".join(clean_extracted(b[4]).strip() for b in blocks if not any(fitz.Rect(b[:4]).intersects(r) for r in boxes))
            chunks += [Chunk(path.name, "text", where, s, number) for s in split_text(text)]

            images = page.get_images(full=True)
            if len(_LETTER.findall(text)) < MIN_PAGE_LETTERS and not found:
                # Scanned page or broken font: read the whole page as one picture.
                report.pages_without_text.append(f"{path.name}: {where}")
                png = page.get_pixmap(dpi=200).tobytes("png")
                chunks += _image_chunk(png, path.name, where, number, cache, report)
                continue
            for i, info in enumerate(images, start=1):
                image = pdf.extract_image(info[0])
                if min(image["width"], image["height"]) >= MIN_IMAGE_PX:
                    chunks += _image_chunk(image["image"], path.name, f"{where}، تصویر {i}", number, cache, report)
    return chunks


def read_docx(path: Path, cache: VisionCache, report: Report) -> list[Chunk]:
    """Word files store no page numbers, so sources are given by heading and table number."""
    document = Document(path)
    chunks: list[Chunk] = []
    heading, buffer, tables, images = "ابتدای سند", [], 0, 0

    def flush():
        chunks.extend(Chunk(path.name, "text", f"بخش «{heading}»", s) for s in split_text("\n".join(buffer)))
        buffer.clear()

    for item in document.iter_inner_content():
        if isinstance(item, Table):
            tables += 1
            report.tables += 1
            n, title = tables, heading
            rows = [[_cell(c.text) for c in row.cells] for row in item.rows]
            chunks += table_chunks(rows, path.name, lambda a, b: f"بخش «{title}»، جدول {n}، ردیف {a} تا {b}", title=title)
        elif isinstance(item, Paragraph):
            for run_image in item._p.xpath(".//a:blip/@r:embed"):
                images += 1
                blob = document.part.related_parts[run_image].blob
                chunks += _image_chunk(blob, path.name, f"بخش «{heading}»، تصویر {images}", None, cache, report)
            if item.style.name.startswith("Heading") and item.text.strip():
                flush()
                heading = item.text.strip()
            elif item.text.strip():
                buffer.append(item.text.strip())
    flush()
    return chunks


def read_xlsx(path: Path, cache: VisionCache, report: Report) -> list[Chunk]:
    chunks: list[Chunk] = []
    book = openpyxl.load_workbook(path, data_only=True)
    for sheet in book.worksheets:
        rows = [[_cell(c) for c in row] for row in sheet.iter_rows(values_only=True)]
        first = next((i for i, r in enumerate(rows) if any(r)), None)
        if first is not None:
            report.tables += 1
            # Row numbers in the citation are the sheet's own: header at first+1, data after it.
            chunks += table_chunks(rows[first:], path.name,
                                   lambda a, b, s=sheet.title, o=first + 1: f"برگه «{s}»، ردیف {a + o} تا {b + o}",
                                   title=f"برگه {sheet.title}")
        for i, image in enumerate(getattr(sheet, "_images", []), start=1):
            chunks += _image_chunk(image._data(), path.name, f"برگه «{sheet.title}»، تصویر {i}", None, cache, report)
    book.close()
    return chunks


READERS = {".pdf": read_pdf, ".docx": read_docx, ".xlsx": read_xlsx}


def ingest_folder(folder: Path, vision: Vision | None, cache_path: Path) -> tuple[list[Chunk], Report]:
    """Read every supported file; one bad file is reported and does not stop the rest."""
    cache, report, chunks = VisionCache(vision, cache_path), Report(), []
    for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in READERS and not p.name.startswith("~$")):
        try:
            found = READERS[path.suffix.lower()](path, cache, report)
        except Exception as e:  # a corrupt upload must not block the other documents
            report.errors.append(f"{path.name}: {type(e).__name__}: {e}")
            continue
        report.files += 1
        chunks += found
    report.chunks = len(chunks)
    return chunks, report


def to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
