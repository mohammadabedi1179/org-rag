"""Persian text helpers shared by indexing, search and answer checking."""
import re

_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "ٱ": "ا", "ؤ": "و",
    "٫": ".", "٬": ",", "،": "،",
    **{d: str(i) for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹")},
    **{d: str(i) for i, d in enumerate("٠١٢٣٤٥٦٧٨٩")},
})
_DIACRITICS = re.compile(r"[ً-ٰٟـ]")  # harakat, superscript alef, tatweel
_WORD = re.compile(r"[^\W_]+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_LIST_MARK = re.compile(r"^\s*\d+[.)\-]\s", re.M)
_CITATION = re.compile(r"\[\s*(\d+(?:\s*[,،]\s*\d+)*)\s*\]")
_SUFFIX = re.compile(r"(?:هایی|های|ها)$")

STOPWORDS = frozenset(
    "و در به از که این آن را با برای است هست بود شد شده می نمی ها های تر ترین یک هر تا بر یا اما اگر "
    "چه چی چند چیست کدام کجا کی چرا چگونه آیا باید میشود شود کند کرد کرده دارد دارند".split()
)


def unify(text: str) -> str:
    """Arabic → Persian letters, any digits → ASCII, drop diacritics/tatweel."""
    return _DIACRITICS.sub("", text.translate(_CHAR_MAP))


def tokens(text: str) -> list[str]:
    """Search terms: half-spaces split words, stop words go, plural endings are cut."""
    out = []
    for word in _WORD.findall(unify(text).replace("‌", " ").lower()):
        if word in STOPWORDS:
            continue
        if len(word) > 4:
            word = _SUFFIX.sub("", word)
        out.append(word)
    return out


def citations(text: str) -> list[int]:
    """Evidence numbers cited as [1] or [1، 2], in order of first appearance."""
    seen = []
    for group in _CITATION.findall(unify(text)):
        for n in re.split(r"[,،]", group):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def numbers(text: str) -> set[str]:
    """Numeric values in the text, without citation marks, list marks or thousands separators."""
    text = _LIST_MARK.sub(" ", _CITATION.sub(" ", unify(text)))
    return {_canonical(m) for m in _NUMBER.findall(text)}


def _canonical(number: str) -> str:
    number = number.replace(",", "")
    return number.rstrip("0").rstrip(".") if "." in number else number
