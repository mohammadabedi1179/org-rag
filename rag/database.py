"""Read-only access for model-written SQL.

Model output is untrusted, so safety does not depend on parsing the query: the
file is opened read-only and SQLite's authorizer refuses everything except
reading user tables. A progress handler stops runaway queries.
"""
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

MAX_ROWS = 50
TIMEOUT_S = 5.0
_ALLOWED_FUNCTIONS = frozenset({
    "abs", "avg", "coalesce", "count", "group_concat", "ifnull", "instr", "length", "lower", "max", "min",
    "nullif", "replace", "round", "strftime", "substr", "sum", "total", "trim", "upper", "date",
})
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_COMMENT = re.compile(r"--[^\n]*")


class QueryError(Exception):
    pass


@dataclass
class Result:
    sql: str
    columns: list[str]
    rows: list[tuple]
    tables: list[str]
    truncated: bool


def clean_sql(text: str) -> str:
    match = _FENCE.search(text)
    return _COMMENT.sub("", match.group(1) if match else text).strip().rstrip(";").strip()


class Database:
    """Optional `<name>.schema.json` next to the file gives Persian meanings:
    {"table": {"_": "what the table holds", "column": "what the column holds"}}.
    """

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        notes = self.path.with_suffix(".schema.json")
        self.notes: dict = json.loads(notes.read_text(encoding="utf-8")) if notes.exists() else {}
        self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, timeout=TIMEOUT_S)

    def schema(self, samples: int = 2) -> str:
        """Table definitions and a few rows, for the prompt that writes SQL."""
        with self._connect() as c:
            parts = []
            for name, sql in c.execute("SELECT name, sql FROM sqlite_master WHERE type='table' "
                                       "AND name NOT LIKE 'sqlite_%' ORDER BY name"):
                rows = c.execute(f'SELECT * FROM "{name}" LIMIT {samples}').fetchall()
                meaning = [f"-- {k if k != '_' else name}: {v}" for k, v in self.notes.get(name, {}).items()]
                parts.append("\n".join([sql, *meaning, "-- نمونه ردیف‌ها:"]
                                       + ["-- " + " | ".join(str(v) for v in r) for r in rows]))
        return "\n\n".join(parts)

    def label(self, column: str, tables: list[str]) -> str:
        """A column name with its Persian meaning, for the evidence shown to the answer model."""
        meaning = next((self.notes[t][column] for t in tables if column in self.notes.get(t, {})), None)
        return f"{column} ({meaning})" if meaning else column

    def run(self, sql: str) -> Result:
        sql = clean_sql(sql)
        tables: list[str] = []

        def authorize(action, a, b, _db, _trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and not a.startswith("sqlite_"):
                if a not in tables:
                    tables.append(a)
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and b.lower() in _ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        deadline = time.monotonic() + TIMEOUT_S
        connection = self._connect()
        try:
            connection.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
            connection.set_authorizer(authorize)
            cursor = connection.execute(sql)
            if cursor.description is None:
                raise QueryError("query returned no result set")
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS + 1)
        except (sqlite3.Error, sqlite3.Warning) as e:
            raise QueryError(str(e)) from e
        finally:
            connection.close()
        return Result(sql, columns, rows[:MAX_ROWS], tables, len(rows) > MAX_ROWS)
