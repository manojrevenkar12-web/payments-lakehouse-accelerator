"""SQL guardrails for model-generated queries.

The LLM is treated as untrusted: whatever it returns must pass these checks before it
touches Spark. A query is allowed only if it is a single read-only SELECT over allowlisted
gold tables, references no raw-PII column, calls no denylisted function, and carries a LIMIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from ..privacy import PII_COLUMNS

FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Merge, exp.Command,
                   exp.Into, exp.TruncateTable, exp.Set, exp.Use)
FORBIDDEN_FUNCTIONS = {"reflect", "java_method", "read_files", "http_request", "ai_query", "secret", "input_file_name",
                       "current_user", "session_user", "uuid", "rand", "randn"}
DIALECT = "databricks"


@dataclass
class Verdict:
    allowed: bool
    sql: str | None = None
    tables: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def check(sql: str, allowed_tables: set[str], max_rows: int = 200) -> Verdict:
    try:
        statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except sqlglot.errors.ParseError as e:
        return Verdict(False, reasons=[f"unparseable SQL: {str(e).splitlines()[0]}"])
    if len(statements) != 1:
        return Verdict(False, reasons=[f"expected exactly one statement, got {len(statements)}"])
    tree = statements[0]
    reasons: list[str] = []

    if not isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect)):
        reasons.append(f"only SELECT queries are allowed (got {type(tree).__name__})")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            reasons.append(f"forbidden operation: {type(node).__name__}")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS:
                reasons.append(f"forbidden function: {name}")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = []
    for t in tree.find_all(exp.Table):
        name = ".".join(p for p in (t.catalog, t.db, t.name) if p).lower()
        if t.name.lower() in cte_names and not t.db:
            continue
        short = ".".join(name.split(".")[-2:]) if "." in name else f"gold.{name}"
        tables.append(short)
        if short not in allowed_tables:
            reasons.append(f"table not allowed: {name}")
    for c in tree.find_all(exp.Column):
        if c.name.lower() in PII_COLUMNS:
            reasons.append(f"raw PII column not allowed: {c.name}")

    if reasons:
        return Verdict(False, tables=sorted(set(tables)), reasons=sorted(set(reasons)))

    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.name)
        except (AttributeError, ValueError):
            current = None
    if current is None or current > max_rows:
        tree = tree.limit(max_rows)
    return Verdict(True, sql=tree.sql(dialect=DIALECT), tables=sorted(set(tables)))


def qualify(sql: str, mapping: dict[str, str]) -> str:
    """Rewrite logical table names (gold.x) to physical ones (temp view or catalog.gold.x)."""
    tree = sqlglot.parse_one(sql, read=DIALECT)
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        if t.name.lower() in cte_names and not t.db:
            continue
        key = f"{t.db or 'gold'}.{t.name}".lower()
        if key in mapping:
            parts = mapping[key].split(".")
            new = exp.to_table(mapping[key])
            new.set("alias", t.args.get("alias"))
            t.replace(new if len(parts) else t)
    return tree.sql(dialect=DIALECT)
