"""Analyst copilot: natural-language questions -> guarded Spark SQL over PII-free gold tables.

Flow: question -> LLM proposes one SQL query -> guardrails (read-only, allowlisted tables,
no PII columns, denylisted functions, enforced LIMIT) -> logical->physical table mapping ->
Spark executes -> rows + the exact SQL returned, so every answer is auditable.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field
from importlib import resources

import yaml

from ..config import Settings
from . import guardrails
from .llm import LLM, AnthropicLLM

SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.+?)```", re.S | re.I)


def load_catalog() -> dict:
    return yaml.safe_load(resources.files(__package__).joinpath("catalog.yaml").read_text(encoding="utf-8"))["tables"]


def system_prompt(catalog: dict) -> str:
    lines = [
        "You translate questions from payment-operations and fraud analysts into ONE Spark SQL (Databricks) query.",
        "Use only these tables and columns:",
    ]
    for t, spec in catalog.items():
        lines.append(f"\n{t}: {spec['description']}")
        lines += [f"  - {c}: {d}" for c, d in spec["columns"].items()]
    lines += [
        "",
        "Rules:",
        "- Return exactly one read-only SELECT in a ```sql code block, nothing else.",
        "- Always use the fully qualified names shown above (gold.<table>).",
        "- Never ask for customer names, IBANs or remittance text; they are not available.",
        "- If the question cannot be answered from these tables, reply: CANNOT_ANSWER: <short reason>",
    ]
    return "\n".join(lines)


@dataclass
class Answer:
    question: str
    model: str
    status: str  # answered | refused_by_model | blocked_by_guardrail | error
    sql_generated: str | None = None
    sql_executed: str | None = None
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Copilot:
    def __init__(self, spark, lake, llm: LLM, settings: Settings | None = None, max_rows: int = 200):
        self.spark, self.lake, self.llm = spark, lake, llm
        self.settings = settings or Settings()
        self.max_rows = max_rows
        self.catalog = load_catalog()
        self.allowed = set(self.catalog)
        self.system = system_prompt(self.catalog)
        self.mapping = self._register_tables()

    @classmethod
    def from_env(cls, spark, lake, settings: Settings | None = None) -> "Copilot":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("Set ANTHROPIC_API_KEY to use the copilot (or inject another LLM client).")
        return cls(spark, lake, AnthropicLLM(key), settings)

    def _register_tables(self) -> dict[str, str]:
        mapping = {}
        for logical in self.catalog:
            if self.settings.is_databricks:  # pragma: no cover
                mapping[logical] = f"{self.settings.catalog}.{logical}"
            elif self.lake.exists(logical):
                view = logical.replace(".", "_")
                self.lake.read(logical).createOrReplaceTempView(view)
                mapping[logical] = view
        return mapping

    def run_sql(self, sql: str) -> tuple[guardrails.Verdict, list, list]:
        verdict = guardrails.check(sql, self.allowed, self.max_rows)
        if not verdict.allowed:
            return verdict, [], []
        physical = guardrails.qualify(verdict.sql, self.mapping)
        df = self.spark.sql(physical)
        rows = [list(r) for r in df.collect()]
        verdict.sql = physical
        return verdict, df.columns, rows

    def ask(self, question: str) -> Answer:
        t0 = time.time()
        ans = Answer(question=question, model=getattr(self.llm, "model", "unknown"), status="error")
        try:
            raw = self.llm.complete(self.system, question)
        except Exception as e:  # network, auth, rate limit
            ans.reasons = [f"llm error: {e}"]
            ans.latency_ms = int((time.time() - t0) * 1000)
            return ans
        if raw.strip().upper().startswith("CANNOT_ANSWER"):
            ans.status, ans.reasons = "refused_by_model", [raw.strip()]
        else:
            m = SQL_BLOCK.search(raw)
            sql = (m.group(1) if m else raw).strip().rstrip(";")
            ans.sql_generated = sql
            try:
                verdict, cols, rows = self.run_sql(sql)
                if not verdict.allowed:
                    ans.status, ans.reasons = "blocked_by_guardrail", verdict.reasons
                else:
                    ans.status, ans.sql_executed, ans.columns, ans.rows = "answered", verdict.sql, cols, rows
            except Exception as e:
                ans.status, ans.reasons = "error", [f"execution error: {str(e).splitlines()[0]}"]
        ans.latency_ms = int((time.time() - t0) * 1000)
        return ans
