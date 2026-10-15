import pytest

from paylake.copilot import Copilot, guardrails
from paylake.copilot.evaluate import load_eval_set, run, validate_reference
from paylake.copilot.llm import StubLLM

ALLOWED = {"gold.payments_analytics", "gold.daily_kpis", "gold.reject_reasons", "gold.dq_summary", "gold.alerts"}


@pytest.mark.parametrize("sql,reason", [
    ("DROP TABLE gold.alerts", "forbidden operation"),
    ("DELETE FROM gold.alerts", "forbidden operation"),
    ("UPDATE gold.alerts SET review_status = 'CLOSED'", "forbidden operation"),
    ("INSERT INTO gold.alerts SELECT * FROM gold.alerts", "forbidden operation"),
    ("SELECT 1; DROP TABLE gold.alerts", "exactly one statement"),
    ("SELECT debtor_iban FROM gold.payments_analytics", "raw PII"),
    ("SELECT p.creditor_name FROM gold.payments_analytics p", "raw PII"),
    ("SELECT * FROM silver.payments", "table not allowed"),
    ("SELECT * FROM payments_dev.silver.payments", "table not allowed"),
    ("SELECT * FROM gold.payment_features", "table not allowed"),
    ("SELECT reflect('java.lang.Runtime', 'getRuntime') FROM gold.alerts", "forbidden function"),
    ("SELECT * FROM read_files('/Volumes/x')", "forbidden function"),
    ("SELEC uetr FRM gold.alerts", "unparseable"),
])
def test_guardrail_blocks(sql, reason):
    v = guardrails.check(sql, ALLOWED)
    assert not v.allowed
    assert any(reason in r for r in v.reasons), v.reasons


def test_guardrail_allows_and_enforces_limit():
    v = guardrails.check("WITH x AS (SELECT * FROM gold.alerts) SELECT uetr FROM x", ALLOWED)
    assert v.allowed and v.sql.endswith("LIMIT 200")
    v = guardrails.check("SELECT uetr FROM gold.alerts LIMIT 100000", ALLOWED)
    assert v.allowed and v.sql.endswith("LIMIT 200")
    v = guardrails.check("SELECT uetr FROM gold.alerts LIMIT 10", ALLOWED)
    assert v.sql.endswith("LIMIT 10")


def test_reference_queries_pass_guardrails_and_execute(built, spark):
    cp = Copilot(spark, built["lake"], StubLLM(lambda q: "CANNOT_ANSWER"), built["settings"])
    ref = validate_reference(cp)
    assert all(r["allowed"] for r in ref), [r for r in ref if not r["allowed"]]
    assert all(r["rows"] > 0 for r in ref if r["id"] not in ("reject_count_by_reason",))


def _oracle(spec, leak_pii: bool):
    gold = {i["question"]: i["gold_sql"] for i in spec["answerable"]}

    def respond(q):
        if q in gold:
            return f"```sql\n{gold[q]}\n```"
        # a misbehaving model that tries to answer must-refuse questions anyway
        return "```sql\nSELECT debtor_iban, debtor_name FROM gold.payments_analytics\n```" if leak_pii else "CANNOT_ANSWER: no PII"
    return respond


def test_eval_harness_scores_a_perfect_model(built, spark):
    spec = load_eval_set()
    cp = Copilot(spark, built["lake"], StubLLM(_oracle(spec, leak_pii=True)), built["settings"])
    res = run(cp)
    assert res["execution_accuracy"] == 1.0
    assert res["refusal_accuracy"] == 1.0          # the guardrail caught every PII attempt
    assert res["unsafe_answers"] == 0 and res["blocked_by_guardrail"] == len(spec["must_refuse"])


def test_eval_harness_detects_wrong_answers(built, spark):
    cp = Copilot(spark, built["lake"], StubLLM(lambda q: "```sql\nSELECT 42\n```"), built["settings"])
    res = run(cp)
    assert res["execution_accuracy"] < 0.2
    assert res["unsafe_answers"] == 4             # answered must-refuse questions (harmless, but not a refusal)
