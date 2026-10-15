"""Command line / job entry points.

Locally:     paylake generate && paylake run
Databricks:  the bundle's job calls the paylake-<stage> wheel entry points, one task per stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from pathlib import Path

from .config import Settings

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _ctx(args=None):
    from .lake import get_lake
    from .spark import get_spark

    if args is not None and getattr(args, "catalog", None):
        os.environ["PAYLAKE_CATALOG"] = args.catalog
    settings = Settings()
    spark = get_spark(settings)
    return settings, spark, get_lake(spark, settings)


def _landing(args, settings: Settings) -> Path:
    return Path(args.landing).resolve() if getattr(args, "landing", None) else settings.landing_root


def _print(stage: str, result, seconds: float) -> None:
    payload = asdict(result) if is_dataclass(result) else result
    print(json.dumps({"stage": stage, "seconds": round(seconds, 1), "result": payload}, default=str))


def run_stage(stage: str, args) -> object:
    settings, spark, lake = _ctx(args)
    t0 = time.time()
    if stage == "bronze":
        from .bronze import ingest

        as_of = date.fromisoformat(args.as_of) if getattr(args, "as_of", None) else None
        res = ingest(spark, lake, _landing(args, settings), as_of=as_of)
    elif stage == "silver":
        from .silver import build

        res = build(spark, lake, settings)
    elif stage == "gold":
        from .gold import build

        res = build(spark, lake)
    elif stage == "train":
        from .ml import train

        r = train(spark, lake, settings, _landing(args, settings))
        res = {"run_id": r.run_id, "version": r.version, "promoted": r.promoted, "metrics": r.metrics}
    elif stage == "score":
        from .ml import score

        res = {"alerts": score(spark, lake, settings)}
    elif stage == "retention":
        res = retention(spark, lake, settings)
    else:
        raise SystemExit(f"unknown stage {stage}")
    _print(stage, res, time.time() - t0)
    return res


def retention(spark, lake, settings: Settings) -> dict:
    """Derived feature data has a shorter retention than payment records.

    Payment records themselves fall under statutory retention (GwG §8 AML record keeping,
    HGB/AO bookkeeping), so the job never deletes silver payments; it trims derived,
    non-regulatory tables and vacuums old file versions.
    """
    from pyspark.sql import functions as F

    if not lake.exists("gold.payment_features"):
        return {"deleted_before": None}
    latest = lake.read("gold.payment_features").agg(F.max("settlement_date")).first()[0]
    cutoff = latest - timedelta(days=settings.feature_retention_days)
    lake.delete("gold.payment_features", f"settlement_date < '{cutoff.isoformat()}'")
    return {"deleted_before": cutoff.isoformat()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="paylake")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write synthetic ISO 20022 traffic to the landing zone")
    g.add_argument("--out", default=None)
    g.add_argument("--days", type=int, default=30)
    g.add_argument("--start", default="2026-09-01")
    g.add_argument("--customers", type=int, default=4000)
    g.add_argument("--seed", type=int, default=7)

    for stage in ("bronze", "silver", "gold", "train", "score", "retention"):
        s = sub.add_parser(stage)
        s.add_argument("--landing", default=None)
        s.add_argument("--catalog", default=None)
        if stage == "bronze":
            s.add_argument("--as-of", dest="as_of", default=None, help="only pick up landing folders up to this date")

    r = sub.add_parser("run", help="bronze -> silver -> gold -> train -> score")
    r.add_argument("--landing", default=None)
    r.add_argument("--catalog", default=None)
    r.add_argument("--as-of", dest="as_of", default=None)
    r.add_argument("--skip-ml", action="store_true")

    a = sub.add_parser("ask", help="ask the analyst copilot a question")
    a.add_argument("question")

    ev = sub.add_parser("eval-copilot", help="validate reference SQL; with ANTHROPIC_API_KEY also run the execution-match eval")
    ev.add_argument("--reference-only", action="store_true")

    sv = sub.add_parser("serve", help="run the alerts API")
    sv.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    if args.cmd == "generate":
        from .datagen import generate

        out = Path(args.out) if args.out else Settings().landing_root
        t0 = time.time()
        m = generate(out, start=date.fromisoformat(args.start), days=args.days, n_customers=args.customers, seed=args.seed)
        _print("generate", m, time.time() - t0)
    elif args.cmd == "run":
        stages = ["bronze", "silver", "gold"] + ([] if args.skip_ml else ["train", "score"])
        for st in stages:
            run_stage(st, args)
    elif args.cmd == "ask":
        from .copilot import Copilot

        settings, spark, lake = _ctx()
        ans = Copilot.from_env(spark, lake).ask(args.question)
        print(json.dumps(ans.to_dict(), indent=2, default=str))
    elif args.cmd == "eval-copilot":
        from .copilot import Copilot
        from .copilot.evaluate import run as run_eval, validate_reference
        from .copilot.llm import StubLLM

        settings, spark, lake = _ctx()
        ref = validate_reference(Copilot(spark, lake, StubLLM(lambda q: "CANNOT_ANSWER"), settings))
        out = {"reference_queries": len(ref), "reference_ok": sum(r["allowed"] for r in ref)}
        if not args.reference_only and os.environ.get("ANTHROPIC_API_KEY"):
            out["eval"] = run_eval(Copilot.from_env(spark, lake, settings))
        elif not args.reference_only:
            out["eval"] = "skipped: set ANTHROPIC_API_KEY to run the model eval"
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("paylake.app:app", host="0.0.0.0", port=args.port)
    else:
        run_stage(args.cmd, args)


def _entry(stage: str):
    def run():
        main([stage, *sys.argv[1:]])
    return run


bronze_entry = _entry("bronze")
silver_entry = _entry("silver")
gold_entry = _entry("gold")
train_entry = _entry("train")
score_entry = _entry("score")

if __name__ == "__main__":
    main()
