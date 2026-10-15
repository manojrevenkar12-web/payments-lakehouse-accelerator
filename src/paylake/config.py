"""Runtime configuration.

One code path, two runtimes:
  * local      - open-source Spark + delta-rs tables under a local lake root (CI, laptops)
  * databricks - Unity Catalog three-level names (catalog.schema.table), native Delta
The runtime is detected from DATABRICKS_RUNTIME_VERSION unless PAYLAKE_RUNTIME is set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _detect_runtime() -> str:
    explicit = os.environ.get("PAYLAKE_RUNTIME")
    if explicit:
        return explicit
    return "databricks" if os.environ.get("DATABRICKS_RUNTIME_VERSION") else "local"


def _databricks_secret(scope: str, key: str) -> str | None:  # pragma: no cover
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        return DBUtils(SparkSession.builder.getOrCreate()).secrets.get(scope, key)
    except Exception:
        return None


@dataclass
class Settings:
    runtime: str = field(default_factory=_detect_runtime)
    # local runtime
    lake_root: Path = field(default_factory=lambda: Path(os.environ.get("PAYLAKE_LAKE_ROOT", "./lake")).resolve())
    landing_root: Path = field(default_factory=lambda: Path(os.environ.get("PAYLAKE_LANDING_ROOT", "./landing")).resolve())
    mlflow_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", ""))
    # databricks runtime (Unity Catalog)
    catalog: str = field(default_factory=lambda: os.environ.get("PAYLAKE_CATALOG", "payments_dev"))
    # pseudonymisation key: a Databricks secret scope in prod, an env var locally
    hmac_key: str = field(default_factory=lambda: os.environ.get("PAYLAKE_HMAC_KEY", "local-dev-key-not-for-production"))
    # the bank whose outgoing SEPA flows we monitor (fictional BIC)
    home_bic: str = "PLHBDEFFXXX"
    # analysts' daily review capacity as a share of payment volume
    alert_budget: float = 0.01
    # derived (non-regulatory) data retention for the retention job
    feature_retention_days: int = 400

    def __post_init__(self) -> None:
        if self.is_databricks and "PAYLAKE_HMAC_KEY" not in os.environ:  # pragma: no cover
            secret = _databricks_secret("paylake", "hmac_key")
            if not secret:
                raise RuntimeError("secret scope 'paylake' / key 'hmac_key' is required on Databricks")
            self.hmac_key = secret
        if not self.mlflow_uri:
            self.mlflow_uri = (
                "databricks" if self.runtime == "databricks" else f"sqlite:///{(self.lake_root / 'mlflow.db').as_posix()}"
            )

    @property
    def is_databricks(self) -> bool:
        return self.runtime == "databricks"
