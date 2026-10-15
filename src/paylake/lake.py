"""Storage adapter.

Pipeline code talks to a small `Lake` interface using logical names such as
"silver.payments". Two implementations:

* DatabricksLake - Unity Catalog tables (`<catalog>.silver.payments`), native Delta MERGE.
* LocalDeltaLake - real Delta tables (transaction log, versions, MERGE, time travel)
  written with delta-rs, so the full pipeline runs in CI without a workspace or
  Maven access. Data moves Spark <-> Arrow at the edges; fine for CI-sized data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from pyspark.sql import DataFrame, SparkSession

from .config import Settings


class Lake(ABC):
    def __init__(self, spark: SparkSession, settings: Settings):
        self.spark = spark
        self.settings = settings

    @abstractmethod
    def exists(self, name: str) -> bool: ...

    @abstractmethod
    def read(self, name: str, version: int | None = None) -> DataFrame: ...

    @abstractmethod
    def overwrite(self, df: DataFrame, name: str, partition_by: Sequence[str] | None = None) -> None: ...

    @abstractmethod
    def append(self, df: DataFrame, name: str, partition_by: Sequence[str] | None = None) -> None: ...

    @abstractmethod
    def merge(
        self,
        df: DataFrame,
        name: str,
        keys: Sequence[str],
        update_cols: Iterable[str] | None = None,
        insert: bool = True,
        partition_by: Sequence[str] | None = None,
    ) -> dict:
        """Upsert `df` into `name` on `keys`.

        update_cols=None  -> matched rows are left untouched (insert-only, idempotent ingest)
        update_cols=[...] -> matched rows get those columns from the source
        insert=False      -> unmatched source rows are ignored (pure update)
        """

    @abstractmethod
    def delete(self, name: str, predicate: str) -> None: ...

    @abstractmethod
    def version(self, name: str) -> int: ...

    @abstractmethod
    def history(self, name: str) -> list[dict]: ...

    def read_or_empty(self, name: str, schema) -> DataFrame:
        return self.read(name) if self.exists(name) else self.spark.createDataFrame([], schema)


# --------------------------------------------------------------------------- local
class LocalDeltaLake(Lake):
    def path(self, name: str) -> Path:
        layer, table = name.split(".", 1)
        return self.settings.lake_root / layer / table

    def exists(self, name: str) -> bool:
        return (self.path(name) / "_delta_log").exists()

    def _arrow(self, df: DataFrame):
        return df.toArrow()

    def read(self, name: str, version: int | None = None) -> DataFrame:
        from deltalake import DeltaTable

        dt = DeltaTable(str(self.path(name)), version=version) if version is not None else DeltaTable(str(self.path(name)))
        table = dt.to_pyarrow_table()
        if table.num_rows == 0:
            from pyspark.sql.pandas.types import from_arrow_schema

            return self.spark.createDataFrame([], from_arrow_schema(table.schema))
        return self.spark.createDataFrame(table)

    def _write(self, df: DataFrame, name: str, mode: str, partition_by) -> None:
        from deltalake import write_deltalake

        self.path(name).parent.mkdir(parents=True, exist_ok=True)
        write_deltalake(
            str(self.path(name)),
            self._arrow(df),
            mode=mode,
            partition_by=list(partition_by) if partition_by else None,
            schema_mode="overwrite" if mode == "overwrite" else None,
        )

    def overwrite(self, df, name, partition_by=None):
        self._write(df, name, "overwrite", partition_by)

    def append(self, df, name, partition_by=None):
        self._write(df, name, "append", partition_by)

    def merge(self, df, name, keys, update_cols=None, insert=True, partition_by=None) -> dict:
        from deltalake import DeltaTable

        if not self.exists(name):
            if not insert:
                return {"num_target_rows_inserted": 0, "num_target_rows_updated": 0}
            self._write(df, name, "overwrite", partition_by)
            n = df.count()
            return {"num_target_rows_inserted": n, "num_target_rows_updated": 0}

        dt = DeltaTable(str(self.path(name)))
        predicate = " AND ".join(f"t.{k} = s.{k}" for k in keys)
        m = dt.merge(self._arrow(df), predicate=predicate, source_alias="s", target_alias="t")
        if update_cols:
            m = m.when_matched_update(updates={c: f"s.{c}" for c in update_cols})
        if insert:
            m = m.when_not_matched_insert_all()
        metrics = m.execute()
        return {
            "num_target_rows_inserted": metrics.get("num_target_rows_inserted", 0),
            "num_target_rows_updated": metrics.get("num_target_rows_updated", 0),
        }

    def delete(self, name, predicate):
        from deltalake import DeltaTable

        DeltaTable(str(self.path(name))).delete(predicate)

    def version(self, name):
        from deltalake import DeltaTable

        return DeltaTable(str(self.path(name))).version()

    def history(self, name):
        from deltalake import DeltaTable

        return DeltaTable(str(self.path(name))).history()

    def vacuum(self, name: str, retention_hours: int = 168) -> list[str]:
        from deltalake import DeltaTable

        return DeltaTable(str(self.path(name))).vacuum(
            retention_hours=retention_hours, dry_run=False, enforce_retention_duration=retention_hours >= 168
        )


# ---------------------------------------------------------------------- databricks
class DatabricksLake(Lake):  # pragma: no cover - exercised on a workspace, not in local CI
    def fq(self, name: str) -> str:
        return f"{self.settings.catalog}.{name}"

    def exists(self, name):
        return self.spark.catalog.tableExists(self.fq(name))

    def read(self, name, version=None):
        if version is None:
            return self.spark.table(self.fq(name))
        return self.spark.sql(f"SELECT * FROM {self.fq(name)} VERSION AS OF {int(version)}")

    def _writer(self, df, mode, partition_by):
        w = df.write.format("delta").mode(mode)
        if partition_by:
            w = w.partitionBy(*partition_by)
        return w

    def overwrite(self, df, name, partition_by=None):
        self._writer(df, "overwrite", partition_by).option("overwriteSchema", "true").saveAsTable(self.fq(name))

    def append(self, df, name, partition_by=None):
        self._writer(df, "append", partition_by).saveAsTable(self.fq(name))

    def merge(self, df, name, keys, update_cols=None, insert=True, partition_by=None):
        from delta.tables import DeltaTable

        if not self.exists(name):
            if insert:
                self.overwrite(df, name, partition_by)
            return {"num_target_rows_inserted": df.count() if insert else 0, "num_target_rows_updated": 0}
        cond = " AND ".join(f"t.{k} = s.{k}" for k in keys)
        m = DeltaTable.forName(self.spark, self.fq(name)).alias("t").merge(df.alias("s"), cond)
        if update_cols:
            m = m.whenMatchedUpdate(set={c: f"s.{c}" for c in update_cols})
        if insert:
            m = m.whenNotMatchedInsertAll()
        m.execute()
        last = self.history(name)[0].get("operationMetrics", {})
        return {
            "num_target_rows_inserted": int(last.get("numTargetRowsInserted", 0)),
            "num_target_rows_updated": int(last.get("numTargetRowsUpdated", 0)),
        }

    def delete(self, name, predicate):
        self.spark.sql(f"DELETE FROM {self.fq(name)} WHERE {predicate}")

    def version(self, name):
        return int(self.spark.sql(f"DESCRIBE HISTORY {self.fq(name)} LIMIT 1").first()["version"])

    def history(self, name):
        return [r.asDict() for r in self.spark.sql(f"DESCRIBE HISTORY {self.fq(name)}").collect()]


def get_lake(spark: SparkSession, settings: Settings | None = None) -> Lake:
    settings = settings or Settings()
    return DatabricksLake(spark, settings) if settings.is_databricks else LocalDeltaLake(spark, settings)
