import os
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("PAYLAKE_SHUFFLE_PARTITIONS", "4")
os.environ.setdefault("PAYLAKE_SPARK_MASTER", "local[2]")


@pytest.fixture(scope="session")
def spark():
    from paylake.spark import get_spark

    return get_spark()


@pytest.fixture(scope="session")
def built(tmp_path_factory, spark):
    """A small end-to-end run shared by the integration tests:
    two bronze batches (incremental), then silver, gold, train and score."""
    from paylake import bronze, gold, ml, silver
    from paylake.config import Settings
    from paylake.datagen import generate
    from paylake.lake import get_lake

    root = tmp_path_factory.mktemp("paylake")
    settings = Settings(runtime="local", lake_root=root / "lake", landing_root=root / "landing")
    manifest = generate(settings.landing_root, start=date(2026, 9, 1), days=10, n_customers=1200, seed=11)
    lake = get_lake(spark, settings)

    b1 = bronze.ingest(spark, lake, settings.landing_root, as_of=date(2026, 9, 6))
    s1 = silver.build(spark, lake, settings)
    b2 = bronze.ingest(spark, lake, settings.landing_root)
    s2 = silver.build(spark, lake, settings)
    b3 = bronze.ingest(spark, lake, settings.landing_root)
    s3 = silver.build(spark, lake, settings)
    g = gold.build(spark, lake)
    t = ml.train(spark, lake, settings, settings.landing_root)
    n_alerts = ml.score(spark, lake, settings)
    return {
        "settings": settings, "lake": lake, "manifest": manifest, "bronze": [b1, b2, b3], "silver": [s1, s2, s3],
        "gold": g, "train": t, "alerts": n_alerts,
    }
