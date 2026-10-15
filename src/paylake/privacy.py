"""Pseudonymisation and IBAN checks as vectorised pandas UDFs.

Analytics, ML features, alerts and the copilot work on keyed HMAC-SHA256 tokens (`*_ref`),
never on raw IBANs or names. The key lives in a Databricks secret scope in production, so
tokens are stable across runs (joins work) but cannot be reversed or recomputed without it.
"""
from __future__ import annotations

import hashlib
import hmac

import pandas as pd
from pyspark.sql import Column
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType, StringType

from .iban import is_valid_iban

PII_COLUMNS = ("debtor_name", "debtor_iban", "creditor_name", "creditor_iban", "remittance_info")


def token(value: str | None, key: str) -> str | None:
    if value is None:
        return None
    return hmac.new(key.encode(), value.strip().upper().encode(), hashlib.sha256).hexdigest()[:24]


def pseudonymise(col: Column, key: str) -> Column:
    @pandas_udf(StringType())
    def _tok(s: pd.Series) -> pd.Series:
        return s.map(lambda v: token(v, key) if isinstance(v, str) else None)

    return _tok(col)


@pandas_udf(BooleanType())
def iban_checksum_ok(s: pd.Series) -> pd.Series:
    return s.map(lambda v: is_valid_iban(v) if isinstance(v, str) else False)


def iban_country(col: Column) -> Column:
    return F.upper(F.substring(col, 1, 2))
