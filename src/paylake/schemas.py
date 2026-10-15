"""Explicit schemas. Production ingestion never relies on XML schema inference:
inference is slow, non-deterministic across batches and silently turns amounts into doubles."""
from pyspark.sql.types import (
    ArrayType, DateType, DecimalType, LongType, StringType, StructField, StructType, TimestampType,
)

AMOUNT = DecimalType(18, 2)


def _s(*fields):
    return StructType(list(fields))


def _f(name, dtype):
    return StructField(name, dtype, True)


FIN_INSTN = _s(_f("FinInstnId", _s(_f("BICFI", StringType()))))
ACCOUNT = _s(_f("Id", _s(_f("IBAN", StringType()))))
PARTY = _s(_f("Nm", StringType()))

PACS008_TX = _s(
    _f("PmtId", _s(_f("InstrId", StringType()), _f("EndToEndId", StringType()), _f("UETR", StringType()))),
    _f("PmtTpInf", _s(_f("SvcLvl", _s(_f("Cd", StringType()))))),
    _f("IntrBkSttlmAmt", _s(_f("_VALUE", AMOUNT), _f("_Ccy", StringType()))),
    _f("IntrBkSttlmDt", DateType()),
    _f("AccptncDtTm", TimestampType()),
    _f("ChrgBr", StringType()),
    _f("Dbtr", PARTY),
    _f("DbtrAcct", ACCOUNT),
    _f("DbtrAgt", FIN_INSTN),
    _f("CdtrAgt", FIN_INSTN),
    _f("Cdtr", PARTY),
    _f("CdtrAcct", ACCOUNT),
    _f("RmtInf", _s(_f("Ustrd", StringType()))),
)

PACS008 = _s(
    _f("GrpHdr", _s(_f("MsgId", StringType()), _f("CreDtTm", TimestampType()), _f("NbOfTxs", LongType()))),
    _f("CdtTrfTxInf", ArrayType(PACS008_TX)),
    _f("_corrupt_record", StringType()),
)

PACS002 = _s(
    _f("GrpHdr", _s(_f("MsgId", StringType()), _f("CreDtTm", TimestampType()))),
    _f("OrgnlGrpInfAndSts", _s(_f("OrgnlMsgId", StringType()), _f("OrgnlMsgNmId", StringType()))),
    _f("TxInfAndSts", ArrayType(_s(
        _f("OrgnlEndToEndId", StringType()),
        _f("OrgnlUETR", StringType()),
        _f("TxSts", StringType()),
        _f("StsRsnInf", _s(_f("Rsn", _s(_f("Cd", StringType()))))),
    ))),
    _f("_corrupt_record", StringType()),
)

ROW_TAGS = {"pacs008": "FIToFICstmrCdtTrf", "pacs002": "FIToFIPmtStsRpt"}
SCHEMAS = {"pacs008": PACS008, "pacs002": PACS002}

INGEST_LOG = _s(
    _f("file_path", StringType()), _f("file_size", LongType()), _f("message_type", StringType()),
    _f("batch_id", LongType()), _f("ingested_at", TimestampType()), _f("records", LongType()),
    _f("corrupt_records", LongType()),
)
