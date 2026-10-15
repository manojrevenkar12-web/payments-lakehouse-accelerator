from datetime import date

from lxml import etree

from paylake.datagen import Generator, _corrupt_iban, generate
from paylake.iban import is_valid_bic, is_valid_iban, make_iban


def test_known_ibans():
    assert is_valid_iban("DE89370400440532013000")
    assert is_valid_iban("de89 3704 0044 0532 0130 00")
    assert not is_valid_iban("DE89370400440532013001")  # checksum
    assert not is_valid_iban("DE8937040044053201300")   # length
    assert not is_valid_iban(None)


def test_generated_ibans_are_valid_and_corruption_is_detected():
    g = Generator(seed=1, n_customers=50, n_counterparties=200, n_mules=5)
    ibans = [c.acct.iban for c in g.customers] + [c.iban for c in g.counterparties] + [m.iban for m in g.mules]
    assert all(is_valid_iban(i) for i in ibans)
    assert not any(is_valid_iban(_corrupt_iban(g.rng, i)) for i in ibans[:200])
    assert make_iban("DE", "370400440532013000") == "DE89370400440532013000"


def test_bic_format():
    assert is_valid_bic("PLHBDEFFXXX") and is_valid_bic("POLDNL2A")
    assert not is_valid_bic("PLHB-DEFF") and not is_valid_bic(None)


def test_generator_is_deterministic_and_well_formed(tmp_path):
    m1 = generate(tmp_path / "a", start=date(2026, 9, 1), days=4, n_customers=300, seed=3)
    m2 = generate(tmp_path / "b", start=date(2026, 9, 1), days=4, n_customers=300, seed=3)
    assert m1 == m2
    files = sorted((tmp_path / "a" / "pacs008").rglob("*.xml"))
    parsed, broken = 0, 0
    for f in files:
        try:
            etree.parse(str(f))
            parsed += 1
        except etree.XMLSyntaxError:
            broken += 1
    assert broken == m1["truncated_files"] == 1
    assert parsed == m1["files_pacs008"] + m1["resent_files"]
    cases = (tmp_path / "a" / "cases" / "case_outcomes.csv").read_text().splitlines()
    assert len(cases) - 1 == m1["fraud_transactions"] > 0
