"""Reference data: a fictional BIC directory (stand-in for the SWIFTRef / EBA STEP2 reachability file)."""
from __future__ import annotations

# (bic, bank_name, country, bank_code_for_bban)
BANKS: list[tuple[str, str, str, str]] = [
    ("PLHBDEFFXXX", "Payments Lakehouse Bank (home)", "DE", "50010517"),
    ("HSPKDEHHXXX", "Hanse Sparkasse", "DE", "20050550"),
    ("RHVBDEMMXXX", "Rhein Volksbank", "DE", "70020270"),
    ("BLNBDEBBXXX", "Berliner Handelsbank", "DE", "10070000"),
    ("SDWBDESSXXX", "Suedwest Bank", "DE", "60050101"),
    ("NRDLDE2HXXX", "Nordland Direktbank", "DE", "25010030"),
    ("ALPBATWWXXX", "Alpen Bank", "AT", "12000"),
    ("DNBKATWWXXX", "Donau Kredit", "AT", "20111"),
    ("POLDNL2AXXX", "Polder Bank", "NL", "PLDR"),
    ("GRCHNL2UXXX", "Grachten Bank", "NL", "GRCH"),
    ("SEINFRPPXXX", "Banque de la Seine", "FR", "30004"),
    ("LOIRFRPPXXX", "Credit de Loire", "FR", "30066"),
    ("MESEESMMXXX", "Banco Meseta", "ES", "2100"),
    ("ATLNESBBXXX", "Banco Atlantico Norte", "ES", "0081"),
    ("LAGOITMMXXX", "Banca dei Laghi", "IT", "03069"),
    ("TEVRITRRXXX", "Banca del Tevere", "IT", "02008"),
    ("FLNDBEBBXXX", "Vlaanderen Bank", "BE", "001"),
    ("WSLAPLPWXXX", "Wisla Bank", "PL", "10901014"),
    # neobank that onboards accounts quickly; useful for mule realism
    ("FSTKDEB2XXX", "Fastkonto Neobank", "DE", "10011001"),
]

BIC_DIRECTORY_COLUMNS = ["bic", "bank_name", "bank_country"]


def bic_directory_rows() -> list[tuple[str, str, str]]:
    return [(b, n, c) for b, n, c, _ in BANKS]
