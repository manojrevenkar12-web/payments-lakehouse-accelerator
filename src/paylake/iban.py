"""IBAN / BIC helpers (ISO 13616 mod-97, ISO 9362)."""
from __future__ import annotations

import re

# country -> total IBAN length and BBAN layout used by the generator ("n" digit, "a" upper letter, "c" alnum)
IBAN_SPECS = {
    "DE": (22, "8n10n"),
    "AT": (20, "5n11n"),
    "NL": (18, "4a10n"),
    "FR": (27, "5n5n11c2n"),
    "ES": (24, "4n4n2n10n"),
    "IT": (27, "1a5n5n12c"),
    "BE": (16, "3n7n2n"),
    "PL": (28, "8n16n"),
}

BIC_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")


def _to_digits(s: str) -> str:
    return "".join(str(int(ch, 36)) for ch in s)


def iban_check_digits(country: str, bban: str) -> str:
    rearranged = _to_digits(bban + country + "00")
    return f"{98 - int(rearranged) % 97:02d}"


def make_iban(country: str, bban: str) -> str:
    return f"{country}{iban_check_digits(country, bban)}{bban}"


def is_valid_iban(iban: str | None) -> bool:
    if not iban:
        return False
    iban = iban.replace(" ", "").upper()
    spec = IBAN_SPECS.get(iban[:2])
    if spec is None:
        # unknown country: fall back to the generic length bounds and checksum
        if not 15 <= len(iban) <= 34:
            return False
    elif len(iban) != spec[0]:
        return False
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", iban):
        return False
    return int(_to_digits(iban[4:] + iban[:4])) % 97 == 1


def is_valid_bic(bic: str | None) -> bool:
    return bool(bic) and bool(BIC_RE.match(bic))
