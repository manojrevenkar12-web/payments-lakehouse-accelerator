"""Synthetic ISO 20022 traffic generator.

Simulates a German bank's outgoing SEPA credit transfers (pacs.008.001.08) and the
clearing house status reports (pacs.002.001.10), with:

* customer behaviour   - per-customer rates, amount profiles, active hours, habitual payees
* popular merchants    - legitimately receive from many customers (confounds naive fan-in rules)
* legit confounders    - one-off large payments to new payees, night-shift workers
* fraud typologies     - APP scam / account takeover, mule fan-in (fake-shop victims), structuring
* data-quality defects - bad IBAN checksums, missing agent BIC, non-EUR, non-positive amounts,
                         far-future settlement dates, missing UETR, re-sent files, a truncated file

Ground truth is written separately as investigation outcomes (cases/case_outcomes.csv),
the way confirmed fraud reaches a bank: from case management, after the fact.
Everything is fictional and seeded.
"""
from __future__ import annotations

import csv
import json
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .iban import IBAN_SPECS, make_iban
from .reference import BANKS

FIRST = ["Anna", "Lukas", "Mia", "Leon", "Emma", "Paul", "Sofia", "Jonas", "Lea", "Felix", "Hannah", "Noah",
         "Marie", "Elias", "Lina", "Ben", "Clara", "Finn", "Ida", "Emil", "Aylin", "Mehmet", "Olga", "Piotr",
         "Lucia", "Marco", "Chloe", "Hugo", "Sanne", "Daan", "Priya", "Arjun", "Fatima", "Yusuf", "Katarina", "Jan"]
LAST = ["Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Hoffmann", "Koch",
        "Richter", "Klein", "Wolf", "Neumann", "Schwarz", "Braun", "Zimmermann", "Krueger", "Hartmann", "Lange",
        "Yilmaz", "Kowalski", "Rossi", "Dubois", "de Vries", "Garcia", "Novak", "Sharma", "Popescu", "Jansen"]
COMPANY = ["Stadtwerke", "Energie", "Wohnbau", "Versand", "Elektro", "Moebel", "Reisen", "Versicherung", "Telekom",
           "Fitness", "Autohaus", "Apotheke", "Buchhandel", "Logistik", "Immobilien", "Handwerk", "Bio Markt"]
CITIES = ["Berlin", "Hamburg", "Muenchen", "Koeln", "Frankfurt", "Leipzig", "Dresden", "Wien", "Amsterdam", "Lyon"]
LEGAL = ["GmbH", "AG", "GmbH & Co. KG", "e.K.", "B.V.", "SAS"]

COUNTRY_WEIGHTS = {"DE": 0.70, "AT": 0.06, "NL": 0.06, "FR": 0.06, "ES": 0.04, "IT": 0.04, "BE": 0.02, "PL": 0.02}
HOME = BANKS[0]
FAST_NEOBANK = next(b for b in BANKS if b[0] == "FSTKDEB2XXX")


@dataclass
class Account:
    iban: str
    bic: str
    name: str
    country: str
    popularity: float = 1.0
    is_mule: bool = False


@dataclass
class Customer:
    acct: Account
    rate: float
    mu: float
    sigma: float
    hour_mode: float
    payees: list = field(default_factory=list)


@dataclass
class Tx:
    ts: datetime
    debtor: Customer
    creditor: Account
    amount: float
    remittance: str
    typology: str | None = None
    defects: tuple = ()
    uetr: str = ""


class Generator:
    def __init__(self, seed: int = 7, n_customers: int = 4000, n_counterparties: int = 6000, n_mules: int = 60):
        self.rng = np.random.default_rng(seed)
        self._acct_seq = 0
        self.banks_by_country: dict[str, list] = {}
        for b in BANKS[1:]:
            self.banks_by_country.setdefault(b[2], []).append(b)
        self.counterparties = [self._counterparty() for _ in range(n_counterparties)]
        pop = np.array([c.popularity for c in self.counterparties])
        self.cp_weights = pop / pop.sum()
        self.customers = [self._customer() for _ in range(n_customers)]
        self.mules = [self._mule() for _ in range(n_mules)]
        self._mule_cursor = 0

    # ------------------------------------------------------------------ parties
    def _bban(self, country: str, bank_code: str) -> str:
        layout = IBAN_SPECS[country][1]
        out, i = [], 0
        parts = []
        while i < len(layout):
            j = i
            while layout[j].isdigit():
                j += 1
            parts.append((int(layout[i:j]), layout[j]))
            i = j + 1
        first = True
        for n, kind in parts:
            if first and len(bank_code) == n:
                out.append(bank_code)
            else:
                alphabet = {"n": "0123456789", "a": "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "c": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"}[kind]
                out.append("".join(self.rng.choice(list(alphabet), n)))
            first = False
        return "".join(out)

    def _person(self) -> str:
        return f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"

    def _account(self, country: str, bank, name: str, **kw) -> Account:
        self._acct_seq += 1
        return Account(make_iban(country, self._bban(country, bank[3])), bank[0], name, country, **kw)

    def _counterparty(self) -> Account:
        countries, w = zip(*COUNTRY_WEIGHTS.items())
        country = str(self.rng.choice(countries, p=np.array(w) / sum(w)))
        banks = self.banks_by_country[country]
        bank = banks[self.rng.integers(len(banks))]
        is_company = self.rng.random() < 0.35
        name = (f"{self.rng.choice(COMPANY)} {self.rng.choice(CITIES)} {self.rng.choice(LEGAL)}" if is_company else self._person())
        # Zipf-like popularity: a few utilities/landlords/retailers are paid by thousands of customers
        popularity = float(self.rng.pareto(1.2) + 1) * (8 if is_company else 1)
        return self._account(country, bank, name, popularity=popularity)

    def _customer(self) -> Customer:
        acct = self._account("DE", HOME, self._person())
        night = self.rng.random() < 0.03
        c = Customer(
            acct=acct,
            rate=float(self.rng.gamma(2.0, 0.6)),
            mu=float(self.rng.normal(np.log(140), 0.7)),
            sigma=float(self.rng.uniform(0.5, 0.95)),
            hour_mode=float(self.rng.normal(2.0, 1.0) if night else self.rng.normal(13.5, 2.5)),
        )
        k = int(self.rng.integers(3, 13))
        c.payees = [self.counterparties[i] for i in self.rng.choice(len(self.counterparties), k, replace=False, p=self.cp_weights)]
        return c

    def _mule(self) -> Account:
        if self.rng.random() < 0.6:
            bank, country = FAST_NEOBANK, "DE"
        else:
            country = str(self.rng.choice(["DE", "NL", "ES", "IT", "PL"]))
            banks = self.banks_by_country[country]
            bank = banks[self.rng.integers(len(banks))]
        return self._account(country, bank, self._person(), is_mule=True)

    def _next_mule(self) -> Account:
        m = self.mules[self._mule_cursor % len(self.mules)]
        self._mule_cursor += 1
        return m

    def _uetr(self) -> str:
        return str(uuid.UUID(bytes=self.rng.bytes(16), version=4))

    def _hour(self, mode: float) -> float:
        return float(self.rng.normal(mode, 2.2)) % 24

    def _amount(self, c: Customer) -> float:
        return round(float(max(1.0, np.exp(self.rng.normal(c.mu, c.sigma)))), 2)

    @staticmethod
    def _remit(rng, kind: str = "normal") -> str:
        if kind == "normal":
            return str(rng.choice(["Rechnung", "Invoice", "Miete", "Beitrag", "Abschlag", "Bestellung", "Danke"])) + f" {rng.integers(1000, 99999)}"
        return str(rng.choice(["Rechnung", "Bestellung", "Kaution", "Zahlung", "Ueberweisung"])) + f" {rng.integers(1000, 99999)}"

    # ------------------------------------------------------------------ traffic
    def simulate(self, start: date, days: int) -> list[Tx]:
        rng = self.rng
        txs: list[Tx] = []
        rates = np.array([c.rate for c in self.customers])
        for d in range(days):
            day = start + timedelta(days=d)
            base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            weekend = day.weekday() >= 5
            counts = rng.poisson(rates * (0.55 if weekend else 1.0))
            for ci in np.nonzero(counts)[0]:
                c = self.customers[ci]
                for _ in range(counts[ci]):
                    if rng.random() < 0.9:
                        payee = c.payees[rng.integers(len(c.payees))]
                    else:
                        payee = self.counterparties[rng.choice(len(self.counterparties), p=self.cp_weights)]
                        if rng.random() < 0.3:
                            c.payees.append(payee)
                    ts = base + timedelta(hours=self._hour(c.hour_mode))
                    txs.append(Tx(ts, c, payee, self._amount(c), self._remit(rng)))
            # legit confounder: one-off large payment to a new payee (car, rent deposit, holiday)
            for ci in rng.choice(len(self.customers), int(len(self.customers) * 0.0015) + 1, replace=False):
                c = self.customers[ci]
                payee = self.counterparties[rng.integers(len(self.counterparties))]
                amt = round(min(float(np.exp(c.mu) * rng.uniform(5, 20)), 25000.0), 2)
                txs.append(Tx(base + timedelta(hours=float(rng.uniform(8, 20))), c, payee, amt, self._remit(rng, "big")))
            # legit near-threshold single payments (tax, tuition)
            for ci in rng.choice(len(self.customers), 2, replace=False):
                c = self.customers[ci]
                txs.append(Tx(base + timedelta(hours=float(rng.uniform(8, 18))), c, c.payees[0],
                              round(float(rng.uniform(9000, 9990)), 2), self._remit(rng)))
            if d >= 3:
                txs += self._fraud_for_day(base)
        txs.sort(key=lambda t: t.ts)
        for t in txs:
            t.uetr = self._uetr()
        return txs

    def _fraud_for_day(self, base: datetime) -> list[Tx]:
        rng, out = self.rng, []
        # APP scam / account takeover: victim sends 1-4 large payments to a mule within ~2h
        for _ in range(rng.poisson(5.0)):
            c = self.customers[rng.integers(len(self.customers))]
            mule = self._next_mule()
            start_h = float(rng.uniform(0, 5)) if rng.random() < 0.45 else float(rng.uniform(7, 22))
            for k in range(int(rng.integers(1, 5))):
                amt = round(min(float(np.exp(c.mu) * rng.uniform(6, 25)), 14900.0), 2)
                ts = base + timedelta(hours=start_h, minutes=float(k * rng.uniform(5, 40)))
                out.append(Tx(ts, c, mule, amt, self._remit(rng, "big"), typology="app_scam"))
        # mule fan-in: fake-shop victims pay one fresh mule within 6-30h, ordinary amounts and hours
        for _ in range(rng.poisson(0.9)):
            mule = self._next_mule()
            span = float(rng.uniform(6, 30))
            first = float(rng.uniform(6, 14))
            for ci in rng.choice(len(self.customers), int(rng.integers(10, 30)), replace=False):
                c = self.customers[ci]
                ts = base + timedelta(hours=first + float(rng.uniform(0, span)))
                out.append(Tx(ts, c, mule, round(float(rng.uniform(60, 700)), 2), self._remit(rng, "big"), typology="mule_fanin"))
        # structuring: 3-6 payments just under EUR 10k within ~20h
        for _ in range(rng.poisson(1.4)):
            c = self.customers[rng.integers(len(self.customers))]
            payees = [self.counterparties[rng.integers(len(self.counterparties))] for _ in range(int(rng.integers(1, 4)))]
            h0 = float(rng.uniform(7, 14))
            for k in range(int(rng.integers(3, 7))):
                ts = base + timedelta(hours=h0 + float(rng.uniform(0, 20)))
                out.append(Tx(ts, c, payees[k % len(payees)], round(float(rng.uniform(9000, 9990)), 2),
                              self._remit(rng), typology="structuring"))
        return out


# ---------------------------------------------------------------------- XML
def _corrupt_iban(rng, iban: str) -> str:
    i = int(rng.integers(4, len(iban)))
    ch = iban[i]
    repl = str((int(ch) + 1) % 10) if ch.isdigit() else ("A" if ch != "A" else "B")
    return iban[:i] + repl + iban[i + 1:]


def _tx_xml(t: Tx, settle: date, rng) -> tuple[str, dict]:
    cred_iban, cred_bic, ccy, amount, uetr = t.creditor.iban, t.creditor.bic, "EUR", t.amount, t.uetr
    for d in t.defects:
        if d == "bad_iban":
            cred_iban = _corrupt_iban(rng, cred_iban)
        elif d == "no_bic":
            cred_bic = None
        elif d == "usd":
            ccy = "USD"
        elif d == "nonpositive":
            amount = 0.0 if rng.random() < 0.5 else -amount
        elif d == "future":
            settle = settle + timedelta(days=9)
        elif d == "no_uetr":
            uetr = None
    e2e = f"E2E{t.uetr.replace('-', '')[:20]}"
    parts = [
        "<CdtTrfTxInf>",
        f"<PmtId><InstrId>{e2e[:16]}</InstrId><EndToEndId>{e2e}</EndToEndId>",
        f"<UETR>{uetr}</UETR>" if uetr else "",
        "</PmtId>",
        "<PmtTpInf><SvcLvl><Cd>SEPA</Cd></SvcLvl></PmtTpInf>",
        f'<IntrBkSttlmAmt Ccy="{ccy}">{amount:.2f}</IntrBkSttlmAmt>',
        f"<IntrBkSttlmDt>{settle.isoformat()}</IntrBkSttlmDt>",
        f"<AccptncDtTm>{t.ts.strftime('%Y-%m-%dT%H:%M:%SZ')}</AccptncDtTm>",
        "<ChrgBr>SLEV</ChrgBr>",
        f"<Dbtr><Nm>{escape(t.debtor.acct.name)}</Nm></Dbtr>",
        f"<DbtrAcct><Id><IBAN>{t.debtor.acct.iban}</IBAN></Id></DbtrAcct>",
        f"<DbtrAgt><FinInstnId><BICFI>{t.debtor.acct.bic}</BICFI></FinInstnId></DbtrAgt>",
        f"<CdtrAgt><FinInstnId><BICFI>{cred_bic}</BICFI></FinInstnId></CdtrAgt>" if cred_bic else "",
        f"<Cdtr><Nm>{escape(t.creditor.name)}</Nm></Cdtr>",
        f"<CdtrAcct><Id><IBAN>{cred_iban}</IBAN></Id></CdtrAcct>",
        f"<RmtInf><Ustrd>{escape(t.remittance)}</Ustrd></RmtInf>",
        "</CdtTrfTxInf>",
    ]
    status = ("RJCT", "AC01") if "bad_iban" in t.defects else ("RJCT", "AM05") if "usd" in t.defects else None
    return "".join(parts), {"uetr": uetr, "e2e": e2e, "status": status}


def _pacs008(msg_id: str, created: datetime, tx_xml: list[str], settle: date) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08"><FIToFICstmrCdtTrf>'
        f"<GrpHdr><MsgId>{msg_id}</MsgId><CreDtTm>{created.strftime('%Y-%m-%dT%H:%M:%SZ')}</CreDtTm>"
        f"<NbOfTxs>{len(tx_xml)}</NbOfTxs><SttlmInf><SttlmMtd>CLRG</SttlmMtd></SttlmInf></GrpHdr>"
        + "".join(tx_xml)
        + "</FIToFICstmrCdtTrf></Document>\n"
    )


def _pacs002(msg_id: str, orig_msg_id: str, created: datetime, statuses: list[tuple]) -> str:
    rows = []
    for uetr, e2e, sts, rsn in statuses:
        reason = f"<StsRsnInf><Rsn><Cd>{rsn}</Cd></Rsn></StsRsnInf>" if rsn else ""
        rows.append(
            f"<TxInfAndSts><OrgnlEndToEndId>{e2e}</OrgnlEndToEndId><OrgnlUETR>{uetr}</OrgnlUETR>"
            f"<TxSts>{sts}</TxSts>{reason}</TxInfAndSts>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.002.001.10"><FIToFIPmtStsRpt>'
        f"<GrpHdr><MsgId>{msg_id}</MsgId><CreDtTm>{created.strftime('%Y-%m-%dT%H:%M:%SZ')}</CreDtTm></GrpHdr>"
        f"<OrgnlGrpInfAndSts><OrgnlMsgId>{orig_msg_id}</OrgnlMsgId><OrgnlMsgNmId>pacs.008.001.08</OrgnlMsgNmId></OrgnlGrpInfAndSts>"
        + "".join(rows)
        + "</FIToFIPmtStsRpt></Document>\n"
    )


def generate(
    out: Path,
    start: date = date(2026, 9, 1),
    days: int = 30,
    seed: int = 7,
    n_customers: int = 4000,
    batch_size: int = 500,
    defect_rate_scale: float = 1.0,
) -> dict:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    gen = Generator(seed=seed, n_customers=n_customers, n_counterparties=int(n_customers * 1.5))
    rng = gen.rng
    txs = gen.simulate(start, days)

    defect_p = {k: v * defect_rate_scale for k, v in
                {"bad_iban": 0.004, "no_bic": 0.003, "usd": 0.002, "nonpositive": 0.0005, "future": 0.0005, "no_uetr": 0.0003}.items()}
    for t in txs:
        if t.typology:
            continue  # keep fraud ground truth clean of unrelated defects
        t.defects = tuple(k for k, p in defect_p.items() if rng.random() < p)

    by_day: dict[date, list[Tx]] = {}
    for t in txs:
        by_day.setdefault(t.ts.date(), []).append(t)

    manifest = {"transactions": len(txs), "files_pacs008": 0, "files_pacs002": 0, "resent_files": 0,
                "truncated_files": 0, "fraud_by_typology": {}, "defects": {}}
    cases = []
    last_day = start + timedelta(days=days - 1)
    seq = 0
    p008_files: list[Path] = []
    for day in sorted(by_day):
        day_txs = by_day[day]
        d8 = out / "pacs008" / day.isoformat()
        d2 = out / "pacs002" / (day + timedelta(days=1)).isoformat()
        d8.mkdir(parents=True, exist_ok=True)
        for b in range(0, len(day_txs), batch_size):
            batch = day_txs[b:b + batch_size]
            seq += 1
            msg_id = f"PLHB{day.strftime('%Y%m%d')}{seq:06d}"
            xmls, statuses = [], []
            for t in batch:
                x, meta = _tx_xml(t, day, rng)
                xmls.append(x)
                for dname in t.defects:
                    manifest["defects"][dname] = manifest["defects"].get(dname, 0) + 1
                if t.typology:
                    manifest["fraud_by_typology"][t.typology] = manifest["fraud_by_typology"].get(t.typology, 0) + 1
                    confirmed = t.ts.date() + timedelta(days=int(rng.integers(2, 21)))
                    cases.append((t.uetr, t.typology, confirmed.isoformat()))
                if meta["uetr"] is None:
                    continue
                if meta["status"]:
                    sts = meta["status"]
                elif rng.random() < 0.003:
                    sts = ("RJCT", "AM04")
                else:
                    sts = ("ACSC", None)
                statuses.append((meta["uetr"], meta["e2e"], *sts))
            created = max(t.ts for t in batch) + timedelta(minutes=5)
            f8 = d8 / f"pacs008_{msg_id}.xml"
            f8.write_text(_pacs008(msg_id, created, xmls, day), encoding="utf-8")
            p008_files.append(f8)
            manifest["files_pacs008"] += 1
            # the clearing's status reports arrive the next day; for the last day only half have arrived
            if day < last_day or rng.random() < 0.5:
                d2.mkdir(parents=True, exist_ok=True)
                (d2 / f"pacs002_STS{msg_id[4:]}.xml").write_text(
                    _pacs002(f"STS{msg_id[4:]}", msg_id, created + timedelta(hours=14), statuses), encoding="utf-8")
                manifest["files_pacs002"] += 1

    # operational noise: three files re-sent by the payment hub under new names, one truncated upload
    for f in [p008_files[i] for i in rng.choice(len(p008_files), 3, replace=False)]:
        shutil.copy(f, f.with_name(f.stem + "_resend.xml"))
        manifest["resent_files"] += 1
    victim = p008_files[len(p008_files) // 2]
    text = victim.read_text(encoding="utf-8")
    (victim.parent / (victim.stem + "_partial.xml")).write_text(text[: len(text) // 3], encoding="utf-8")
    manifest["truncated_files"] = 1

    (out / "cases").mkdir(parents=True, exist_ok=True)
    with open(out / "cases" / "case_outcomes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uetr", "typology", "confirmed_on"])
        w.writerows(cases)
    manifest["fraud_transactions"] = len(cases)
    manifest["start"], manifest["days"] = start.isoformat(), days
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
