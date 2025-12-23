"""
@file parsing.py
@brief Parsing euristico del testo OCR in metadati strutturati.
@ingroup domain_module

@details
Contiene regole/regex euristiche per estrarre:
- merchant (nome, P.IVA, indirizzo)
- receipt info (data/ora, numero documento)
- items (righe prodotto)
- totals (totale documento)

Questo è il modulo che tipicamente evolve di più.
"""

from __future__ import annotations
import re
from typing import Optional
from .models import Merchant, ReceiptInfo, Item, Totals


VAT_RE = re.compile(r"\b(?:P\.?\s*IVA|PIVA|VAT)\s*[:\-]?\s*([A-Z0-9]{8,15})\b", re.I)
DATE_RE = re.compile(r"\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{2,4})\b")
TIME_RE = re.compile(r"\b(\d{2}:\d{2})\b")
TOTAL_RE = re.compile(r"\b(?:TOTALE|TOT|TOTAL)\b.*?(\d+[\,\.]\d{2})\b", re.I)

VAT_CODE_RATE_RE = re.compile(
    r"\b([A-Z])\s*[:\-]?\s*IVA\s*([0-9]{1,2}(?:[\,\.][0-9]{1,2})?)\s*%\b",
    re.I
)

SECTION_ITEMS_START_RE = re.compile(r"\bDESCRIZIONE\b", re.I)
SECTION_ITEMS_END_RE = re.compile(r"\bARTICOLI\b", re.I)

def _lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]

def _slice_between(lines: list[str], start_re: re.Pattern, end_re: re.Pattern) -> list[str]:
    start_idx = next((i for i, l in enumerate(lines) if start_re.search(l)), None)
    if start_idx is None:
        return []
    end_idx = next((i for i in range(start_idx + 1, len(lines)) if end_re.search(lines[i])), None)
    if end_idx is None:
        return lines[start_idx + 1 :]
    return lines[start_idx + 1 : end_idx]


def _to_float_eur(s: str) -> Optional[float]:
    """
    @brief Converte stringa euro in float normalizzato.
    @param s Stringa numerica potenzialmente con '.' migliaia e ',' decimali.
    @return Float o None se conversione fallisce.

    @note Esempio: '1.234,56' -> 1234.56
    """
    try:
        s = s.replace(".", "").replace(",", ".")
        return float(s)
    except Exception:
        return None


def _parse_vat_code_map(text: str) -> dict[str, float]:
    """
    Estrae mapping codice IVA -> aliquota percentuale dal footer.
    Esempio: 'A: IVA 4,00%' => {'A': 4.0}
    """
    code_map: dict[str, float] = {}

    for m in VAT_CODE_RATE_RE.finditer(text):
        code = m.group(1).upper()
        rate_s = m.group(2).replace(",", ".")
        try:
            code_map[code] = float(rate_s)
        except Exception:
            # ignora valori non convertibili
            pass

    return code_map


def parse_merchant(text: str) -> Merchant:
    """
    @brief Estrae informazioni merchant (nome, P.IVA, indirizzo).
    @param text Testo OCR normalizzato.
    @return Merchant con campi valorizzati quando possibile.

    @note Heuristics:
    - Prima riga non vuota come nome.
    - Ricerca P.IVA tramite regex VAT_RE.
    - Indirizzo stimato da riga contenente CAP (5 cifre).
    """
    lines = _lines(text)
    merchant = Merchant()

    def score_name(l: str) -> int:
        s = 0
        if re.search(r"\b(SPA|S\.P\.A\.|SRL|S\.R\.L\.|SUPERMERCATI|MARKET|IPER)\b", l, re.I):
            s += 6
        if len(re.findall(r"[A-Z]", l)) >= 6:
            s += 2
        if re.search(r"[|_]{2,}", l):
            s -= 6
        if len(l) < 4:
            s -= 10
        return s

    # scegli candidato nelle prime righe, altrimenti fallback alla prima riga come prima
    if lines:
        top = lines[:10]
        best = max(top, key=score_name, default=None)
        if best and score_name(best) > 0:
            merchant.name = best[:120]
        else:
            merchant.name = lines[0][:120]

    m = VAT_RE.search(text)
    if m:
        merchant.vat_id = m.group(1)

    cap_line = next((l for l in lines if re.search(r"\b\d{5}\b", l)), None)
    if cap_line:
        merchant.address = cap_line[:200]

    return merchant


def parse_receipt_info(text: str) -> ReceiptInfo:
    """
    @brief Estrae metadati del documento (datetime, numero).
    @param text Testo OCR normalizzato.
    @return ReceiptInfo popolato euristicamente.

    @note La standardizzazione ISO della data può essere gestita in uno step successivo.
    """
    info = ReceiptInfo()

    date = DATE_RE.search(text)
    time = TIME_RE.search(text)
    if date and time:
        info.datetime = f"{date.group(1)} {time.group(1)}"
    elif date:
        info.datetime = date.group(1)

    # Priorità a DOC.NUM (con punti/spazi variabili)
    doc = re.search(r"\bDOC\.?\s*NUM\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/]{2,})\b", text, re.I)
    if not doc:
        doc = re.search(r"\b(?:DOC|DOCUMENTO|N\.|NR)\s*[:\-]?\s*([A-Z0-9\-\/]+)\b", text, re.I)

    if doc:
        info.document_number = doc.group(1)

    return info


def parse_totals(text: str) -> Totals:
    """
    @brief Estrae i totali (attualmente solo total).
    @param text Testo OCR normalizzato.
    @return Totals con campo total valorizzato se trovato.
    """
    totals = Totals()
    m = TOTAL_RE.search(text)
    if m:
        totals.total = _to_float_eur(m.group(1))
    return totals


def parse_items(text: str) -> list[Item]:
    """
    Estrae righe prodotto/servizio dalla sezione DESCRIZIONE -> ARTICOLI quando presente.
    Supporta formato tipico: 'DESC 0,75 B' (prezzo + codice IVA).
    Gestisce righe SCONTO come importi negativi.
    """
    items: list[Item] = []
    lines = _lines(text)

    # Preferisci la sezione articoli; fallback all'intero testo per compatibilità.
    item_lines = _slice_between(lines, SECTION_ITEMS_START_RE, SECTION_ITEMS_END_RE)
    if not item_lines:
        item_lines = lines

    vat_code_map = _parse_vat_code_map(text)

    price_end = re.compile(r"(.+?)\s+(-?\d+[\,\.]\d{2})$")              # legacy
    price_iva_end = re.compile(r"(.+?)\s+(-?\d+[\,\.]\d{2})\s+([ABC])$") # nuovo
    discount_re = re.compile(r"\bSCONTO\b", re.I)

    for l in item_lines:
        m = price_iva_end.match(l)
        iva_code = None
        if m:
            name_part = m.group(1).strip()
            price = _to_float_eur(m.group(2))
            iva_code = m.group(3).upper()
        else:
            m2 = price_end.match(l)
            if not m2:
                continue
            name_part = m2.group(1).strip()
            price = _to_float_eur(m2.group(2))

        if not name_part or price is None:
            continue

        # Heuristica sconti: se la riga contiene 'SCONTO' e il prezzo è positivo,
        # in molti casi OCR ha perso il segno '-'.
        if discount_re.search(name_part) and price > 0:
            price = -price

        vat_rate = vat_code_map.get(iva_code) if iva_code else None

        items.append(
            Item(
                raw_line=l,
                name=name_part[:120],
                total_price=price,
                # vat_rate: lasciato a None; mapping A/B/C a % si può fare in un commit successivo
                vat_rate=vat_rate,
            )
        )

    return items
