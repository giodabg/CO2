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


# Sezioni tipiche scontrini
SECTION_ITEMS_START_RE = re.compile(r"\bDESCRIZIONE\b", re.I)
SECTION_ITEMS_END_RE = re.compile(r"\bARTICOLI\b", re.I)

# Totali (fallback quando "TOTALE COMPLESSIVO" non porta il numero sulla stessa riga)
MONETA_RE = re.compile(r"\bMoneta\s+altro\b.*?(\d+[\,\.]\d{2})\b", re.I)
PAGATO_RE = re.compile(r"\bIMPORTO\s+PAGATO\b.*?(\d+[\,\.]\d{2})\b", re.I)
DI_CUI_IVA_RE = re.compile(r"\bDI\s+CUI\s+IVA\b.*?(\d+[\,\.]\d{2})\b", re.I)

# Legenda IVA: standard (A: IVA 4,00%) + variante OCR frequente "AAIVA 4,00%"
VAT_CODE_RATE_STD = re.compile(
    r"\b([ABC])\s*[:\-]?\s*IVA\s*([0-9]{1,2}(?:[\,\.][0-9]{1,2})?)\s*%\b",
    re.I,
)

VAT_CODE_RATE_A_OCR = re.compile(
    r"\bA{1,2}\s*[:\-]?\s*IVA\s*([0-9]{1,2}(?:[\,\.][0-9]{1,2})?)\s*%\b",
    re.I,
)

VAT_CODE_RATE_AAIVA = re.compile(
    r"\bAAIVA\s*([0-9]{1,2}(?:[\,\.][0-9]{1,2})?)\s*%\b",
    re.I,
)

import re

_ALLOWED_NAME_RE = re.compile(r"[^A-Za-z0-9À-ÿ\s\.\,]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")

def clean_item_name(name: str) -> str:
    """
    Ripulisce un nome prodotto OCR:
    - rimuove caratteri non plausibili (simboli OCR, pipe, apici strani, ecc.)
    - mantiene lettere (incl. accentate), cifre, spazi, punto e virgola
    - collassa spazi multipli
    - rimuove punteggiatura ripetuta ai margini
    """
    if not name:
        return name

    # 1) elimina caratteri non ammessi
    s = _ALLOWED_NAME_RE.sub(" ", name)

    # 2) collassa spazi
    s = _MULTI_SPACE_RE.sub(" ", s).strip()

    # 3) pulizia margini: toglie puntini/virgole isolate all'inizio/fine
    s = re.sub(r"^[\.\,]+\s*", "", s)
    s = re.sub(r"\s*[\.\,]+$", "", s)

    return s[:120]

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



def _lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


def _slice_between(lines: list[str], start_re: re.Pattern, end_re: re.Pattern) -> list[str]:
    """
    Ritorna le righe tra una riga che matcha start_re e una riga che matcha end_re.
    Se end_re non viene trovato, prende fino a fine testo.
    """
    start_idx = next((i for i, l in enumerate(lines) if start_re.search(l)), None)
    if start_idx is None:
        return []
    end_idx = next((i for i in range(start_idx + 1, len(lines)) if end_re.search(lines[i])), None)
    if end_idx is None:
        return lines[start_idx + 1 :]
    return lines[start_idx + 1 : end_idx]


def _clean_merchant_line(s: str) -> str:
    # Mantiene caratteri utili e normalizza gli spazi
    s = re.sub(r"[^A-Za-z0-9À-ÿ\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_item_line(s: str) -> str:
    # Rimuove separatori tipici OCR (pipe) e compatta spazi
    s = s.replace("|", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_vat_code_map(text: str) -> dict[str, float]:
    """
    Estrae mapping codice IVA -> aliquota percentuale.
    Supporta:
    - 'A: IVA 4,00%' (standard)
    - 'A IVA 4,00%' / 'AA IVA 4,00%' (OCR)
    - 'AAIVA 4,00%' (OCR comune)
    """
    code_map: dict[str, float] = {}

    for m in VAT_CODE_RATE_STD.finditer(text):
        code = m.group(1).upper()
        rate_s = m.group(2).replace(",", ".")
        try:
            code_map[code] = float(rate_s)
        except Exception:
            pass

    if "A" not in code_map:
        m2 = VAT_CODE_RATE_A_OCR.search(text)
        if m2:
            try:
                code_map["A"] = float(m2.group(1).replace(",", "."))
            except Exception:
                pass

    if "A" not in code_map:
        m3 = VAT_CODE_RATE_AAIVA.search(text)
        if m3:
            try:
                code_map["A"] = float(m3.group(1).replace(",", "."))
            except Exception:
                pass

    return code_map

def _normalize_desc(desc: str) -> str:
    desc = desc.upper()
    desc = re.sub(r"[^A-Z0-9À-ÿ\s\.\-]", " ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    # rimuove prefissi OCR comuni che non sono parte del nome
    desc = re.sub(r"^(?:O|A|I|E)\s+", "", desc)
    desc = re.sub(r"^(?:\-+|\.+)\s*", "", desc)
    return desc

def strip_leading_singleton(name: str) -> str:
    # rimuove un singolo token di 1 char se seguito da testo significativo
    return re.sub(r"^(?:[A-Za-z])\s+(?=\w{3,})", "", name).strip()


SECTION_ITEMS_START_RE = re.compile(r"\bDESCRIZIONE\b", re.I)
SECTION_ITEMS_END_RE = re.compile(r"\bARTICOLI\b", re.I)

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
        merchant.address = _clean_merchant_line(cap_line)[:200]

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
    # total: prima prova "TOTALE...", poi fallback su "Moneta altro"
    m = TOTAL_RE.search(text)
    if m:
        totals.total = _to_float_eur(m.group(1))
    else:
        m2 = MONETA_RE.search(text)
        if m2:
            totals.total = _to_float_eur(m2.group(1))

    # vat_total: 'DI CUI IVA 4,42'
    m3 = DI_CUI_IVA_RE.search(text)
    if m3:
        totals.vat_total = _to_float_eur(m3.group(1))
    return totals


def parse_items(text: str) -> list[Item]:
    """
    Estrae righe prodotto/servizio dalla sezione DESCRIZIONE -> ARTICOLI quando presente.

    Supporta due layout tipici:
    - Layout A (es. IPERAL):  'DESC 0,75 B'  -> prezzo + codice IVA (A/B/C)
    - Layout B (es. MD):     'DESC 4,00 0,99' -> IVA% + prezzo (due numeri)

    Gestisce righe SCONTO come importi negativi e scarta falsi sconti tipo "su SCONTO 21,24".
    Ripulisce il nome prodotto eliminando caratteri non plausibili (mantiene lettere/cifre/spazi/punti/virgole).
    """
    items: list[Item] = []
    lines = _lines(text)

    # Preferisci la sezione articoli; fallback all'intero testo per compatibilità.
    item_lines = _slice_between(lines, SECTION_ITEMS_START_RE, SECTION_ITEMS_END_RE)
    if not item_lines:
        item_lines = lines

    # Mappa A/B/C -> aliquota (se presente nel footer)
    vat_code_map = _parse_vat_code_map(text)

    # Numeri tipo 0,99 / 25.55 / 10,00
    num_re = re.compile(r"\d+[\,\.]\d{2}")
    iva_code_re = re.compile(r"\b([ABC])\b", re.I)

    # Righe sicuramente NON-articolo
    non_item_re = re.compile(
        r"\b(TOTALE|TOT|SUBTOT|SUBTOT\.?|IMPORTO|PAGATO|PAGAMENTO|RESTO|MONETA|ARTICOLI|DOC\.?|DOCUMENTO)\b",
        re.I,
    )

    seen: set[tuple[str, float]] = set()

    for raw in item_lines:
        cl = _clean_item_line(raw)
        if not cl:
            continue

        # Scarta righe di totali/pagamenti ecc.
        if non_item_re.search(cl):
            continue

        # Estrai tutti i numeri decimali presenti nella riga
        nums = num_re.findall(cl)
        if not nums:
            continue

        # Heuristica: descrizione = parte prima del primo numero
        first_num_pos = None
        m_first = num_re.search(cl)
        if m_first:
            first_num_pos = m_first.start()

        desc = cl if first_num_pos is None else cl[:first_num_pos].strip()
        if not desc or len(desc) < 3:
            continue

        # Clean nome prodotto (whitelist)
        desc_clean = clean_item_name(desc)

        # Se dopo pulizia è troppo corto, scarta
        if not desc_clean or len(desc_clean) < 3:
            continue

        # Prezzo e IVA rate
        price: Optional[float] = None
        vat_rate: Optional[float] = None

        # Layout B: due numeri -> primo = IVA%, ultimo = prezzo
        if len(nums) >= 2:
            vat_rate = _to_float_eur(nums[0])
            price = _to_float_eur(nums[-1])
        else:
            # Layout A: un numero -> prezzo (codice IVA eventuale)
            price = _to_float_eur(nums[0])

        if price is None:
            continue

        # Codice IVA A/B/C (layout A). Se presente e mappabile, valorizza vat_rate.
        m_code = iva_code_re.search(cl)
        if m_code:
            code = m_code.group(1).upper()
            mapped = vat_code_map.get(code)
            if mapped is not None:
                vat_rate = mapped

        # Normalizza per logica sconti e dedup
        norm_desc = _normalize_desc(desc_clean)

        # Sconti: scarta falsi sconti tipo "su SCONTO 21,24"
        if "SCONTO" in norm_desc:
            has_minus = "-" in cl
            has_percent = "%" in cl
            if (not has_minus) and (not has_percent) and abs(price) > 5:
                continue
            if (not has_minus) and price > 0:
                price = -price

        # Dedup soft: descrizione normalizzata + prezzo
        key = (norm_desc, round(float(price), 2))
        if key in seen:
            continue
        seen.add(key)

        items.append(
            Item(
                raw_line=raw,
                name=desc_clean[:120],
                total_price=price,
                vat_rate=vat_rate,
            )
        )

    return items
