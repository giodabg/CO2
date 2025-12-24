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


def get_items_parsers_registry():
    """
    Registry centralizzato dei parser items disponibili.
    Aggiungi qui nuovi formati senza cambiare il dispatcher.
    """
    return {
        "iperal": parse_items_iperal,
        "md": parse_items_md,
        "generic": _parse_items_dual_layout,  # o parse_items_generic se preferisci
    }

def _extract_declared_items_count(text: str) -> Optional[int]:
    m = re.search(r"\bARTICOLI\s+(\d+)\b", text, re.I)
    return int(m.group(1)) if m else None

def _score_items_result(text: str, items: list[Item], totals: Optional[Totals] = None) -> float:
    """
    Score più alto = migliore.
    Criteri:
    - item count vicino a 'ARTICOLI N' (se presente) (escludendo sconti)
    - delta tra somma item e totale (se totale disponibile) -> penalità
    - penalità per righe chiaramente "non item" finite negli items
    - reward per percentuale di items con prezzo plausibile
    """
    if totals is None:
        totals = parse_totals(text)

    declared = _extract_declared_items_count(text)

    # Conteggio prodotti: escludi sconti dal conteggio articoli
    products = [it for it in items if it.name and "SCONTO" not in _normalize_desc(it.name)]
    n_products = len(products)

    # Somma prezzi (include sconti: serve per coerenza col totale)
    s = 0.0
    priced = 0
    for it in items:
        if it.total_price is not None:
            s += float(it.total_price)
            priced += 1

    # Percentuale righe con prezzo
    price_ratio = (priced / max(len(items), 1))

    score = 0.0

    # Reward: più prezzi validi
    score += 10.0 * price_ratio

    # Reward: preferisci avere "abbastanza" righe (ma non esplodere)
    score += min(len(items), 30) * 0.2

    # Penalità: mismatch con ARTICOLI N
    if declared is not None:
        score -= abs(n_products - declared) * 2.5

    # Penalità: incoerenza con totale
    if totals.total is not None:
        delta = abs(s - float(totals.total))
        # penalità crescente: 0.1€ -> piccola, >2€ -> grande
        score -= min(delta * 3.0, 50.0)

    # Penalità: item sospetti che contengono parole "non item"
    suspicious_re = re.compile(r"\b(PAGAMENTO|PAGATO|RESTO|TOTALE|SUBTOT|IMPORTO|DOC|DOCUMENTO)\b", re.I)
    suspicious = 0
    for it in items:
        if it.name and suspicious_re.search(it.name):
            suspicious += 1
    score -= suspicious * 5.0

    return score

def _ordered_formats_for_try(merchant: Merchant) -> list[str]:
    primary = detect_receipt_format(merchant)
    registry = get_items_parsers_registry()

    # Metti primary davanti, poi gli altri (senza duplicati)
    ordered = []
    if primary in registry:
        ordered.append(primary)
    for k in registry.keys():
        if k not in ordered:
            ordered.append(k)
    return ordered


def strip_leading_singleton(name: str) -> str:
    # rimuove un singolo token di 1 char se seguito da testo significativo
    return re.sub(r"^(?:[A-Za-z])\s+(?=\w{3,})", "", name).strip()


def detect_receipt_format(merchant: Merchant) -> str:
    """
    Determina il formato dello scontrino basandosi sul merchant estratto.
    Ritorna: 'iperal', 'md', 'generic'
    """
    name = (merchant.name or "").upper()
    addr = (merchant.address or "").upper()

    # IPERAL
    if "IPERAL" in name:
        return "iperal"

    # MD (come parola intera per ridurre falsi positivi)
    if re.search(r"\bMD\b", name) or "MD SPA" in name or re.search(r"\bMD\b", addr):
        return "md"

    return "generic"




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

def parse_items_iperal(text: str) -> list[Item]:
    """
    Parser specializzato per scontrini tipo IPERAL:
    righe item del tipo: 'DESC 0,75 B' (prezzo + codice IVA A/B/C)
    """
    items: list[Item] = []
    lines = _lines(text)

    item_lines = _slice_between(lines, SECTION_ITEMS_START_RE, SECTION_ITEMS_END_RE)
    if not item_lines:
        item_lines = lines

    vat_code_map = _parse_vat_code_map(text)

    # Prezzo + codice IVA A/B/C (anche con junk dopo)
    iperal_re = re.compile(
        r"^(?P<desc>.+?)\s+(?P<price>-?\d+[\,\.]\d{2})\s*(?P<code>[ABC])\b.*$",
        re.I,
    )

    non_item_re = re.compile(
        r"\b(TOTALE|TOT|SUBTOT|SUBTOT\.?|IMPORTO|PAGATO|PAGAMENTO|RESTO|MONETA|ARTICOLI|DOC\.?|DOCUMENTO|IVA)\b",
        re.I,
    )

    seen: set[tuple[str, float]] = set()

    for raw in item_lines:
        cl = _clean_item_line(raw)
        if not cl or non_item_re.search(cl):
            continue

        m = iperal_re.match(cl)
        if not m:
            continue

        desc_raw = m.group("desc").strip()
        desc_clean = clean_item_name(desc_raw)
        desc_clean = strip_leading_singleton(desc_clean)

        price = _to_float_eur(m.group("price"))
        code = m.group("code").upper()

        if not desc_clean or price is None:
            continue

        norm_desc = _normalize_desc(desc_clean)

        # Sconti robusti (scarta falsi "SCONTO 21,24")
        if "SCONTO" in norm_desc:
            has_minus = "-" in cl
            has_percent = "%" in cl
            if (not has_minus) and (not has_percent) and abs(price) > 5:
                continue
            if (not has_minus) and price > 0:
                price = -price

        vat_rate = vat_code_map.get(code)

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

def parse_items_md(text: str) -> list[Item]:
    """
    Parser specializzato per scontrini tipo MD:
    righe item del tipo: 'DESC 4,00 0,99' (IVA% + prezzo)
    """
    items: list[Item] = []
    lines = _lines(text)

    item_lines = _slice_between(lines, SECTION_ITEMS_START_RE, SECTION_ITEMS_END_RE)
    if not item_lines:
        item_lines = lines

    num_re = re.compile(r"\d+[\,\.]\d{2}")

    non_item_re = re.compile(
        r"\b(TOTALE|TOT|SUBTOT|SUBTOT\.?|IMPORTO|PAGATO|PAGAMENTO|RESTO|MONETA|ARTICOLI|DOC\.?|DOCUMENTO)\b",
        re.I,
    )

    seen: set[tuple[str, float]] = set()

    for raw in item_lines:
        cl = _clean_item_line(raw)
        if not cl or non_item_re.search(cl):
            continue

        nums = num_re.findall(cl)
        if len(nums) < 2:
            # Layout MD richiede IVA% e prezzo
            continue

        # Descrizione = parte prima del primo numero
        m_first = num_re.search(cl)
        desc_raw = cl[: m_first.start()].strip() if m_first else cl.strip()
        if not desc_raw or len(desc_raw) < 3:
            continue

        desc_clean = clean_item_name(desc_raw)
        desc_clean = strip_leading_singleton(desc_clean)

        vat_rate = _to_float_eur(nums[0])
        price = _to_float_eur(nums[-1])

        if not desc_clean or price is None:
            continue

        norm_desc = _normalize_desc(desc_clean)

        # Sconti robusti
        if "SCONTO" in norm_desc:
            has_minus = "-" in cl
            has_percent = "%" in cl
            if (not has_minus) and (not has_percent) and abs(price) > 5:
                continue
            if (not has_minus) and price > 0:
                price = -price

        key = (norm_desc, round(float(price), 2))
        if key in seen:
            continue
        seen.add(key)

        items.append(
            Item(
                raw_line=raw,
                name=desc_clean[:120],
                total_price=price,
                vat_rate=vat_rate,  # su MD è IVA% numerica
            )
        )

    return items


def _parse_items_dual_layout(text: str) -> list[Item]:
    """
    Parser generico (fallback): supporta sia layout A (prezzo + A/B/C) sia layout B (IVA% + prezzo).
    È la precedente implementazione di parse_items.
    """
    items: list[Item] = []
    lines = _lines(text)

    item_lines = _slice_between(lines, SECTION_ITEMS_START_RE, SECTION_ITEMS_END_RE)
    if not item_lines:
        item_lines = lines

    vat_code_map = _parse_vat_code_map(text)

    num_re = re.compile(r"\d+[\,\.]\d{2}")
    iva_code_re = re.compile(r"\b([ABC])\b", re.I)

    non_item_re = re.compile(
        r"\b(TOTALE|TOT|SUBTOT|SUBTOT\.?|IMPORTO|PAGATO|PAGAMENTO|RESTO|MONETA|ARTICOLI|DOC\.?|DOCUMENTO)\b",
        re.I,
    )

    seen: set[tuple[str, float]] = set()

    for raw in item_lines:
        cl = _clean_item_line(raw)
        if not cl:
            continue

        if non_item_re.search(cl):
            continue

        nums = num_re.findall(cl)
        if not nums:
            continue

        m_first = num_re.search(cl)
        first_num_pos = m_first.start() if m_first else None

        desc = cl if first_num_pos is None else cl[:first_num_pos].strip()
        if not desc or len(desc) < 3:
            continue

        desc_clean = clean_item_name(desc)
        desc_clean = strip_leading_singleton(desc_clean)

        if not desc_clean or len(desc_clean) < 3:
            continue

        price: Optional[float] = None
        vat_rate: Optional[float] = None

        if len(nums) >= 2:
            vat_rate = _to_float_eur(nums[0])
            price = _to_float_eur(nums[-1])
        else:
            price = _to_float_eur(nums[0])

        if price is None:
            continue

        m_code = iva_code_re.search(cl)
        if m_code:
            code = m_code.group(1).upper()
            mapped = vat_code_map.get(code)
            if mapped is not None:
                vat_rate = mapped

        norm_desc = _normalize_desc(desc_clean)

        if "SCONTO" in norm_desc:
            has_minus = "-" in cl
            has_percent = "%" in cl
            if (not has_minus) and (not has_percent) and abs(price) > 5:
                continue
            if (not has_minus) and price > 0:
                price = -price

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

def parse_items_with_meta(
    text: str,
    merchant: Optional[Merchant] = None,
) -> tuple[list[Item], str, float]:
    """
    Ritorna:
    - items estratti
    - formato selezionato ('iperal', 'md', 'generic', ...)
    - score di qualità
    """
    if merchant is None:
        merchant = parse_merchant(text)

    registry = get_items_parsers_registry()
    totals = parse_totals(text)

    best_items: list[Item] = []
    best_fmt: str = "unknown"
    best_score: float = float("-inf")

    for fmt in _ordered_formats_for_try(merchant):
        parser_fn = registry.get(fmt)
        if not parser_fn:
            continue
        try:
            candidate = parser_fn(text)
        except Exception:
            continue

        score = _score_items_result(text, candidate, totals=totals)

        if score > best_score:
            best_items = candidate
            best_fmt = fmt
            best_score = score

    return best_items, best_fmt, best_score


def parse_items(text: str, merchant: Optional[Merchant] = None) -> list[Item]:
    """
    Dispatcher robusto:
    - determina un formato primario via merchant
    - prova più parser (primario + fallback)
    - sceglie il risultato con score migliore (coerenza totale, ARTICOLI, ecc.)
    - permette di aggiungere formati registrandoli nel registry
    """
    if merchant is None:
        merchant = parse_merchant(text)

    registry = get_items_parsers_registry()
    totals = parse_totals(text)

    best_items: list[Item] = []
    best_fmt: Optional[str] = None
    best_score: float = float("-inf")

    for fmt in _ordered_formats_for_try(merchant):
        parser_fn = registry.get(fmt)
        if parser_fn is None:
            continue

        try:
            candidate = parser_fn(text)
        except Exception:
            # Se un parser fallisce, non bloccare tutto
            continue

        score = _score_items_result(text, candidate, totals=totals)

        if score > best_score:
            best_score = score
            best_items = candidate
            best_fmt = fmt

    # Opzionale: se nessun parser produce risultati, fallback vuoto
    return best_items


