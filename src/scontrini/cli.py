"""
@file cli.py
@brief CLI per esecuzione pipeline (OCR+parsing) e persistenza.
@ingroup cli_module

@details
Comandi:
- dump-json: stampa contratto JSON senza DB
- run: salva su SQLite e stampa receipt_id
"""

from __future__ import annotations
import argparse
import json
import cv2
import re

from scontrini.domain.models import OcrInfo, Quality, ReceiptContractV1, Source
from scontrini.ocr.preprocess import preprocess_for_ocr
from scontrini.ocr.engine import run_tesseract
from scontrini.ocr.postprocess import normalize_ocr_text
from scontrini.domain.parsing import parse_merchant, parse_receipt_info, parse_items, parse_totals
from scontrini.storage.db import connect
from scontrini.storage.repository import insert_receipt


def _sum_items(items):
    s = 0.0
    for it in items:
        if it.total_price is not None:
            s += float(it.total_price)
    return s

def _extract_declared_items_count(text: str):
    m = re.search(r"\bARTICOLI\s+(\d+)\b", text, re.I)
    return int(m.group(1)) if m else None


def _extract_paid_amount(text: str):
    m = re.search(r"\bIMPORTO\s+PAGATO\b.*?(\d+[\,\.]\d{2})\b", text, re.I)
    if not m:
        return None
    s = m.group(1).replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def build_contract(image_path: str, captured_at: str, lang: str) -> ReceiptContractV1:
    """
    @brief Costruisce il contratto ReceiptContractV1 a partire da una foto.
    @param image_path Path immagine (jpg/png).
    @param captured_at Timestamp cattura (stringa ISO).
    @param lang Lingua OCR (Tesseract).
    @return ReceiptContractV1 completo.

    @throws FileNotFoundError Se l'immagine non è leggibile da OpenCV.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere immagine: {image_path}")

    pre, steps = preprocess_for_ocr(img)
    ocr = run_tesseract(pre, lang=lang)
    text = normalize_ocr_text(ocr.text)

    merchant = parse_merchant(text)
    receipt_info = parse_receipt_info(text)
    items = parse_items(text)
    totals = parse_totals(text)

    warnings = []

    declared = _extract_declared_items_count(text)
    extracted = len([i for i in items if i.total_price is not None])

    if declared is not None and extracted != declared:
        warnings.append(f"items_count_mismatch: declared={declared} extracted={extracted}")

    # Coerenza totale: se totals.total c'è, confronta con somma items
    items_sum = _sum_items(items)

    if totals.total is None:
        warnings.append("total_missing")
    else:
        delta = abs(items_sum - float(totals.total))
        if delta > 0.05:
            warnings.append(
                f"totals_inconsistent: sum_items={items_sum:.2f} total={float(totals.total):.2f} delta={delta:.2f}"
            )

    paid = _extract_paid_amount(text)
    if paid is not None and totals.total is not None:
        # Se l'importo pagato è molto diverso dal totale, segnala (tipico errore OCR 32,52 -> 92,52)
        if abs(paid - float(totals.total)) > 0.50:
            warnings.append(f"paid_amount_suspect: paid={paid:.2f} total={float(totals.total):.2f}")

    return ReceiptContractV1(
        source=Source(image_path=image_path, captured_at=captured_at),
        merchant=merchant,
        receipt=receipt_info,
        items=items,
        totals=totals,
        ocr=OcrInfo(engine="tesseract", lang=lang, text=text, confidence=ocr.confidence),
        quality=Quality(preprocess_steps=steps, warnings=warnings),
    )


def main() -> None:
    """
    @brief Entry point CLI.
    @return None

    @details
    Parse args e invoca build_contract, poi:
    - dump-json: stampa JSON
    - run: persiste su DB e stampa receipt_id
    """
    p = argparse.ArgumentParser(prog="scontrini")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Esegue OCR+parsing e salva su SQLite")
    run.add_argument("--image", required=True)
    run.add_argument("--captured-at", required=True, help="ISO string, es. 2025-12-23T10:20:00+01:00")
    run.add_argument("--db", default="data/scontrini.sqlite")
    run.add_argument("--lang", default="ita")

    dump = sub.add_parser("dump-json", help="Esegue OCR+parsing e stampa il JSON (senza DB)")
    dump.add_argument("--image", required=True)
    dump.add_argument("--captured-at", required=True)
    dump.add_argument("--lang", default="ita")

    args = p.parse_args()

    if args.cmd in ("run", "dump-json"):
        contract = build_contract(args.image, args.captured_at, args.lang)

    if args.cmd == "dump-json":
        print(contract.model_dump_json(indent=2, ensure_ascii=False))
        return

    if args.cmd == "run":
        conn = connect(args.db)
        rid = insert_receipt(conn, contract)
        print(json.dumps({"receipt_id": rid, "db": args.db}, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
