"""
@file repository.py
@brief Layer repository per persistenza del contratto su SQLite.
@ingroup storage_module

@details
Isola SQL e schema dal resto dell'applicazione.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path

from scontrini.domain.models import ReceiptContractV1


def init_schema(conn: sqlite3.Connection) -> None:
    """
    @brief Applica lo schema SQL (idempotente).
    @param conn Connessione SQLite.
    @return None
    """
    schema_path = Path(__file__).with_name("schema.sql")
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    conn.commit()


def insert_receipt(conn: sqlite3.Connection, contract: ReceiptContractV1) -> int:
    """
    @brief Inserisce un ReceiptContractV1 nel database.
    @param conn Connessione SQLite.
    @param contract Contratto dati da persistere.
    @return ID (PK) della receipt inserita.

    @details
    - Inserisce record in receipts
    - Inserisce righe in receipt_items con FK verso receipts
    """
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO receipts(
          captured_at, image_path,
          merchant_name, merchant_address, vat_id,
          receipt_datetime, document_number, currency, total, ocr_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            contract.source.captured_at,
            contract.source.image_path,
            contract.merchant.name,
            contract.merchant.address,
            contract.merchant.vat_id,
            contract.receipt.datetime,
            contract.receipt.document_number,
            contract.receipt.currency,
            contract.totals.total,
            contract.ocr.text,
        ),
    )
    receipt_id_row = cur.lastrowid
    if receipt_id_row is None:
        raise RuntimeError("Failed to get lastrowid after inserting receipt")
    receipt_id = int(receipt_id_row)

    for it in contract.items:
        cur.execute(
            """
            INSERT INTO receipt_items(
              receipt_id, raw_line, name, quantity, unit, unit_price, total_price, vat_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                it.raw_line,
                it.name,
                it.quantity,
                it.unit,
                it.unit_price,
                it.total_price,
                it.vat_rate,
            ),
        )

    conn.commit()
    return receipt_id
