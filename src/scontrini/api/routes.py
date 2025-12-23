"""
@file routes.py
@brief Endpoints HTTP per ingest scontrino e health.
@ingroup api_module

@details
Espone API minimali:
- GET /health
- POST /ingest
"""

from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from scontrini.cli import build_contract
from scontrini.storage.db import connect
from scontrini.storage.repository import insert_receipt

router = APIRouter()


class IngestRequest(BaseModel):
    """
    @brief Payload ingest.
    @details
    Permette ingest da path locale immagine e salva su SQLite.
    """
    image_path: str
    captured_at: str
    db_path: str = "data/scontrini.sqlite"
    lang: str = "ita"


@router.post("/ingest")
def ingest(req: IngestRequest):
    """
    @brief Esegue OCR+parsing e persiste il risultato.
    @param req IngestRequest con path immagine, timestamp, db path e lingua.
    @return JSON con receipt_id e contract serializzato.

    @note
    Questo endpoint è pensato per essere consumato da una UI WebView.
    """
    contract = build_contract(req.image_path, req.captured_at, req.lang)
    conn = connect(req.db_path)
    rid = insert_receipt(conn, contract)
    return {"receipt_id": rid, "contract": contract.model_dump()}


@router.get("/health")
def health():
    """
    @brief Healthcheck semplice.
    @return {"ok": True}
    """
    return {"ok": True}
