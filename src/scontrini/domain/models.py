"""
@file models.py
@brief Modelli dominio (contratto JSON) tramite Pydantic.
@ingroup domain_module

@details
Definisce il contratto dati versionato ReceiptContractV1.
Questi modelli fungono da:
- DTO tra layer (OCR/parsing/storage/api)
- schema implicito per serializzazione JSON
- base per validazione input/output
"""

from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """@brief Metadati della sorgente (immagine e timestamp cattura)."""
    image_path: str
    captured_at: str


class Merchant(BaseModel):
    """@brief Dati dell'esercente (azienda)."""
    name: Optional[str] = None
    address: Optional[str] = None
    vat_id: Optional[str] = None
    fiscal_code: Optional[str] = None
    country: str = "IT"


class ReceiptInfo(BaseModel):
    """@brief Metadati documento (valuta, data/ora, numero documento, metodo pagamento)."""
    currency: str = "EUR"
    datetime: Optional[str] = None
    document_number: Optional[str] = None
    payment_method: Optional[str] = None


class Item(BaseModel):
    """@brief Riga prodotto/servizio estratta dallo scontrino."""
    raw_line: Optional[str] = None
    name: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    vat_rate: Optional[float] = None


class Totals(BaseModel):
    """@brief Totali del documento."""
    subtotal: Optional[float] = None
    vat_total: Optional[float] = None
    total: Optional[float] = None


class OcrInfo(BaseModel):
    """@brief Metadati OCR (engine, lingua, testo, confidenza)."""
    engine: str = "tesseract"
    lang: str = "ita"
    text: Optional[str] = None
    confidence: Optional[float] = None


class Quality(BaseModel):
    """
    @brief Informazioni di qualità/audit.
    @details
    preprocess_steps: step eseguiti lato immagine.
    warnings: anomalie rilevate durante pipeline o parsing.
    """
    preprocess_steps: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ReceiptContractV1(BaseModel):
    """
    @brief Contratto principale versionato (v1).
    @details
    È l’oggetto canonico generato dalla pipeline e consumato da storage/API/UI.
    """
    contract_version: str = "1.0"
    source: Source
    merchant: Merchant = Field(default_factory=Merchant)
    receipt: ReceiptInfo = Field(default_factory=ReceiptInfo)
    items: List[Item] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    ocr: OcrInfo = Field(default_factory=OcrInfo)
    quality: Quality = Field(default_factory=Quality)
