"""
@file engine.py
@brief Wrapper per il motore OCR.
@ingroup ocr_module

@details
Attualmente usa Tesseract tramite pytesseract. Il wrapper separa:
- configurazione OCR
- esecuzione OCR
- standardizzazione output (testo + confidenza)
"""

from __future__ import annotations
from dataclasses import dataclass
import pytesseract
import numpy as np

@dataclass
class OcrResult:
    """
    @brief Risultato standardizzato del processo OCR.
    @details
    - text: testo estratto
    - confidence: confidenza (se disponibile), altrimenti None
    """
    text: str
    confidence: float | None


def run_tesseract(
    image_bin: np.ndarray,
    lang: str = "ita",
    psm: int = 6,
    extra_config: str | None = None,
) -> OcrResult:
    """
    @brief Esegue OCR Tesseract su immagine binarizzata.
    @param image_bin Immagine binaria (tipicamente output di preprocess_for_ocr).
    @param lang Codice lingua Tesseract (es. "ita", "eng").
    @param psm Page Segmentation Mode da passare a Tesseract (default 6, singola colonna).
    @param extra_config Configurazione aggiuntiva Tesseract (stringa, opzionale).
    @return OcrResult con testo e confidenza (se disponibile).

    @note
    - Configurazione OCR: OEM 1 (LSTM), PSM 6 (blocco uniforme).
    - La confidenza non è sempre affidabile senza image_to_data; qui è None.
    @see preprocess_for_ocr
    """
    base_config = (
        f"--oem 1 --psm {psm} "
        "-c textord_heavy_nr=1 "
        "-c edges_max_children=1 "
        "-c preserve_interword_spaces=0"
    )
    config = f"{base_config} {extra_config}" if extra_config else base_config

    text = pytesseract.image_to_string(image_bin, lang=lang, config=config)
    return OcrResult(text=text, confidence=None)