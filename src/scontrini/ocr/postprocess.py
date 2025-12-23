"""
@file postprocess.py
@brief Normalizzazione e pulizia del testo OCR.
@ingroup ocr_module

@details
Il testo OCR può includere spaziature e ritorni a capo anomali.
La normalizzazione riduce la variabilità prima del parsing.
"""

from __future__ import annotations
import re


def normalize_ocr_text(text: str) -> str:
    """
    @brief Normalizza testo OCR per parsing consistente.
    @param text Testo grezzo OCR.
    @return Testo normalizzato (trim, spazi ridotti, newline consolidati).

    @note
    Questa funzione non “corregge” semanticamente il testo; riduce artefatti.
    """
    t = text.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
