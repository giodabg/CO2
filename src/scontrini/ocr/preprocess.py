"""
@file preprocess.py
@brief Pre-processing di immagini di scontrini per aumentare la qualità OCR.
@ingroup ocr_module

@details
Il pre-processing mira a rendere il testo più leggibile da OCR:
- conversione in scala di grigi
- riduzione rumore
- binarizzazione adattiva
- morfologia per connettere stroke dei caratteri

L'output include anche la lista degli step applicati per auditing/qualità.
"""

from __future__ import annotations
import cv2
import numpy as np


def preprocess_for_ocr(image_bgr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    @brief Applica una pipeline di pre-processing ad un'immagine BGR.
    @param image_bgr Immagine in formato BGR (tipico output di cv2.imread).
    @return Tuple (image_bin, steps) dove:
            - image_bin è un'immagine binarizzata pronta per OCR
            - steps è l'elenco degli step applicati in ordine

    @throws ValueError Se l'immagine è vuota o non valida (estendibile).
    @note La pipeline è deliberatamente semplice; è prevista evoluzione.
    """
    steps: list[str] = []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    steps.append("to_gray")

    # denoise
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    steps.append("bilateral_filter")

    # adaptive threshold to handle lighting
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    steps.append("adaptive_threshold")

    # small morphology to connect characters
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    steps.append("morph_close")

    return th, steps
