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


def _crop_by_largest_contour(gray: np.ndarray, margin: int = 8) -> tuple[np.ndarray, bool]:
    """
    @brief Prova a ritagliare l'area principale (lo scontrino) usando il contorno più grande.
    @param gray Immagine in scala di grigi.
    @param margin Margine da lasciare intorno al bounding box (in pixel).
    @return (immagine_ritagliata, cropped) dove cropped indica se il crop è avvenuto.

    @note È un ritaglio "safe": se il contorno non è affidabile, restituisce l'immagine originale.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, False

    h, w = gray.shape[:2]
    max_area = 0
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            best = c

    if best is None:
        return gray, False

    x, y, bw, bh = cv2.boundingRect(best)

    # Se il bounding box è troppo piccolo rispetto all'immagine, evita crop aggressivi.
    if bw < 0.2 * w or bh < 0.2 * h:
        return gray, False

    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + bw + margin, w)
    y1 = min(y + bh + margin, h)

    return gray[y0:y1, x0:x1], True


def preprocess_for_ocr(
    image_bgr: np.ndarray,
    *,
    enable_crop: bool = True,
    crop_margin: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """
    @brief Applica una pipeline di pre-processing ad un'immagine BGR.
    @param image_bgr Immagine in formato BGR (tipico output di cv2.imread).
    @param enable_crop Se True prova a ritagliare lo scontrino rilevando il contorno principale.
    @param crop_margin Margine (px) aggiuntivo attorno al contorno prima del crop.    @return Tuple (image_bin, steps) dove:
    @return Tuple (image_bin, steps) dove:
            - image_bin è un'immagine binarizzata pronta per OCR
            - steps è l'elenco degli step applicati in ordine

    @throws ValueError Se l'immagine è vuota o non valida (estendibile).
    @note La pipeline è deliberatamente semplice; è prevista evoluzione.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Immagine vuota o non valida")

    steps: list[str] = []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    steps.append("to_gray")

    if enable_crop:
        cropped_gray, cropped = _crop_by_largest_contour(gray, margin=crop_margin)
        if cropped:
            steps.append(f"crop_by_largest_contour(margin={crop_margin})")
        gray = cropped_gray

    # denoise leggero ma che preserva i bordi
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    steps.append("bilateral_filter")

        # blur + Otsu per ridurre il rumore di fondo e binarizzare
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    steps.append("gaussian_blur")

    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps.append("otsu_threshold")

    # opening per rimuovere linee sottili verticali/orizzontali tipiche dei bordi
    kernel_open = np.ones((1, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open, iterations=1)
    steps.append("morph_open_1x3")

    # closing leggero per chiudere piccoli buchi eventualmente introdotti
    kernel_close = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    steps.append("morph_close_2x2")

    return th, steps