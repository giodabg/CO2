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
import re
import pytesseract

_KEYWORDS_RE = re.compile(r"\b(TOTALE|IVA|EURO|DOC|DOCUMENTO|P\.?\s*IVA|IMPORTO|PAGATO)\b", re.I)

def _rotation_candidates(img: np.ndarray) -> list[tuple[int, np.ndarray]]:
    # img: BGR o gray
    return [
        (0, img),
        (90, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(img, cv2.ROTATE_180)),
        (270, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

def _score_ocr_text(text: str) -> int:
    # score cheap basato su keyword + quantità di token utili
    tokens = re.findall(r"[A-Z0-9]{3,}", text.upper())
    kw = len(_KEYWORDS_RE.findall(text))
    return len(tokens) + kw * 10

def auto_rotate_for_ocr(img: np.ndarray, lang: str = "ita") -> tuple[np.ndarray, int]:
    """
    Ritorna (img_ruotata, angolo_gradi) scegliendo la rotazione con OCR migliore.
    """
    best_score = -1
    best_angle = 0
    best_img = img

    # Config OCR veloce per scoring (non serve perfetto)
    config = "--oem 1 --psm 6"

    for angle, cand in _rotation_candidates(img):
        # leggero boost contrasto per scoring (opzionale)
        gray = cand if len(cand.shape) == 2 else cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)

        # OCR rapido
        txt = pytesseract.image_to_string(gray, lang=lang, config=config)

        score = _score_ocr_text(txt)
        if score > best_score:
            best_score = score
            best_angle = angle
            best_img = cand

    return best_img, best_angle

def deskew(gray: np.ndarray) -> np.ndarray:
    # gray: uint8
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect

def extract_and_warp_receipt(bgr: np.ndarray) -> np.ndarray:
    """
    Estrae lo scontrino dallo sfondo e lo raddrizza con prospettiva.
    Se non trova un contorno affidabile, ritorna l'immagine originale.
    """
    img = bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection robusto
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Cerca un contorno approssimabile a quadrilatero
    receipt_cnt = None
    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_cnt = approx
            break

    if receipt_cnt is None:
        # fallback: usa minAreaRect sul contorno più grande
        c = cnts[0]
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        receipt_cnt = box.astype(np.int32)

    pts = receipt_cnt.reshape(4, 2).astype("float32")
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    if maxW < 200 or maxH < 200:
        # contorno troppo piccolo => probabilmente sbagliato
        return bgr

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))

    return warped

def ensure_vertical_receipt(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w > h:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    return bgr

"""
def preprocess_for_ocr(
    image_bgr: np.ndarray,
    *,
    enable_crop: bool = True,
    crop_margin: int = 8,
) -> tuple[np.ndarray, list[str]]:

    @brief Applica una pipeline di pre-processing ad un'immagine BGR.
    @param image_bgr Immagine in formato BGR (tipico output di cv2.imread).
    @param enable_crop Se True prova a ritagliare lo scontrino rilevando il contorno principale.
    @param crop_margin Margine (px) aggiuntivo attorno al contorno prima del crop.    @return Tuple (image_bin, steps) dove:
    @return Tuple (image_bin, steps) dove:
            - image_bin è un'immagine binarizzata pronta per OCR
            - steps è l'elenco degli step applicati in ordine

    @throws ValueError Se l'immagine è vuota o non valida (estendibile).
    @note La pipeline è deliberatamente semplice; è prevista evoluzione.

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Immagine vuota o non valida")

    steps: list[str] = []

    img = extract_and_warp_receipt(image_bgr)
    steps.append("warp_receipt")

    img = ensure_vertical_receipt(img)
    steps.append("ensure_vertical")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append("to_gray")

    # (opzionale ma utile con ombre) CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    steps.append("clahe")

    # refine crop (ora è molto più affidabile)
    if enable_crop:
        gray2, cropped = _crop_by_largest_contour(gray, margin=crop_margin)
        if cropped:
            steps.append(f"crop_by_largest_contour(margin={crop_margin})")
            gray = gray2

    # denoise
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    steps.append("bilateral_filter")

    # (opzionale) rotazione grossolana e deskew fine
    gray, angle = auto_rotate_for_ocr(gray, lang="ita")
    steps.append(f"auto_rotate:{angle}")

    gray = deskew(gray)
    steps.append("deskew")

    # blur + otsu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    steps.append("gaussian_blur")
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps.append("otsu_threshold")

    # morph open/close
    kernel_open = np.ones((1, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open, iterations=1)
    steps.append("morph_open_1x3")

    kernel_close = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    steps.append("morph_close_2x2")

    return th, steps
    """

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


def _normalize_illumination(gray: np.ndarray, ksize: int = 51) -> np.ndarray:
    """
    @brief Compensa ombre lente stimando il background e dividendo.
    @param gray Immagine in scala di grigi.
    @param ksize Kernel del filtro median blur usato per stimare lo sfondo (deve essere dispari).
    @return Immagine equalizzata rispetto al background.

    @note ksize elevati (>=51) aiutano a rimuovere ombre ampie senza intaccare il testo.
    """
    if ksize % 2 == 0:
        ksize += 1  # garantisce kernel dispari
    bg = cv2.medianBlur(gray, ksize)
    normalized = cv2.divide(gray, bg, scale=255)
    return normalized


def preprocess_for_ocr(
    image_bgr: np.ndarray,
    *,
    enable_crop: bool = True,
    crop_margin: int = 8,
    auto_rotate_landscape: bool = True,
    normalize_illumination: bool = True,
    illumination_ksize: int = 51,
    upscale_min_dim: int | None = 1800,
) -> tuple[np.ndarray, list[str]]:
    """
    @brief Applica una pipeline di pre-processing ad un'immagine BGR.
    @param image_bgr Immagine in formato BGR (tipico output di cv2.imread).
    @param enable_crop Se True prova a ritagliare lo scontrino rilevando il contorno principale.
    @param crop_margin Margine (px) aggiuntivo attorno al contorno prima del crop.    @return Tuple (image_bin, steps) dove:
    @param auto_rotate_landscape Se True ruota di 90° se l'immagine è orizzontale (receipt sdraiato).
    @param normalize_illumination Se True compensa ombre lente (utile per ombre parziali).
    @param illumination_ksize Kernel (dispari) per stimare il background durante la normalizzazione.
    @param upscale_min_dim Se impostato, porta il lato lungo ad almeno questo valore (px) per testi piccoli.
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

    h, w = gray.shape[:2]
    if auto_rotate_landscape and w > h:
        gray = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        steps.append("rotate_landscape_to_portrait")

    if normalize_illumination:
        gray = _normalize_illumination(gray, ksize=illumination_ksize)
        steps.append(f"normalize_illumination(ksize={illumination_ksize})")

    if upscale_min_dim is not None:
        h, w = gray.shape[:2]
        long_edge = max(h, w)
        if long_edge < upscale_min_dim:
            scale = upscale_min_dim / float(long_edge)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            steps.append(f"upscale_to_min_dim({upscale_min_dim})")

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