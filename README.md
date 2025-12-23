# CO2
A tool for calculating a company's carbon footprint

# scontrini

Pipeline locale:
1) Foto scontrino (jpg/png)
2) Preprocess OpenCV
3) OCR Tesseract (pytesseract)
4) Parsing euristico (merchant, items, totals)
5) Salvataggio SQLite
6) API FastAPI (base per WebView Android/Windows)

## Setup (Windows / VS Code)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
