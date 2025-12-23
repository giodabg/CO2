"""
@file main.py
@brief Entry point FastAPI.
@ingroup api_module
"""

from __future__ import annotations
from fastapi import FastAPI
from .routes import router

app = FastAPI(title="Scontrini API", version="0.1.0")
app.include_router(router)
