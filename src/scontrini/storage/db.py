"""
@file db.py
@brief Connessione SQLite e inizializzazione schema.
@ingroup storage_module

@details
Centralizza la creazione della connessione e l'invocazione init_schema().
"""

from __future__ import annotations
import sqlite3
from pathlib import Path

from .repository import init_schema


def connect(db_path: str) -> sqlite3.Connection:
    """
    @brief Apre una connessione SQLite e garantisce schema pronto.
    @param db_path Path al file SQLite.
    @return sqlite3.Connection con row_factory impostata.

    @note Crea automaticamente la directory padre se non esiste.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    return conn
