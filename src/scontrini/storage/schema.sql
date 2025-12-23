PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS receipts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  captured_at TEXT NOT NULL,
  image_path TEXT NOT NULL,
  merchant_name TEXT,
  merchant_address TEXT,
  vat_id TEXT,
  receipt_datetime TEXT,
  document_number TEXT,
  currency TEXT NOT NULL,
  total REAL,
  ocr_text TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS receipt_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  receipt_id INTEGER NOT NULL,
  raw_line TEXT,
  name TEXT,
  quantity REAL,
  unit TEXT,
  unit_price REAL,
  total_price REAL,
  vat_rate REAL,
  FOREIGN KEY(receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
);
