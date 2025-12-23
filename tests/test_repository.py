from scontrini.storage.db import connect
from scontrini.domain.models import ReceiptContractV1, Source

def test_insert_receipt(tmp_path):
    db = tmp_path / "t.sqlite"
    conn = connect(str(db))
    contract = ReceiptContractV1(source=Source(image_path="x.jpg", captured_at="2025-12-23T10:20:00+01:00"))
    from scontrini.storage.repository import insert_receipt
    rid = insert_receipt(conn, contract)
    assert rid > 0
