
---

## Test minimi

### `tests/test_parsing.py`
```python
from scontrini.domain.parsing import parse_totals

def test_parse_totals():
    text = "TOTALE 12,34"
    totals = parse_totals(text)
    assert totals.total == 12.34
