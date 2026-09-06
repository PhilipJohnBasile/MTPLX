from pathlib import Path
import sys

root = Path(sys.argv[1])
if int(sys.argv[2]) == 219:
    path = root / 'tests/test_request_observability_golden.py'
    source = path.read_text()
    old = '        {"x-mtplx-client-turn-id": "turn-123", "x-mtplx-client-entry-id": "bad value\\nsecret"},\n'
    new = '        headers={"x-mtplx-client-turn-id": "turn-123", "x-mtplx-client-entry-id": "bad value\\nsecret"},\n        metadata={}, session_source=None, request_generation_mode="mtp", request_depth=1,\n'
    assert source.count(old) == 1
    path.write_text(source.replace(old, new))
