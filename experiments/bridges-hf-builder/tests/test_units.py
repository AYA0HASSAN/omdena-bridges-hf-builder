"""Small unit tests: licence filter and OAM metadata parsing (fake HTTP)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridgebuild import net, sources


def test_licences():
    ok = ["CC-BY 4.0", "CC BY 4.0", "CC-BY-SA 4.0", "CC0"]
    assert sources.licence_allowed("CC-BY 4.0", ok)
    assert sources.licence_allowed("cc-by 4.0", ok)
    assert not sources.licence_allowed("CC-BY-NC 4.0", ok)
    assert not sources.licence_allowed("CC BY-NC-SA 4.0", ok)
    assert not sources.licence_allowed("", ok)
    assert not sources.licence_allowed("All rights reserved", ok)


class _R:
    status_code = 200
    def __init__(self, js): self._js = js
    def json(self): return self._js
    def raise_for_status(self): pass


def test_oam_parsing():
    res = [{"_id": "a", "uuid": "https://x/a.tif", "gsd": 0.1, "properties": {"license": "CC-BY 4.0"}, "bbox": [0, 0, 1, 1]},
           {"_id": "b", "uuid": "https://x/b.tif", "gsd": 0.1, "properties": {"license": "CC-BY-NC 4.0"}, "bbox": [0, 0, 1, 1]},
           {"_id": "c", "uuid": "https://x/c.tif", "gsd": 2.0, "properties": {"license": "CC-BY 4.0"}, "bbox": [0, 0, 1, 1]},
           {"_id": "d", "uuid": "https://x/d.jpg", "gsd": 0.1, "properties": {"license": "CC-BY 4.0"}, "bbox": [0, 0, 1, 1]}]

    class Fake:
        @staticmethod
        def get(url, **kw): return _R({"meta": {"found": 4}, "results": res})
    old = net.requests; net.requests = Fake
    try:
        rows = sources.oam_images((0, 0, 1, 1), 0.6, ["CC-BY 4.0"])
    finally:
        net.requests = old
    assert [r["image_id"] for r in rows] == ["a"]


if __name__ == "__main__":
    test_licences(); test_oam_parsing(); print("UNIT TESTS PASSED")
