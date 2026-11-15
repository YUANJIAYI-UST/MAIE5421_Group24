from transparent_seg.prepare import write_json


def test_write_json_is_utf8_lf_on_all_platforms(tmp_path):
    path = tmp_path / "portable.json"
    write_json(path, {"label": "透明", "values": [1, 2]})
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert "透明" in raw.decode("utf-8")
