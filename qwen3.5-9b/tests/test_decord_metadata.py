"""The Decord compatibility step must never alter decoder code or binaries."""
import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decord_metadata_fix", ROOT / "scripts/repair_decord_wheel.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def fixture_distribution(tmp_path, tag=None):
    dist_dir = tmp_path / "decord-0.6.0.dist-info"
    dist_dir.mkdir()
    wheel = dist_dir / "WHEEL"
    wheel.write_text("Wheel-Version: 1.0\n" + (tag or repair.OLD_TAG) + "\n")
    record = dist_dir / "RECORD"
    record.write_text("decord-0.6.0.dist-info/WHEEL,,\ndecord-0.6.0.dist-info/RECORD,,\n")
    binary = tmp_path / "libdecord.so"
    binary.write_bytes(b"unchanged decoder fixture")
    dist = SimpleNamespace(version="0.6.0",
        files=[Path("decord-0.6.0.dist-info/WHEEL"), Path("decord-0.6.0.dist-info/RECORD")],
        locate_file=lambda p: tmp_path / p)
    return dist, wheel, record, binary


def test_metadata_repair_preserves_decoder_and_record(tmp_path):
    dist, wheel, record, binary = fixture_distribution(tmp_path)
    original = binary.read_bytes()
    assert repair.repair_distribution(dist)
    assert repair.CORRECT_TAG in wheel.read_text()
    row = next(csv.reader(io.StringIO(record.read_text())))
    digest = base64.urlsafe_b64encode(hashlib.sha256(wheel.read_bytes()).digest()).rstrip(b"=").decode()
    assert row[1:] == ["sha256=" + digest, str(len(wheel.read_bytes()))]
    assert binary.read_bytes() == original
    assert not repair.repair_distribution(dist)


def test_unknown_wheel_is_not_modified(tmp_path):
    dist, wheel, record, _ = fixture_distribution(tmp_path, "Tag: cp36-cp36m-win_amd64")
    original = wheel.read_bytes(), record.read_bytes()
    with pytest.raises(ValueError, match="Unrecognized"):
        repair.repair_distribution(dist)
    assert (wheel.read_bytes(), record.read_bytes()) == original
