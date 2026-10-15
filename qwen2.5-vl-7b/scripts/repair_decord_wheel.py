"""Normalize Decord 0.6's incorrectly tagged, ABI-independent Linux wheel.

The official wheel is named py3-none-manylinux2010_x86_64, but its internal
WHEEL metadata says cp36-cp36m. Only WHEEL and its RECORD entry are updated;
the Python decoder and libdecord.so are never changed.
"""
import base64
import csv
import hashlib
from importlib import metadata
import io
from pathlib import Path
import platform
import sys

OLD_TAG = "Tag: cp36-cp36m-manylinux2010_x86_64"
CORRECT_TAG = "Tag: py3-none-manylinux2010_x86_64"


def repair_distribution(dist):
    if dist.version != "0.6.0":
        raise ValueError("This compatibility fix is restricted to decord==0.6.0")
    files = list(dist.files or [])
    wheel_name = next(p for p in files if str(p).endswith(".dist-info/WHEEL"))
    record_name = next(p for p in files if str(p).endswith(".dist-info/RECORD"))
    wheel = Path(dist.locate_file(wheel_name))
    record = Path(dist.locate_file(record_name))
    original = wheel.read_bytes()
    text = original.decode("utf-8")
    tags = [line for line in text.splitlines() if line.startswith("Tag: ")]
    if tags not in ([OLD_TAG], [CORRECT_TAG]):
        raise ValueError(f"Unrecognized Decord wheel tags; refusing to modify: {tags}")
    updated = text.replace(OLD_TAG, CORRECT_TAG).encode("utf-8")
    digest = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(updated).digest()).rstrip(b"=").decode()
    rows = list(csv.reader(io.StringIO(record.read_text())))
    matched = [row for row in rows if row[0] == str(wheel_name)]
    if len(matched) != 1:
        raise ValueError("Missing or duplicate WHEEL entry in RECORD")
    matched[0][1:] = [digest, str(len(updated))]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    record_bytes = output.getvalue().encode("utf-8")
    changed = original != updated or record.read_bytes() != record_bytes
    if changed:
        wheel.write_bytes(updated)
        record.write_bytes(record_bytes)
    return changed


def main():
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise SystemExit("This metadata fix applies only to the Linux x86_64 Decord wheel")
    changed = repair_distribution(metadata.distribution("decord"))
    print("[OK] Decord wheel metadata " + ("normalized" if changed else "already correct"))


if __name__ == "__main__":
    main()
