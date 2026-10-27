import hashlib

from cwfc.verify_data import verify_checksums


def test_verify_checksum_manifest(tmp_path):
    (tmp_path / "sample.txt").write_text("example", encoding="utf-8")
    digest = hashlib.sha256(b"example").hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  sample.txt\n", encoding="utf-8")
    assert verify_checksums(tmp_path) == {"sample.txt": True}
    (tmp_path / "sample.txt").write_text("tampered", encoding="utf-8")
    assert verify_checksums(tmp_path) == {"sample.txt": False}
