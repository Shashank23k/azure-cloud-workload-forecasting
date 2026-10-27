"""Portable SHA-256 integrity checker for the fixed Atomity data-v1 release.

Usage: python -m cwfc.verify_data --data challenge_data
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def verify_checksums(data_dir: str | Path) -> dict[str, bool]:
    data_dir = Path(data_dir)
    manifest = data_dir / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"SHA256SUMS not found: {manifest}")
    results: dict[str, bool] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, separator, name = line.partition(" ")
        if not separator:
            raise ValueError(f"Bad SHA256SUMS entry: {line}")
        name = name.strip().lstrip("*")
        if (not name or Path(name).name != name or name in {".", ".."}
                or "\\" in name):
            raise ValueError(f"Unsafe or invalid SHA256SUMS entry: {name}")
        file_path = data_dir / name
        if not file_path.is_file():
            results[name] = False
            continue
        digest = hashlib.sha256()
        with file_path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        results[name] = digest.hexdigest().lower() == expected.lower()
    if not results:
        raise ValueError("Empty SHA256SUMS manifest")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the dataset against SHA256SUMS")
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    args = parser.parse_args()
    results = verify_checksums(args.data)
    for name, ok in results.items():
        print(f"{name}: {'OK' if ok else 'FAILED'}")
    if not all(results.values()):
        raise SystemExit("Dataset checksum verification failed")


if __name__ == "__main__":
    main()
