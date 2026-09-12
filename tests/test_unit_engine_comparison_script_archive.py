"""Check historical script bytes without executing GPU/network side effects."""

import hashlib
from pathlib import Path


ARCHIVE = Path(__file__).resolve().parents[1] / "scripts/experiments/2026-09-08-engine-comparison"


def test_all_archived_scripts_match_original_bytes_and_compile():
    entries = {}
    for line in (ARCHIVE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert name not in entries
        assert Path(name).name == name
        entries[name] = digest
    assert len(entries) == 19
    assert set(entries) == {path.name for path in ARCHIVE.glob("*.py")}
    for name, digest in entries.items():
        source = (ARCHIVE / name).read_bytes()
        assert hashlib.sha256(source).hexdigest() == digest, name
        compile(source, str(ARCHIVE / name), "exec")
