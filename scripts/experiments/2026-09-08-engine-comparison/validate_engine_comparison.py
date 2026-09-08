import hashlib
import json
import math
from pathlib import Path

path = Path("/home/lhl/hipEngine-main/benchmarks/results/2026-09-08-rx7900xtx-engine-comparison.json")
x = json.loads(path.read_text())
assert x["status"] == "complete"
assert len(x["pp_tg_summary"]) == 6
assert len(x["mtp_summary"]) == 7
assert len(x["capacity"]) == 12
assert x["strix_source"]["commit"] == "5f851647fe5ed795dfd6c0a3fba543114879e874"
for digest, ids in x["output_id_rows"].items():
    actual = hashlib.sha256(b"".join(t.to_bytes(8, "little", signed=True) for t in ids)).hexdigest()
    assert digest == actual
    assert all(type(t) is int and 0 <= t < 248320 for t in ids)


def walk(value):
    if isinstance(value, dict):
        if "output_ids_ref" in value:
            ids = x["output_id_rows"][value["output_ids_ref"]]
            assert len(ids) == value["output_count"]
            assert len(ids) - 1 == value["transitions"]
            assert value["decode_seconds"] > 0
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


walk(x)
for row in x["mtp_summary"].values():
    for arm in ("ar", "mtp"):
        assert math.isclose(row[arm]["tok_s"], row[arm]["transitions"] / row[arm]["decode_seconds"], rel_tol=1e-12)
    assert math.isclose(row["ratio"], row["mtp"]["tok_s"] / row["ar"]["tok_s"], rel_tol=1e-12)
    assert row["exact_matches"] == sum(c["exact"] for c in row["checks"])
for evidence in x["runs"].values():
    if "monitor" in evidence and "foreign_gpu1_owners" in evidence["monitor"]:
        assert not evidence["monitor"]["foreign_gpu1_owners"]
        assert evidence["monitor"]["source_clean"]
assert x["mtp_summary"]["hipEngine-short"]["exact_matches"] == 30
assert x["mtp_summary"]["hipEngine-long"]["exact_matches"] == 10
print("PASS: integrity, output IDs, denominators, ratios, ownership, and completeness")
