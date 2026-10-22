"""EVIE-4.5B fp32 and fp16 numerics against the torch fp32 oracle fixture.

Reports, per precision, embedding cosine and absolute error for queries,
documents and visual features; the MaxSim score error; how many query tokens
select the same document token as the oracle (and, where one differs, the
oracle's own margin there); and batched-versus-sequential deltas. Prints JSON
and optionally writes it.

Usage:
    python3 scripts/evie_numerics_report.py [--out report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "evie" / "evie_4p5b_doc_query.npz"
MODEL_ID = "tencent/EVIE-4.5B"


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.sum(a * b, axis=-1)


def _compare(got: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    got, ref = got.astype(np.float64), ref.astype(np.float64)
    cos = _cos_rows(got, ref)
    diff = np.abs(got - ref)
    return {
        "cos_mean": float(cos.mean()),
        "cos_min": float(cos.min()),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
    }


def _maxsim_selection(query: np.ndarray, doc: np.ndarray, ref_q: np.ndarray, ref_d: np.ndarray) -> dict:
    ref = ref_q.astype(np.float64) @ ref_d.astype(np.float64).T
    got = query.astype(np.float64) @ doc.astype(np.float64).T
    ranked = np.sort(ref, axis=1)
    changed = []
    for token in range(ref.shape[0]):
        want, have = int(ref[token].argmax()), int(got[token].argmax())
        if want != have:
            changed.append({
                "query_token": token,
                "oracle_choice": want,
                "selected": have,
                "oracle_gap_to_selected": float(ref[token, want] - ref[token, have]),
            })
    return {
        "query_tokens": int(ref.shape[0]),
        "same_choice": int(ref.shape[0] - len(changed)),
        "changed": changed,
        "oracle_smallest_top2_margin": float((ranked[:, -1] - ranked[:, -2]).min()),
    }


def _measure(precision: str, fx: dict[str, np.ndarray], snapshot: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    from hipengine.loading.evie import load_evie_model
    from hipengine.runtime.evie import EvieRunner, maxsim

    runner = EvieRunner(load_evie_model(snapshot, runtime=None, precision=precision), precision=precision)
    try:
        q_ids, q_mask = fx["query_input_ids"][0], fx["query_attention_mask"][0]
        query = runner.encode_query(q_ids, q_mask)
        doc = runner.encode_document(
            fx["input_ids"][0], fx["attention_mask"][0], fx["pixel_values"][0], fx["image_grid_thw"],
        )
        n_img = int((fx["input_ids"][0] == runner.spec.image_token_id).sum())
        scratch = runner._scratch_for(max(len(fx["input_ids"][0]), len(fx["pixel_values"][0])))
        visual_buffer = runner.vision_forward(fx["pixel_values"][0], fx["image_grid_thw"], scratch)
        visual = runner._to_host(visual_buffer.ptr, n_img * 2560).reshape(n_img, 2560)
        repeats = runner.encode_queries([(q_ids, q_mask)] * 3)
        base = q_ids[5:25]
        segments = [(base[:n], np.ones(n, dtype=np.int64)) for n in (1, 2, 3, 5, len(base))]
        segment_out = runner.encode_queries(segments)
        segment_delta = max(
            float(np.abs(out - runner.encode_query(ids, mask)).max())
            for (ids, mask), out in zip(segments, segment_out)
        )
    finally:
        runner.close()
    score = float(maxsim(query, doc))
    ref_score = float(fx["maxsim_score"][0, 0])
    report = {
        "query": _compare(query, fx["query_embeddings_128"][0]),
        "document": _compare(doc, fx["doc_embeddings_128"][0]),
        "visual_features": _compare(visual, fx["visual_features"]),
        "maxsim": {"score": score, "oracle": ref_score, "abs_error": abs(score - ref_score),
                   "rel_error": abs(score - ref_score) / abs(ref_score)},
        "maxsim_selection": _maxsim_selection(
            query, doc, fx["query_embeddings_128"][0], fx["doc_embeddings_128"][0]
        ),
        "batching": {
            "three_repeats_max_abs_between_rows": float(max(np.abs(r - repeats[0]).max() for r in repeats)),
            "three_repeats_vs_single_max_abs": float(np.abs(repeats[0] - query).max()),
            "short_segments_vs_sequential_max_abs": segment_delta,
        },
    }
    return report, query, doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from hipengine.loading.hf_cache import resolve_model_path

    snapshot = resolve_model_path(MODEL_ID)
    with np.load(FIXTURE) as data:
        fx = {key: data[key] for key in data.files}
    report: dict = {"model": MODEL_ID, "snapshot": snapshot.name, "fixture": str(FIXTURE.name)}
    outputs = {}
    for precision in ("fp32", "fp16"):
        report[precision], query, doc = _measure(precision, fx, snapshot)
        outputs[precision] = (query, doc)
    report["fp16_vs_fp32"] = {
        "query": _compare(outputs["fp16"][0], outputs["fp32"][0]),
        "document": _compare(outputs["fp16"][1], outputs["fp32"][1]),
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
