"""EVIE-8B fp16 tier: deterministic, with coarse agreement with fp32.

This test loads an fp32 and then an fp16 EVIE-8B, one at a time. It lives
apart from ``test_live_evie_gpu_runtime_8b.py`` because that module keeps a
module-scoped fp32 runner (about 33 GiB) resident for its tests; a third load
next to it does not fit a 48 GB card. Module-scoped fixtures are torn down
before the next module starts, so here the card is free. The fp16 accuracy
limits are described in that module's docstring.
"""

from __future__ import annotations

import numpy as np

from tests.test_live_evie_gpu_runtime_8b import FIXTURE, _cos_rows, _runner


def test_8b_fp16_doc_quality_tier() -> None:
    """fp16 tier: deterministic within a process, coarse agreement with fp32.

    The 8B vision tower amplifies fp16 storage noise (doc cos min can
    dip to ~0.5 on a few tokens — see the module docstring); the gate is
    the attainable floor, not 4.5B-class parity.
    """

    with np.load(FIXTURE) as fx:
        embs = {}
        for prec in ("fp32", "fp16"):
            r = _runner(prec)
            d1 = r.encode_document(
                fx["input_ids"][0],
                fx["attention_mask"][0],
                fx["pixel_values"][0],
                fx["image_grid_thw"],
            )
            d2 = r.encode_document(
                fx["input_ids"][0],
                fx["attention_mask"][0],
                fx["pixel_values"][0],
                fx["image_grid_thw"],
            )
            r.close()
            assert np.isfinite(d1).all()
            assert np.array_equal(d1, d2), "doc encode must be deterministic"
            embs[prec] = d1
        c = _cos_rows(embs["fp32"], embs["fp16"])
        assert c.mean() >= 0.98, f"fp16-vs-fp32 doc cos mean {c.mean()}"
