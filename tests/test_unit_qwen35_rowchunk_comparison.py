"""Host-only contracts for the two-batched-peer rowchunk diagnostic."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qwen35_batch_hidden_bisect as bisect


@pytest.mark.parametrize("width,chunk", [(3, 2), (6, 2), (7, 3)])
def test_rowchunk_comparison_runs_two_batch_peers_with_explicit_labels(monkeypatch, width, chunk):
    monkeypatch.setattr(bisect.os, "environ", dict(bisect.os.environ))
    monkeypatch.setattr(bisect, "_load_prompt_slices", lambda *a, **kw: [[1, 2]] * width)
    monkeypatch.setattr(bisect, "_compiler_version", lambda *a: "synthetic")
    monkeypatch.setattr(bisect, "Qwen35ParoNextTokenRunner", lambda *a: SimpleNamespace(config=SimpleNamespace(layer_types=())))
    monkeypatch.setattr(bisect, "_run_c1_hidden", lambda *a, **kw: pytest.fail("comparison must not run independent c1"))
    size_env = "HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_SIZE"
    layers_env = "HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_LAYERS"
    calls = []

    def batch_run(runner, prompts, **kwargs):
        calls.append((bisect.os.environ[size_env], bisect.os.environ[layers_env], kwargs))
        assert len(prompts) == width
        return bisect.HiddenRun(
            seed_tokens=[1] * width,
            generated_tokens=[[2] for _ in range(width)],
            hidden_bits_by_step=[np.zeros((width, 4), dtype=np.uint16)],
        )

    monkeypatch.setattr(bisect, "_run_batch_hidden", batch_run)
    argv = [
        "--batch-size", str(width), "--decode-tokens", "1", "--max-layers", "4",
        "--layer-limits", "1,4", "--batch-decode-full-attn-path", "native_batch",
        "--batch-decode-full-attn-row-chunk-size", str(chunk),
        "--batch-decode-full-attn-row-chunk-layers", "3",
        "--compare-full-attn-rowchunk-boundary",
    ]
    payload = bisect.run(bisect.build_parser().parse_args(argv), argv)
    assert [(size, layers) for size, layers, _ in calls] == [("0", ""), (str(chunk), "3")] * 2
    assert [kwargs["layer_limit"] for _, _, kwargs in calls] == [1, 1, 4, 4]
    assert calls[0][2] == calls[1][2] and calls[2][2] == calls[3][2]
    assert bisect.os.environ[size_env] == str(chunk)
    assert bisect.os.environ[layers_env] == "3"
    assert payload["mode"] == "qwen35_paro_native_hidden_bisect_rowchunk_boundary"
    assert payload["status"] == "eq_ok" and payload["correctness"]["passed"] is True
    assert payload["performance_claim"] is False
    labels = {"batch": "native_no_rowchunk", "c1": "native_full_attention_rowchunk_repair"}
    assert payload["workload"]["comparison_variant_labels"] == labels
    for summary in payload["layer_summaries"]:
        assert summary["comparison_kind"] == "full_attention_rowchunk_boundary"
        assert summary["variant_labels"] == labels


@pytest.mark.parametrize("extra,error", [
    (["--batch-decode-full-attn-row-chunk-size", "0"], "requires native_batch"),
    (["--batch-decode-full-attn-row-chunk-size", "6"], "requires native_batch"),
    (["--batch-decode-full-attn-path", "per_row"], "requires native_batch"),
    (["--batch-decode-full-attn-context-row-chunk-size", "2"], "cannot be combined"),
])
def test_rowchunk_comparison_rejects_incompatible_modes_without_runtime(monkeypatch, extra, error):
    monkeypatch.setattr(bisect, "Qwen35ParoNextTokenRunner", lambda *a: pytest.fail("dry-run loaded model"))
    argv = [
        "--dry-run", "--batch-size", "6", "--batch-decode-full-attn-path", "native_batch",
        "--batch-decode-full-attn-row-chunk-size", "2", "--compare-full-attn-rowchunk-boundary", *extra,
    ]
    with pytest.raises(ValueError, match=error):
        bisect.run(bisect.build_parser().parse_args(argv), argv)
