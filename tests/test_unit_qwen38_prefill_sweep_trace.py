from scripts.qwen38_prefill_sweep_analyze import analyze, classify_kernel


def test_classify_prefill_kernel_families() -> None:
    assert classify_kernel("gguf_q4_t16_dense_wmma_prefill") == "q4"
    assert classify_kernel("gguf_q5_t16_dense_wmma_prefill") == "q5"
    assert classify_kernel("q6_k_t16_qmicro_planar_wmma_prefill") == "q6"
    assert classify_kernel("Cijk_Alik_Bljk_HB_MT128x128x32") == "q5"
    assert classify_kernel("fp16_to_bf16_strided_rows_kernel") == "q5"
    assert classify_kernel("qwen35_gdn_prefill_recurrent") == "gdn"
    assert classify_kernel("rmsnorm_bf16") == "other"


def test_analyze_sizes_q4_single_sweep_bound() -> None:
    raw = {
        "kind": "qwen38-prefill-sweep-trace-raw",
        "model": "/model.gguf",
        "model_sha256": "abc",
        "prompts": "/prompts.jsonl",
        "host": "test-host",
        "backend": "hip_gfx1151",
        "token_source": {"sha256": "tokens"},
        "model_family_inventory": {
            "q4": {"resident_bytes": 1600, "logical_elements": 3200},
            "q5": {"resident_bytes": 0, "logical_elements": 0},
            "q6": {"resident_bytes": 0, "logical_elements": 0},
            "other": {"resident_bytes": 0, "logical_elements": 0},
        },
        "records": [
            {
                "rows": 32,
                "start_monotonic_ns": 100,
                "stop_monotonic_ns": 10_000_100,
                "wall_ms": 10.0,
                "gpu_span_ms": 10.0,
                "wall_minus_gpu_ms": 0.0,
                "weight_ledger": {
                    "q4": {
                        "active_weight_bytes": 1600,
                        "logical_row_elements": 102400,
                        "entries": [
                            {
                                "active_weight_bytes": 1600,
                                "logical_elements": 3200,
                                "rows": 32,
                                "weight_count": 1,
                            }
                        ],
                    },
                    "q5": {
                        "active_weight_bytes": 0,
                        "logical_row_elements": 0,
                        "entries": [],
                    },
                    "q6": {
                        "active_weight_bytes": 0,
                        "logical_row_elements": 0,
                        "entries": [],
                    },
                    "other": {
                        "active_weight_bytes": 0,
                        "logical_row_elements": 0,
                        "entries": [],
                    },
                },
            }
        ],
    }
    dispatches = [
        {
            "name": "gguf_q4_t16_dense_wmma_prefill",
            "family": "q4",
            "start_ns": 100,
            "stop_ns": 8_000_100,
            "grid_y": 2,
        },
        {
            "name": "rmsnorm",
            "family": "other",
            "start_ns": 8_000_100,
            "stop_ns": 10_000_100,
            "grid_y": 1,
        },
    ]

    point = analyze(raw, dispatches)["points"][0]

    assert point["families"]["q4"]["sweep_multiplicity"] == 2
    assert point["families"]["q4"]["swept_weight_bytes"] == 3200
    assert point["y1_q4_single_sweep_upper_bound"]["savings_ms"] == 4.0
    assert point["y1_q4_single_sweep_upper_bound"]["projected_tick_wall_ms"] == 6.0
