"""Boundary contract for the Prometheus metrics owner.

The counters, snapshot normalization and exposition were extracted from
``hipengine/server/api.py`` into ``hipengine/server/metrics.py``. These checks
pin four properties:

* the owner imports only the standard library and takes explicit data, never
  the HTTP application, the server engine, generation scheduling or mutable
  application state;
* the ``hipengine.server.api`` private names still resolve to the owner objects;
* the pre-move corpus (captured from ``api.py`` at ``c16bc3daf`` before the
  move) renders byte-for-byte identically;
* absent, full and malformed optional inputs keep the established defaults,
  counter accumulation and label escaping.

The corpus and AST hashes in
``tests/fixtures/rf7_metrics_premove_corpus.json`` are an oracle captured from
the pre-move implementation, not a restatement of the new module.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from hipengine.server import metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "hipengine" / "server" / "metrics.py"
API_PATH = REPO_ROOT / "hipengine" / "server" / "api.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "rf7_metrics_premove_corpus.json"

BLOCKED_IMPORT_PREFIXES = (
    "fastapi",
    "starlette",
    "torch",
    "hipengine.server.api",
    "hipengine.generation",
    "hipengine.runtime",
    "hipengine.models",
    "hipengine.kvcache",
    "hipengine.speculative",
    "hipengine.llm",
    "hipengine.kernels",
    "hipengine.dispatch",
    "hipengine.loading",
    "hipengine.quant",
)

API_METRICS_IMPORTS = (
    "_MetricsSnapshot",
    "_ServerMetrics",
    "_execution_manifest_metric_values",
    "_generation_queue_metric_values",
    "_graph_bucket_metric_values",
    "_kv_pool_metric_values",
    "_nested_mapping",
    "_prefix_cache_metric_values",
    "_render_prometheus_metrics",
    "_resident_loop_metric_values",
    "_scheduler_metric_values",
)


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[value["__float__"]]
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


CORPUS = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _snapshot(case):
    raw = _decode(case["snapshot"])
    raw.pop("pool_source", None)
    raw.pop("graph_source", None)
    raw_resident = raw.get("resident")
    resident = metrics._resident_loop_metric_values(raw_resident)
    runner = raw_resident.get("runner") if isinstance(raw_resident, dict) else None
    routes = runner.get("routes") if isinstance(runner, dict) else None
    manifest = (
        routes.get("last_execution_manifest")
        if isinstance(routes, dict)
        else None
    )
    resident["last_execution_manifest"] = metrics._execution_manifest_metric_values(
        manifest
    )
    return metrics._MetricsSnapshot(
        resident=resident,
        pool=metrics._kv_pool_metric_values(raw.get("pool")),
        graph=metrics._graph_bucket_metric_values(
            raw.get("graph_buckets") or {},
            raw.get("graph_stats"),
            kernel_time_buckets=raw.get("graph_kernel_time_buckets") or (),
        ),
        queue=metrics._generation_queue_metric_values(raw.get("queue")),
        chat_sessions_active=raw.get("chat_sessions_active"),
        chat_sessions_pending=raw.get("chat_sessions_pending"),
        chat_sessions_max_active=raw.get("chat_sessions_max_active"),
        mtp_serving=raw.get("mtp_serving", 0.0),
        scheduler_fairness=metrics._scheduler_metric_values(
            raw.get("scheduler_fairness")
        ),
    )


def _counters(case):
    counters = metrics._ServerMetrics()
    for op, payload in case["ops"]:
        if op == "success":
            counters.record_success(payload or {})
        else:
            getattr(counters, f"record_{op}")()
    return counters


def _metric_value(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(f"{name} "):
            return float(line.split(" ", 1)[1])
    raise AssertionError(f"{name} not found")


def test_module_depends_only_on_the_standard_library() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".", 1)[0])
        elif isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
    assert modules <= {"__future__", "collections", "copy", "dataclasses", "math", "types", "typing"}, (
        modules
    )


def test_module_file_loads_without_the_http_app_or_engine() -> None:
    probe = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location("
        f"'metrics_probe', {str(MODULE_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules['metrics_probe'] = module\n"
        "spec.loader.exec_module(module)\n"
        "assert not any(name == 'hipengine' or name.startswith('hipengine.') "
        "for name in sys.modules), [n for n in sys.modules if n.startswith('hipengine')]\n"
        "assert 'fastapi' not in sys.modules and 'starlette' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
        "counters = module._ServerMetrics()\n"
        "counters.record_success({'prompt_tokens': 1, 'completion_tokens': 2})\n"
        "text = module._render_prometheus_metrics("
        "counters, snapshot=module._MetricsSnapshot())\n"
        "assert 'hipengine_requests_total 1\\n' in text\n"
        "assert 'hipengine_mtp_serving 0\\n' in text\n"
        "print('METRICS_PROBE_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "METRICS_PROBE_OK"


def test_package_import_of_the_module_stays_metrics_only() -> None:
    blocked = "(" + ", ".join(repr(prefix) for prefix in BLOCKED_IMPORT_PREFIXES) + ",)"
    probe = (
        "import sys\n"
        "import hipengine.server.metrics as metrics\n"
        f"blocked = {blocked}\n"
        "loaded = [name for name in sys.modules if any(name == prefix or "
        "name.startswith(prefix + '.') for prefix in blocked)]\n"
        "assert loaded == [], loaded\n"
        "assert metrics._format_metric_value(1.0) == '1'\n"
        "print('METRICS_PACKAGE_IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "METRICS_PACKAGE_IMPORT_OK"


def test_api_uses_the_metrics_objects_without_duplicate_definitions() -> None:
    from hipengine.server import api

    for name in API_METRICS_IMPORTS:
        assert getattr(api, name) is getattr(metrics, name), name
        if inspect.isclass(getattr(metrics, name)) or callable(getattr(metrics, name)):
            assert getattr(api, name).__module__ == "hipengine.server.metrics", name


def test_api_imports_only_the_owner_names_it_uses() -> None:
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"), filename=str(API_PATH))
    imported: set[str] = set()
    loaded = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hipengine.server.metrics":
            imported.update(alias.name for alias in node.names)
    assert imported == set(API_METRICS_IMPORTS)
    assert imported <= loaded, sorted(imported - loaded)
    api_text = API_PATH.read_text(encoding="utf-8")
    for name in API_METRICS_IMPORTS:
        assert f"def {name}(" not in api_text, name
        assert f"class {name}" not in api_text, name


def test_snapshot_is_recursively_immutable() -> None:
    resident = {
        "loop": {"requests": {"pending": 1}},
        "rows": [1, {"slot": 2}],
        "pair": (3, 4),
        "flag": True,
        "ratio": 0.5,
        "none": None,
    }
    snapshot = metrics._MetricsSnapshot(resident=resident)

    # Later mutation of the collected source cannot reach the snapshot.
    resident["loop"]["requests"]["pending"] = 9
    resident["rows"][1]["slot"] = 9
    resident["new"] = 1
    assert snapshot.resident["loop"]["requests"]["pending"] == 1
    assert snapshot.resident["rows"][1]["slot"] == 2
    assert "new" not in snapshot.resident

    # Nested writes are refused at every level.
    with pytest.raises(TypeError):
        snapshot.resident["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.resident["loop"]["requests"]["pending"] = 9  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.resident["rows"][1]["slot"] = 9  # type: ignore[index]
    with pytest.raises(TypeError):
        snapshot.resident["rows"][0] = 9  # type: ignore[index]

    # Collection shape is preserved: lists stay list-shaped and tuples stay
    # tuples, so a thaw can restore the collected value type.
    assert isinstance(snapshot.resident["rows"], metrics._FrozenList)
    assert isinstance(snapshot.resident["pair"], tuple)
    assert not isinstance(snapshot.resident["pair"], metrics._FrozenList)
    assert snapshot.resident["flag"] is True
    assert snapshot.resident["ratio"] == 0.5
    assert snapshot.resident["none"] is None
    refrozen = metrics._MetricsSnapshot(resident=snapshot.resident)
    assert isinstance(refrozen.resident["rows"], metrics._FrozenList)
    with pytest.raises(FrozenInstanceError):
        snapshot.mtp_serving = 1.0  # type: ignore[misc]


def test_raw_extras_are_dropped_before_the_snapshot() -> None:
    opaque = object()
    raw = {"unused": opaque, "loop": {"requests": {"active": opaque}}}

    resident = metrics._resident_loop_metric_values(raw)
    manifest = metrics._execution_manifest_metric_values(
        {"kind": "manifest", "unused": opaque}
    )

    assert "unused" not in resident
    assert resident["requests"]["active"] == 0.0
    assert "unused" not in manifest
    assert manifest["kind"] == "manifest"

    snapshot = metrics._MetricsSnapshot(resident=resident)
    text = metrics._render_prometheus_metrics(
        metrics._ServerMetrics(), snapshot=snapshot
    )
    assert _metric_value(text, "hipengine_resident_requests_active") == 0.0


def test_falsey_scalar_labels_keep_pre_move_defaults() -> None:
    values = metrics._resident_loop_metric_values(
        {
            "loop": {
                "scheduler_policy": {
                    "prefill_decode_policy": "",
                    "last_work_kind": False,
                }
            },
            "runner": {
                "routes": {
                    "last_execution_manifest": {
                        "kind": 0,
                        "claim_level": False,
                    }
                }
            },
        }
    )
    assert values["policy"]["name"] == "unavailable"
    assert values["policy"]["last_work_kind"] == "none"
    manifest = metrics._execution_manifest_metric_values(
        values["last_execution_manifest"]
    )
    assert manifest["kind"] == "unavailable"
    assert manifest["claim_level"] == "unavailable"


def test_pre_move_object_extras_and_invalid_numeric_leaves_stay_benign() -> None:
    """Pre-move control captured from ``api.py`` at ``c16bc3daf``.

    ``unused`` and the route-count extra were ignored; the object-valued
    numeric leaves defaulted to 0; the object-valued worker-active flag stayed
    truthy; the active mask stayed empty. The immutable boundary must not turn
    any of those into a new ``/metrics`` failure.
    """

    opaque = object()
    resident = {
        "unused": opaque,
        "loop": {
            "requests": {"pending": 1, "active": opaque, "admitted_total": 2},
            "physical_bucket": {"capacity": 2, "active_mask": opaque},
            "work_counts": {"prefill": 1},
            "latency_seconds": {"queue": {"count": 1, "sum": 0.5}},
            "scheduler_policy": {
                "prefill_decode_policy": "protect_ttft",
                "prefill_chunk_tokens": 8,
            },
        },
        "runner": {
            "kv_pool": {"current_bytes": opaque, "grow_events": 1},
            "graph_buckets": {
                "entries": opaque,
                "hits_total": 2,
                "captures_total": 1,
            },
            "routes": {
                "counts": {"native": 1, "bad": opaque},
                "last_execution_manifest": {"kind": "manifest", "unused": opaque},
            },
        },
    }
    normalized_resident = metrics._resident_loop_metric_values(resident)
    normalized_resident["last_execution_manifest"] = (
        metrics._execution_manifest_metric_values(
            resident["runner"]["routes"]["last_execution_manifest"]
        )
    )
    snapshot = metrics._MetricsSnapshot(
        resident=normalized_resident,
        pool=metrics._kv_pool_metric_values(resident["runner"]["kv_pool"]),
        graph=metrics._graph_bucket_metric_values(
            resident["runner"]["graph_buckets"],
            None,
            kernel_time_buckets=CORPUS["graph_kernel_time_buckets"],
        ),
        queue=metrics._generation_queue_metric_values(
            {
                "depth": opaque,
                "max_depth": None,
                "worker_active": opaque,
                "active_requests": opaque,
                "max_active_requests": None,
            }
        ),
        scheduler_fairness=metrics._scheduler_metric_values(
            CORPUS["render_cases"][0]["snapshot"]["scheduler_fairness"]
        ),
    )
    counters = metrics._ServerMetrics()
    counters.record_success({"prompt_tokens": 1, "completion_tokens": 2})
    text = metrics._render_prometheus_metrics(counters, snapshot=snapshot)

    assert hashlib.sha256(text.encode()).hexdigest() == (
        "44001ebf539842a0c7d2712fbdab7b7f7ce8370213ca79b0e7a73e1b6eed1bf1"
    )
    assert "hipengine_resident_requests_active 0\n" in text
    assert "hipengine_kv_pool_current_bytes 0\n" in text
    assert "hipengine_graph_bucket_entries 0\n" in text
    assert "hipengine_generation_queue_depth 0\n" in text
    assert "hipengine_generation_worker_active 1\n" in text
    assert 'hipengine_resident_bucket_info{active_mask="",' in text
    assert (
        'hipengine_resident_route_manifest_info{claim_level="unavailable",'
        'kind="manifest",' in text
    )


def test_prefix_cache_payload_thaws_back_to_mutable_collections() -> None:
    resident = {
        "runner": {
            "prefix_cache": {
                "mode": "radix",
                "block_size_tokens": 256,
                "stats": {"hits": [1, 2]},
                "phase_calls": {"prefill": 3},
            }
        }
    }
    snapshot = metrics._MetricsSnapshot(resident=resident)
    resident["runner"]["prefix_cache"]["stats"]["hits"][0] = 9
    payload = metrics._prefix_cache_metric_values(snapshot.resident)
    assert payload["stats"] == {"hits": [1, 2]}
    assert isinstance(payload["stats"]["hits"], list)
    payload["stats"]["hits"].append(3)
    payload["phase_calls"]["prefill"] = 4
    assert snapshot.resident["runner"]["prefix_cache"]["stats"]["hits"] == (1, 2)
    assert snapshot.resident["runner"]["prefix_cache"]["phase_calls"]["prefill"] == 3


def test_snapshot_coerces_non_mapping_optional_inputs_to_empty() -> None:
    snapshot = metrics._MetricsSnapshot(
        resident=[1, 2],  # type: ignore[arg-type]
        pool=5,  # type: ignore[arg-type]
        graph="bad",  # type: ignore[arg-type]
        queue=None,  # type: ignore[arg-type]
        scheduler_fairness=None,  # type: ignore[arg-type]
    )
    assert dict(snapshot.resident) == {}
    assert dict(snapshot.pool) == {}
    assert dict(snapshot.graph) == {}
    assert dict(snapshot.queue) == {}
    assert dict(snapshot.scheduler_fairness) == {}


def test_absent_and_malformed_optional_data_keep_defaults() -> None:
    assert metrics._kv_pool_metric_values(None) == dict(metrics._KV_POOL_METRIC_DEFAULTS)
    assert metrics._kv_pool_metric_values({"current_bytes": True, "grow_events": -1})[
        "current_bytes"
    ] == 0.0
    assert metrics._generation_queue_metric_values(None) == {
        "depth": 0.0,
        "max_depth": 0.0,
        "worker_active": 0.0,
        "active_requests": 0.0,
        "max_active_requests": 0.0,
    }
    empty_graph = metrics._graph_bucket_metric_values(
        {}, None, kernel_time_buckets=("le_10us",)
    )
    assert empty_graph["entries"] == 0.0
    assert empty_graph["kernel_time_histogram_ns"] == {}
    exposed_empty_graph = metrics._graph_bucket_metric_values(
        {}, {}, kernel_time_buckets=("le_10us",)
    )
    assert exposed_empty_graph["kernel_time_histogram_ns"] == {"le_10us": 0.0}
    assert metrics._scheduler_metric_values(None) == {
        "policy": "unavailable",
        "compatible_sampling_coalescing": False,
        "continuous_decode": False,
        "preemptive_fairness": False,
    }


@pytest.mark.parametrize(
    "case", CORPUS["normalization"]["resident"], ids=lambda case: case["name"]
)
def test_resident_normalization_matches_pre_move_corpus(case) -> None:
    assert metrics._resident_loop_metric_values(_decode(case["input"])) == _decode(
        case["expected"]
    )


@pytest.mark.parametrize(
    "case", CORPUS["normalization"]["prefix_cache"], ids=lambda case: case["name"]
)
def test_prefix_cache_normalization_matches_pre_move_corpus(case) -> None:
    assert metrics._prefix_cache_metric_values(_decode(case["input"])) == _decode(
        case["expected"]
    )


@pytest.mark.parametrize("case", CORPUS["normalization"]["pool"], ids=lambda case: case["name"])
def test_pool_normalization_matches_pre_move_corpus(case) -> None:
    assert metrics._kv_pool_metric_values(_decode(case["input"])) == _decode(
        case["expected"]
    )


@pytest.mark.parametrize("case", CORPUS["normalization"]["queue"], ids=lambda case: case["name"])
def test_queue_normalization_matches_pre_move_corpus(case) -> None:
    assert metrics._generation_queue_metric_values(_decode(case["input"])) == _decode(
        case["expected"]
    )


@pytest.mark.parametrize("case", CORPUS["normalization"]["graph"], ids=lambda case: case["name"])
def test_graph_normalization_matches_pre_move_corpus(case) -> None:
    raw = _decode(case["input"])
    assert metrics._graph_bucket_metric_values(
        raw["live_graph"],
        raw["graph_stats"],
        kernel_time_buckets=CORPUS["graph_kernel_time_buckets"],
    ) == _decode(case["expected"])


@pytest.mark.parametrize(
    "case", CORPUS["normalization"]["string_counts"], ids=lambda case: case["name"]
)
def test_string_counts_match_pre_move_corpus(case) -> None:
    assert metrics._string_value_counts(_decode(case["input"])) == _decode(case["expected"])


@pytest.mark.parametrize(
    "case", CORPUS["normalization"]["non_negative"], ids=lambda case: case["name"]
)
def test_non_negative_values_match_pre_move_corpus(case) -> None:
    assert metrics._non_negative_metric_value(_decode(case["input"])) == _decode(
        case["expected"]
    )


@pytest.mark.parametrize("case", CORPUS["normalization"]["format"], ids=lambda case: case["name"])
def test_metric_value_formatting_matches_pre_move_corpus(case) -> None:
    assert metrics._format_metric_value(_decode(case["input"])) == _decode(case["expected"])


@pytest.mark.parametrize("case", CORPUS["normalization"]["escape"], ids=lambda case: case["name"])
def test_label_escaping_matches_pre_move_corpus(case) -> None:
    assert metrics._escape_prometheus_label_value(_decode(case["input"])) == _decode(
        case["expected"]
    )


@pytest.mark.parametrize("case", CORPUS["counters"], ids=lambda case: case["name"])
def test_counter_accumulation_matches_pre_move_corpus(case) -> None:
    counters = _counters(case)
    for name, expected in case["expected_fields"].items():
        assert getattr(counters, name) == expected, name


@pytest.mark.parametrize("case", CORPUS["render_cases"], ids=lambda case: case["name"])
def test_render_matches_pre_move_corpus(case) -> None:
    text = metrics._render_prometheus_metrics(_counters(case["counters"]), snapshot=_snapshot(case))
    assert text == case["expected"]


def test_unchanged_definition_asts_match_the_pre_move_control() -> None:
    # ``_prefix_cache_metric_values`` changed only its final return, from
    # ``deepcopy`` to a thaw of the recursively frozen snapshot. Its output,
    # including nested collections, is checked by the prefix-cache corpus and
    # the thaw test below; every other moved definition is byte-identical.
    changed_definitions = {"_prefix_cache_metric_values"}
    for name, expected in CORPUS["ast_hashes"].items():
        if name in changed_definitions:
            continue
        source = inspect.getsource(getattr(metrics, name))
        actual = hashlib.sha256(
            ast.dump(ast.parse(source).body[0], include_attributes=False).encode()
        ).hexdigest()
        assert actual == expected, name


def test_multiple_counter_owners_do_not_share_state() -> None:
    first = metrics._ServerMetrics()
    second = metrics._ServerMetrics()
    first.record_success(
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 5,
                "rejected_prediction_tokens": 1,
            },
        }
    )
    second.record_rejected()
    second.record_cancelled()
    assert first.request_total == 1
    assert first.mtp_requests_total == 1
    assert second.request_total == 2
    assert second.request_failed_total == 2
    assert second.mtp_requests_total == 0
    first_text = metrics._render_prometheus_metrics(first, snapshot=metrics._MetricsSnapshot())
    second_text = metrics._render_prometheus_metrics(second, snapshot=metrics._MetricsSnapshot())
    assert _metric_value(first_text, "hipengine_requests_total") == 1
    assert _metric_value(first_text, "hipengine_mtp_serving") == 0
    assert _metric_value(first_text, "hipengine_mtp_draft_acceptance_rate") == pytest.approx(5 / 6)
    assert _metric_value(second_text, "hipengine_requests_total") == 2
    assert _metric_value(second_text, "hipengine_request_rejected_total") == 1
    assert _metric_value(second_text, "hipengine_request_cancelled_total") == 1
    assert _metric_value(second_text, "hipengine_mtp_requests_total") == 0


def test_mtp_off_on_queue_pressure_and_cancel_are_explicit_snapshot_inputs() -> None:
    absent = CORPUS["render_cases"][0]
    full = CORPUS["render_cases"][1]
    malformed = CORPUS["render_cases"][2]
    absent_text = metrics._render_prometheus_metrics(
        _counters(absent["counters"]), snapshot=_snapshot(absent)
    )
    full_text = metrics._render_prometheus_metrics(
        _counters(full["counters"]), snapshot=_snapshot(full)
    )
    malformed_text = metrics._render_prometheus_metrics(
        _counters(malformed["counters"]), snapshot=_snapshot(malformed)
    )
    assert _metric_value(absent_text, "hipengine_mtp_serving") == 0
    assert _metric_value(full_text, "hipengine_mtp_serving") == 1
    assert _metric_value(full_text, "hipengine_generation_queue_depth") == 2
    assert _metric_value(full_text, "hipengine_generation_queue_max_depth") == 3
    assert _metric_value(full_text, "hipengine_generation_requests_active") == 2
    assert _metric_value(full_text, "hipengine_chat_sessions_pending") == 1
    assert _metric_value(malformed_text, "hipengine_generation_queue_depth") == 3
    assert _metric_value(malformed_text, "hipengine_generation_queue_max_depth") == 0
    assert _metric_value(malformed_text, "hipengine_request_rejected_total") == 1
    assert _metric_value(malformed_text, "hipengine_request_cancelled_total") == 1
    assert _metric_value(malformed_text, "hipengine_kv_pool_current_bytes") == 4096
    assert _metric_value(malformed_text, "hipengine_graph_bucket_entries") == 6
    assert _metric_value(malformed_text, "hipengine_graph_bucket_hits_total") == 7
    assert _metric_value(malformed_text, "hipengine_graph_bucket_misses_total") == 8
