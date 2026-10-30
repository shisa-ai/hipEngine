"""Exercise the verifier's GDN-to-projection boundary without launching HIP."""

import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.loading.qwen35_gguf_consumer_surface import resolve_gdn_output_handoff
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16, LAYOUT_GGUF_Q5_K_T16, LAYOUT_GGUF_Q8_0_T16,
)
from hipengine.runtime import qwen35_gguf_runner as runner


def _projection_boundary():
    # Execute only the contiguous input-selection/cast statements from the real
    # method. Earlier GDN arithmetic needs HIP and is not mocked into an oracle.
    tree = ast.parse(textwrap.dedent(inspect.getsource(
        runner.Qwen35GGUFFullStackRunner._run_linear_attention_prefill_layer_rows
    )))
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index, node in enumerate(body):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "ssm_out_input_ptr"
                for target in node.targets
            ):
                selected = body[index:index + 3]
                assert isinstance(selected[1], ast.Assign)
                assert isinstance(selected[2], ast.If)
                return compile(ast.fix_missing_locations(ast.Module(
                    body=copy.deepcopy(selected), type_ignores=[]
                )), "<verifier-gdn-projection-boundary>", "exec")
    raise AssertionError("verifier projection input boundary is missing")


@pytest.mark.parametrize("rows", [1, 2, 5])
@pytest.mark.parametrize("layout", [LAYOUT_DENSE_BF16, LAYOUT_GGUF_Q5_K_T16, LAYOUT_GGUF_Q8_0_T16])
@pytest.mark.parametrize("already_bf16", ["none", "prefill-gdn", "score"])
@pytest.mark.parametrize("force_bf16", [False, True])
def test_verifier_gdn_projection_respects_input_capability(rows, layout, already_bf16, force_bf16):
    calls = []
    weight = SimpleNamespace(spec=SimpleNamespace(layout=layout))
    namespace = {
        **vars(runner),
        "scratch": SimpleNamespace(recurrent_out=SimpleNamespace(ptr=100), recurrent_bf16=SimpleNamespace(ptr=200)),
        "layer": SimpleNamespace(weight=lambda name: weight if name == "ssm_out" else pytest.fail(name)),
        "use_prefill_gdn_capture": already_bf16 == "prefill-gdn",
        "prefill_score_ready": already_bf16 == "score",
        "_gguf_verify_capture_bf16_gdn_out_enabled": lambda: force_bf16,
        "f32_to_bf16": lambda *args, **kwargs: calls.append((args, kwargs)),
        "rows": rows, "cfg": SimpleNamespace(ssm_inner_size=16),
        "stream": 7, "cast_library": "cast-library", "runtime": "runtime",
        "stage_timings": None, "sync_stages": False, "stage_prefix": "test",
        "t_stage": 0, "gpu_stage_recorder": None,
        "_mark_sync_stage": lambda *args: 0,
    }
    exec(_projection_boundary(), namespace)
    needs_cast = bool(resolve_gdn_output_handoff(layout).adapters)
    expected_bf16 = already_bf16 != "none" or force_bf16 or needs_cast
    assert namespace["ssm_out_activation_dtype"] == ("bf16" if expected_bf16 else "f32")
    assert namespace["ssm_out_input_ptr"] == (200 if expected_bf16 else 100)
    # A BF16 producer must not be replaced with a cast of stale recurrent_out.
    should_cast = already_bf16 == "none" and (force_bf16 or needs_cast)
    assert calls == ([((100, 200, rows * 16), {"stream": 7, "library": "cast-library", "runtime": "runtime"})] if should_cast else [])
