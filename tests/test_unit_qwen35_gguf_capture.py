"""Host diagnostic readers stay independent of execution and session state."""
from __future__ import annotations

import ast
import inspect
import pickle
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hipengine.runtime import qwen35_gguf_capture as capture
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION


def _inputs():
    cfg = NS(is_moe=True, expert_used_count=3, expert_count=7,
             expert_feed_forward_length=8, expert_shared_feed_forward_length=9,
             ssm_inner_size=6, ssm_time_step_rank=2)
    fields = ('norm', 'attn_out', 'post_norm', 'residual', 'moe_down_out', 'ffn_down',
              'linear_qkv', 'linear_z', 'linear_alpha', 'linear_beta', 'conv_out',
              'recurrent_out', 'recurrent_bf16', 'moe_router_logits', 'ffn_intermediate',
              'moe_shared_intermediate', 'moe_shared_out', 'moe_routing_weights',
              'moe_selected_experts')
    scratch = NS(**{name: NS(ptr=1000 + i * 100) for i, name in enumerate(fields)})
    calls = []
    runtime = object()

    def reader(kind):
        def copy(ptr, count, *, runtime):
            calls.append((kind, ptr, count, runtime))
            return np.full(count, ptr, dtype=np.int64 if kind == 'i64' else np.float32)
        return copy

    readers = capture.CaptureReaders(*(reader(kind) for kind in ('bf16', 'f32', 'i64')))
    return cfg, scratch, runtime, readers, calls


def test_boundary_copy_order_and_sizes():
    cfg, scratch, runtime, readers, calls = _inputs()
    result = capture.read_linear_attention_boundary(
        cfg=cfg, hidden_size=4, linear_qkv_width=10, scratch=scratch,
        layer_id=0, token_id=7, position=3, runtime=runtime, readers=readers,
    )
    expected = [('bf16', 'norm', 4), ('bf16', 'linear_qkv', 10), ('bf16', 'linear_z', 6),
                ('bf16', 'linear_alpha', 2), ('bf16', 'linear_beta', 2),
                ('f32', 'conv_out', 10), ('f32', 'recurrent_out', 6),
                ('bf16', 'recurrent_bf16', 6), ('bf16', 'attn_out', 4)]
    assert calls == [(kind, getattr(scratch, name).ptr, count, runtime) for kind, name, count in expected]
    assert result.as_summary_dict()['finite'] is True
    result.conv_out_f32[0] = np.nan
    assert result.as_summary_dict()['finite'] is False


@pytest.mark.parametrize('layer_type', [LINEAR_ATTENTION, FULL_ATTENTION])
@pytest.mark.parametrize('is_moe', [False, True])
def test_layer_reader_uses_only_selected_scratch(layer_type, is_moe):
    cfg, scratch, runtime, readers, calls = _inputs()
    cfg.is_moe = is_moe
    expected = []
    if layer_type == LINEAR_ATTENTION:
        expected += [('bf16', 'linear_qkv', 10), ('bf16', 'linear_z', 6),
                     ('bf16', 'linear_alpha', 2), ('bf16', 'linear_beta', 2),
                     ('f32', 'conv_out', 10), ('f32', 'recurrent_out', 6),
                     ('bf16', 'recurrent_bf16', 6)]
    else:
        for name in ('linear_qkv', 'linear_z', 'linear_alpha', 'linear_beta', 'conv_out', 'recurrent_out', 'recurrent_bf16'):
            delattr(scratch, name)
    if is_moe:
        expected += [('f32', 'moe_router_logits', 7), ('bf16', 'ffn_intermediate', 24),
                     ('bf16', 'moe_shared_intermediate', 9), ('bf16', 'moe_shared_out', 4),
                     ('f32', 'moe_routing_weights', 3)]
        del scratch.ffn_down
    else:
        for name in list(vars(scratch)):
            if name.startswith('moe_') or name == 'ffn_intermediate':
                delattr(scratch, name)
    expected_calls = [(kind, getattr(scratch, name).ptr, count, runtime) for kind, name, count in expected]
    if is_moe:
        expected_calls += [('f32', scratch.moe_router_logits.ptr + 28, 1, runtime),
                           ('i64', scratch.moe_selected_experts.ptr, 3, runtime)]
    expected_calls += [('bf16', 91, 4, runtime)]
    expected_calls += [('bf16', getattr(scratch, name).ptr, 4, runtime)
                       for name in ('norm', 'attn_out', 'post_norm', 'residual')]
    expected_calls += [('bf16', scratch.moe_down_out.ptr if is_moe else scratch.ffn_down.ptr,
                        12 if is_moe else 4, runtime), ('bf16', 92, 4, runtime)]
    result = capture.read_attention_layer(
        cfg=cfg, hidden_size=4, linear_qkv_width=10, scratch=scratch,
        layer_id=3, layer_type=layer_type, token_id=7, position=11,
        run_preceding_layers=True, target_src_ptr=91, target_dst_ptr=92,
        runtime=runtime, readers=readers,
    )
    assert calls == expected_calls
    summary = result.as_summary_dict()
    assert summary['preceding_layer_count'] == 3
    assert summary['post_norm_source'] == 'bf16_scratch.post_norm'
    assert summary['finite'] is True
    assert result.moe_selected_experts_i64 is None if not is_moe else result.moe_selected_experts_i64.shape == (3,)
    assert result.linear_qkv_f32 is None if layer_type == FULL_ATTENTION else result.linear_qkv_f32.shape == (10,)


def test_router_reader_preserves_pre_execution_hidden_and_copy_order():
    _, scratch, runtime, readers, calls = _inputs()
    hidden = np.arange(8, dtype=np.float64)[::2]
    result = capture.read_router_trace_layer(
        layer_id=2, layer_type=FULL_ATTENTION, token_id=9, position=4,
        hidden_size=4, expert_count=7, top_k=3, hidden_in=hidden,
        target_dst_ptr=99, scratch=scratch, runtime=runtime, readers=readers,
    )
    assert calls == [('bf16', 99, 4, runtime), ('f32', scratch.moe_router_logits.ptr, 7, runtime),
                     ('f32', scratch.moe_routing_weights.ptr, 3, runtime),
                     ('f32', scratch.moe_router_logits.ptr + 28, 1, runtime),
                     ('i64', scratch.moe_selected_experts.ptr, 3, runtime)]
    np.testing.assert_array_equal(result.hidden_in_f32, hidden)
    assert result.hidden_in_f32.flags.c_contiguous
    assert result.hidden_in_f32.dtype == np.float32
    assert result.as_summary_dict()['finite'] is True


def test_capture_reader_failure_propagates_without_ownership_operations():
    cfg, scratch, runtime, readers, _ = _inputs()
    def fail(*args, **kwargs):
        raise RuntimeError('copy failed')
    readers = capture.CaptureReaders(fail, readers.f32, readers.i64)
    with pytest.raises(RuntimeError, match='copy failed'):
        capture.read_linear_attention_boundary(
            cfg=cfg, hidden_size=4, linear_qkv_width=10, scratch=scratch,
            layer_id=0, token_id=7, position=3, runtime=runtime, readers=readers,
        )


def test_compatibility_types_are_aliases_and_pickle_round_trips():
    from hipengine.runtime import qwen35_gguf_runner as runner
    from hipengine.runtime.qwen35_gguf_results import Qwen35GGUFNextTokenProbeResult
    for name in ('Qwen35GGUFLinearAttentionBoundaryCapture', 'Qwen35GGUFLinearAttentionLayerCapture', 'Qwen35GGUFRouterTraceLayerCapture'):
        assert getattr(runner, name) is getattr(capture, name)
    assert runner.Qwen35GGUFNextTokenProbeResult is Qwen35GGUFNextTokenProbeResult
    sample = Qwen35GGUFNextTokenProbeResult(3, 1.0, np.ones(4, dtype=np.float32))
    restored = pickle.loads(pickle.dumps(sample))
    assert type(restored) is type(sample)
    np.testing.assert_array_equal(restored.logits, sample.logits)


def test_capture_owner_has_no_execution_or_session_dependency():
    tree = ast.parse(inspect.getsource(capture))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(part in (node.module or '') for part in ('runner', 'generation', 'kernels', 'torch'))
    for function in (capture.read_attention_layer, capture.read_linear_attention_boundary, capture.read_router_trace_layer):
        assert not {'runner', 'session', 'self'} & set(inspect.signature(function).parameters)


def test_source_audit_requires_matching_capture_source_for_copied_runner(tmp_path):
    from scripts.gguf_capture_path_audit import read_capture_source

    path = tmp_path / 'runner.py'
    delegated = 'return _diagnostic_capture.read_attention_layer('
    assert read_capture_source(path, 'historical inline runner') == ''
    with pytest.raises(ValueError, match='place the matching revision'):
        read_capture_source(path, delegated)
    owner = path.with_name('qwen35_gguf_capture.py')
    owner.write_text('matching source')
    assert read_capture_source(path, delegated) == 'matching source'


def test_source_audits_follow_reader_binding_and_reject_broken_binding():
    from hipengine.runtime import qwen35_gguf_runner as runner
    from scripts.gguf_capture_path_audit import audit_runner_source
    from scripts.gguf_hidden_precision_decision_audit import audit_runner_hidden_precision
    runner_text, capture_text = inspect.getsource(runner), inspect.getsource(capture)
    assert audit_runner_source(runner_text, capture_text=capture_text)['facts']['capture_hidden_tap_copies_target_src_ptr']
    assert audit_runner_hidden_precision(runner_text, capture_text=capture_text)['facts']['capture_hidden_in_uses_bf16_copy']
    for broken_runner, broken_capture in (
        (runner_text, ''),
        (runner_text.replace('bf16=_copy_bf16_ptr_to_host_f32', 'bf16=_copy_f32_ptr_to_host'), capture_text),
        (runner_text, capture_text.replace('hidden_in_f32=readers.bf16(', 'hidden_in_f32=readers.f32(')),
    ):
        assert not audit_runner_source(broken_runner, capture_text=broken_capture)['facts']['capture_hidden_tap_copies_target_src_ptr']
