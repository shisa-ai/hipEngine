"""Retirement contract for the Qwen4Exp legacy full-attention context switch.

``HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY`` used to send the QSA prefill token
mixer's dense dispatch back to the pre-promotion fixed256 context leaf. The
promoted leaf owns every chunk with two or more dense rows and a single dense row
keeps the adaptive batch leaf, so the retired key has no reader left.

The dispatcher itself needs a live device, so these tests compile its real body
out of the runtime module and run it against fake leaves. The routing, the
positional ABI, and the KV-capacity argument are the production statements under
test rather than a transcription of them. The shared fixed256 leaves are checked
separately, because the supported prefill microbench imports them directly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import textwrap

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import resolve_paged_attn_decode
from hipengine.kernels.hip_gfx1100 import attention as gfx1100_attention
from hipengine.kernels.hip_gfx1100.attention import (
    paged_attn_decode as leaf_module,
    register_qwen35_paged_attn_decode_kernels,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import clear_registry_for_tests
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime import qwen4_exp_runner as module

MIXER = 'run_qwen4_exp_qsa_prefill_token_mixer'
LEGACY_KEY = 'HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY'
LEGACY_HELPER = '_qwen4_exp_fullattn_context_legacy_enabled'
PROMOTED = 'qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans'
SHIPPED = 'qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans'
ADAPTIVE = 'qwen35_paged_full_attn_decode_context_bf16_batch_spans'

# Geometry the mixer passes through unchanged; only the leaf choice and the
# capacity argument are under test.
QUERY_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
BLOCK_SIZE = 256
STREAM = 7
QUERY_PTR = 0x3000
KEY_PTR = 0x1000
VALUE_PTR = 0x2000
CONTEXT_PTR = 0x4000
RUNTIME = object()

stale_key = pytest.mark.parametrize(
    'stale',
    [None, '0', '1', 'false'],
    ids=['unset', 'zero', 'one', 'false'],
)


@dataclass(frozen=True)
class _Launch:
    name: str
    args: tuple
    kwargs: dict


class _Buffer:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = ptr
        self.nbytes = nbytes


def _dense_dispatch_source() -> str:
    """The mixer's real ``elif dense_rows:`` body, dedented for compilation."""

    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    mixers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == MIXER
    ]
    assert len(mixers) == 1, f'expected one {MIXER} definition'
    branches = [
        node for node in ast.walk(mixers[0])
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == 'dense_rows'
        and 'batch_fixed256' in (ast.get_source_segment(source, node) or '')
    ]
    assert len(branches) == 1, 'expected one dense fixed256 dispatch branch'
    branch = branches[0]
    lines = source.splitlines()
    body = lines[branch.body[0].lineno - 1:branch.body[-1].end_lineno]
    return textwrap.dedent('\n'.join(body))


def _compile_dispatch(**namespace):
    program = f'def _dense_dispatch():\n{textwrap.indent(_dense_dispatch_source(), "    ")}\n'
    scope = dict(vars(module))
    scope.update(namespace)
    exec(compile(program, '<qwen4-exp dense dispatch>', 'exec'), scope)
    return scope['_dense_dispatch']


def _run_dispatch(rows: int, *, capacity_tokens: int = 2051):
    """Run the real dense branch with fake leaves and a fake KV slab."""

    calls: list[_Launch] = []
    spans: list[tuple[int, int, bool]] = []

    def leaf(name):
        def launch(*args, **kwargs):
            calls.append(_Launch(name=name, args=args, kwargs=kwargs))
            return object()
        return launch

    def record_spans(*, start_row, rows, decode):
        spans.append((start_row, rows, decode))
        return ('spans', start_row, rows, decode)

    kv_bytes = capacity_tokens * KV_HEADS * HEAD_DIM * 2
    dispatch = _compile_dispatch(
        **{
            PROMOTED: leaf(PROMOTED),
            SHIPPED: leaf(SHIPPED),
            ADAPTIVE: leaf(ADAPTIVE),
            'attention_state': SimpleNamespace(
                key_cache=_Buffer(KEY_PTR, kv_bytes),
                value_cache=_Buffer(VALUE_PTR, 1024),
                block_size=BLOCK_SIZE,
            ),
            'scratch': SimpleNamespace(
                query=_Buffer(QUERY_PTR, 1024),
                context=_Buffer(CONTEXT_PTR, 1024),
            ),
            'metadata': SimpleNamespace(spans=record_spans),
            'dense_rows': rows,
            'start': 0,
            'query_heads': QUERY_HEADS,
            'kv_heads': KV_HEADS,
            'head_dim': HEAD_DIM,
            'stream': STREAM,
            'active_runtime': RUNTIME,
        }
    )
    dispatch()
    return calls, spans


def _assert_launch(launch: _Launch, *, name: str, rows: int, capacity) -> None:
    assert launch.name == name
    args = launch.args
    assert args[:4] == (QUERY_PTR, KEY_PTR, VALUE_PTR, CONTEXT_PTR)
    assert args[4] == ('spans', 0, rows, True)
    assert args[5] == rows
    assert args[6] == rows, 'the context bound is the chunk start plus its rows'
    assert args[7] == BLOCK_SIZE
    assert args[8] == QUERY_HEADS
    assert args[9] == KV_HEADS
    assert args[10] == HEAD_DIM
    assert args[11] == HEAD_DIM ** -0.5
    if capacity is None:
        assert len(args) == 12, 'the capacity argument must be omitted entirely'
    else:
        assert len(args) == 13
        assert args[12] == capacity
    assert launch.kwargs == {'stream': STREAM, 'runtime': RUNTIME}
    assert launch.kwargs['runtime'] is RUNTIME


@stale_key
@pytest.mark.parametrize('rows', [1, 2, 7])
def test_dense_dispatch_routes_by_rows_for_every_stale_key_value(
        monkeypatch, rows, stale):
    if stale is None:
        monkeypatch.delenv(LEGACY_KEY, raising=False)
    else:
        monkeypatch.setenv(LEGACY_KEY, stale)

    calls, spans = _run_dispatch(rows)

    assert spans == [(0, rows, True)]
    assert len(calls) == 1
    assert calls[0].name != SHIPPED, (
        f'the retired {LEGACY_KEY}={stale!r} path still selects the shipped leaf'
    )
    if rows >= 2:
        # Promoted fixed256, with the KV slab's per-token capacity forwarded.
        _assert_launch(calls[0], name=PROMOTED, rows=rows, capacity=2051)
    else:
        # One dense row keeps the adaptive batch leaf and takes no capacity.
        _assert_launch(calls[0], name=ADAPTIVE, rows=rows, capacity=None)


@pytest.mark.parametrize('rows,capacity_tokens,capacity', [
    (2, 700, 700),
    (4, 0, None),
    (7, 1, 1),
])
def test_promoted_dispatch_forwards_or_omits_the_kv_capacity(
        monkeypatch, rows, capacity_tokens, capacity):
    monkeypatch.setenv(LEGACY_KEY, '1')

    calls, _ = _run_dispatch(rows, capacity_tokens=capacity_tokens)

    assert len(calls) == 1
    _assert_launch(calls[0], name=PROMOTED, rows=rows, capacity=capacity)


def test_runtime_module_has_no_reader_for_the_retired_key():
    source = Path(module.__file__).read_text()

    assert LEGACY_KEY not in source
    names = {
        node.name for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }
    assert LEGACY_HELPER not in names


def test_shared_fixed256_leaves_stay_exported_and_registered():
    for name in (SHIPPED, PROMOTED):
        assert name in gfx1100_attention.__all__
        leaf = getattr(gfx1100_attention, name)
        assert leaf is getattr(leaf_module, name)
        # Qwen4Exp retains only the promoted import; the shared shipped leaf
        # remains exported for registry and direct diagnostic consumers.
        if name == PROMOTED:
            assert getattr(module, name) is leaf
        else:
            assert not hasattr(module, name)

    clear_registry_for_tests()
    register_qwen35_paged_attn_decode_kernels()
    register_gfx1151_kernels(replace=True)

    policy = FixedPagedKVPolicy(storage_dtype='bf16')
    policy.register(
        7,
        block_table=Tensor.from_handle(0x1000, (1,), 'int32', Device('hip', 0)),
        live_counts=Tensor.from_handle(0x2000, (1,), 'int64', Device('hip', 0)),
        max_live_count=2,
        capacity_tokens=256,
    )
    resolved = resolve_paged_attn_decode(
        backend='hip_gfx1151',
        spans=policy.batch_spans([7]),
        kind='context_batch',
        model_quant='w4_paro',
    )

    assert resolved.__name__ == SHIPPED
    assert resolved is getattr(leaf_module, SHIPPED)
