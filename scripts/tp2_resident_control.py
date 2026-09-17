"""Optimized resident TP1 adapter for the existing TP2 coverage machinery.

Diagnostics only: capture all causal prefill hidden rows, then use the SAME c1
output norm/head as normal resident generation for every position. This is not
verifier-shaped head arithmetic, a token-eager TP1 denominator, or a benchmark.
"""
from __future__ import annotations
from types import SimpleNamespace
import os

import numpy as np

from scripts.tp2_xtx_tp1_eager_stage_probe import validate_result


def bind_resident_profile(profile: str) -> dict:
    from hipengine.generation import register_builtin_generators
    from hipengine.execution_profiles import resolve_runtime_profile, validate_variant_manifest
    register_builtin_generators()
    resolved = resolve_runtime_profile(model='qwen3_5_gguf', backend='hip_gfx1100',
                                       quant='gguf_q4_k_m', profile=profile)
    resolved.construct_generator(SimpleNamespace)
    return {'requested': profile, 'manifest': validate_variant_manifest(resolved.manifest),
            'manifest_sha256': resolved.manifest_sha256,
            'strict_manifest_sha256': resolved.strict_manifest_sha256,
            'fell_back_to_strict': resolved.fell_back_to_strict,
            'env_after_binding': {k: v for k, v in os.environ.items() if k.startswith('HIPENGINE_')},
            'scope_note': 'Named TP1 arithmetic binder; not a distributed/AR-graph promotion certificate.'}


class ResidentTP1Control:
    """Factory adapter: bulk causal hidden capture with product c1 norm/head."""
    def __init__(self, session, hidden_buffer, *, capacity, row_hook=None, owns_buffer=False):
        self.session = session
        self.hidden_buffer = hidden_buffer
        self.max_sequence_length = int(capacity)
        self.runtime = session.runtime
        self.vocab_size = int(session.runner.vocab_size)
        self.devices = (0,)
        self.mode, self.schedule = 'resident_tp1', 'bulk-prefill/c1-head'
        self.driver, self.reduce_mode, self.head_shard = 'resident', 'none', False
        self.prefill_schedule = 'bulk'
        self.row_hook = row_hook or (lambda record: None)
        self.owns_buffer = owns_buffer

    def teacher_forced_logits(self, token_ids):
        tokens = tuple(int(t) for t in token_ids)
        if not tokens or len(tokens) > self.max_sequence_length:
            raise ValueError('teacher trajectory exceeds declared capacity or is empty')
        row_bytes = int(self.session.runner.hidden_size) * 2
        if len(tokens) * row_bytes > self.hidden_buffer.nbytes:
            raise ValueError('hidden capture buffer too small')
        self.session.reset()
        final = self.session.prefill(tokens, use_bulk=None, bulk_attention_mode='bulk',
            return_logits=True, capture_target_hidden_rows=self.hidden_buffer)
        self.runtime.device_synchronize()
        validate_result(final, self.vocab_size)
        expected_final = np.asarray(final.logits).reshape(-1).copy()
        if int(self.session.position) != len(tokens):
            raise ValueError('prefill did not consume every teacher input position')
        rows = []
        for position, token in enumerate(tokens):
            self.row_hook({'phase': 'head-start', 'input_token': token, 'position': position,
                           'session_position': int(self.session.position), 'slot': 0})
            norm = self.session._run_output_norm_hidden(
                self.hidden_buffer.ptr + position * row_bytes, self.session.scratch.norm.ptr,
                stream=0, capture_hidden_seed_fp32=False)
            sample = self.session._sample_from_hidden(norm, return_logits=True, stream=0)
            self.runtime.device_synchronize()
            validate_result(sample, self.vocab_size)
            if int(self.session.position) != len(tokens):
                raise ValueError('head readback mutated recurrent/KV position')
            rows.append(np.asarray(sample.logits).reshape(-1).copy())
            self.row_hook({'phase': 'head-complete', 'input_token': token, 'position': position,
                           'session_position': int(self.session.position), 'slot': 0,
                           'sampled_token': int(sample.token_id)})
        if not np.array_equal(rows[-1], expected_final):
            raise ValueError('captured final row differs from normal product prefill logits')
        return np.stack(rows)

    def generate(self, tokens, *, max_new_tokens):
        self.session.reset()
        first = self.session.prefill(tokens, use_bulk=None, return_logits=True)
        validate_result(first, self.vocab_size)
        graph = self.session.capture_decode_graph(position=int(self.session.position),
            steps_per_replay=1, max_replay_steps=int(max_new_tokens),
            attention_max_context_len=int(self.session.position) + int(max_new_tokens))
        output = []
        start = int(self.session.position)
        transitions = []
        for step in range(int(max_new_tokens)):
            if int(self.session.position) != start + step:
                raise ValueError('generation graph input position mismatch')
            graph.replay(1)
            sample = graph.read_sample(return_logits=True)
            validate_result(sample, self.vocab_size)
            if int(self.session.position) != start + step + 1:
                raise ValueError('generation graph output position mismatch')
            output.append(int(sample.token_id))
            transitions.append({'input_position': start + step, 'output_position': int(self.session.position),
                                'sampled_token': int(sample.token_id)})
        self.generation_control = {'start_position': start, 'transitions': transitions,
                                   'graph': graph.transport_provenance()}
        graph.close()  # No finally: a failed stage must not launch more GPU work.
        return SimpleNamespace(token_ids=output)

    def close(self):
        if self.owns_buffer:
            from hipengine.core.memory import free
            free(self.hidden_buffer, runtime=self.runtime)
            self.owns_buffer = False
        self.session.close()


def create_resident_control(model, *, capacity, row_hook=None, capture_rows=True):
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.core.memory import malloc
    os.environ['HIPENGINE_GGUF_DECODE_REPACK'] = '1'
    session = Qwen35GGUFResidentSession(model, max_sequence_length=capacity,
        use_wmma_prefill=True, use_gemv_decode=True,
        prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
        kv_policy=None, kv_scale_dtype='fp16', kv_scale_granularity=None)
    capture = malloc(capacity * session.runner.hidden_size * 2, runtime=session.runtime) if capture_rows else None
    return ResidentTP1Control(session, capture, capacity=capacity, row_hook=row_hook, owns_buffer=capture_rows)


def read_position(runtime, scratch):
    values = []
    for name in ('position_buf', 'context_buf'):
        host = np.empty(1, dtype=np.int64)
        runtime.memcpy(host.ctypes.data, getattr(scratch, name).ptr, host.nbytes, 2)
        values.append(int(host[0]))
    return values


def create_coverage_session(model, *, devices, mode, capacity, row_hook):
    """Fresh-process factory; TP1 visibility must map its physical GPU to zero."""
    hook = row_hook or (lambda record: None)
    if mode == 'tp1':
        session = create_resident_control(model, capacity=capacity, row_hook=hook)
        original = session.teacher_forced_logits
        def checked(tokens):
            output = original(tokens)
            if session.runtime.get_device() != 0:
                raise ValueError('resident control left its logical device')
            observed = read_position(session.runtime, session.session.scratch)
            if observed != [len(tokens), len(tokens) + 1]:
                raise ValueError(f'resident position/context ownership mismatch: {observed}')
            actual_tokens = np.empty(len(tokens), dtype=np.int64)
            session.runtime.memcpy(actual_tokens.ctypes.data, session.session._prefill_token_buf.ptr,
                                   actual_tokens.nbytes, 2)
            if not np.array_equal(actual_tokens, np.asarray(tokens, dtype=np.int64)):
                raise ValueError('resident device token rows differ from the declared prompt')
            hook({'phase': 'resident-state', 'position_context': observed, 'logical_device': 0,
                  'device_token_rows': actual_tokens.tolist()})
            return output
        session.teacher_forced_logits = checked
        return session
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
    from hipengine.core.device import scoped_current_device
    class CheckedTP2(MlpTP2GenerationSession):
        def _enqueue_embedding(self, token_id, position, stages):
            result = super()._enqueue_embedding(token_id, position, stages)
            for device in self.devices:
                with scoped_current_device(self.runtime, device):
                    self.runtime.stream_synchronize(self._rank_stream(device))
                    observed = read_position(self.runtime, self._scratches[device])
                    token = np.empty(1, dtype=np.int64)
                    self.runtime.memcpy(token.ctypes.data, self._step_buffers[device]['token_buf'].ptr, token.nbytes, 2)
                    if observed != [position, position + 1] or int(token[0]) != token_id:
                        raise ValueError(f'TP2 rank {device} input/position mismatch')
                    hook({'phase': 'tp2-input', 'logical_device': device, 'input_token': int(token[0]),
                          'position_context': observed, 'stream': self._rank_stream(device)})
            return result
        def _forward_token(self, token_id, position, *, kind):
            logits, trace = super()._forward_token(token_id, position, kind=kind)
            row = np.asarray(logits).reshape(-1)
            validate_result(SimpleNamespace(token_id=int(np.argmax(row)), logits=row), self.vocab_size)
            hook({'phase': 'tp2-output', 'position': position, 'input_token': token_id,
                  'sampled_token': int(np.argmax(row)), 'kind': kind})
            return logits, trace
    return CheckedTP2(model, devices=devices, mode='tp2', max_sequence_length=capacity)


def resolved_scope_manifest(session):
    """Actual owner/schedule/precision/MLP choices, separate from the TP1 profile plan."""
    import hashlib
    import json
    from pathlib import Path
    from hipengine.runtime.qwen35_gguf_runner import _gguf_dense_pair_silu_decode_variant
    resident = isinstance(session, ResidentTP1Control)
    runners = {0: session.session.runner} if resident else session._runners
    ranks = {}
    for device, runner in runners.items():
        ranks[str(device)] = {
            'backend': runner.backend, 'hidden_size': runner.hidden_size, 'vocab_size': runner.vocab_size,
            'ffn_size': runner.ffn_size, 'fp16_recurrent_state': runner.fp16_recurrent_state,
            'c1_full_mlp_variant': (_gguf_dense_pair_silu_decode_variant(runner, rows=1,
                in_features=runner.hidden_size, out_features=runner.ffn_size) if resident else None)}
    group = getattr(session, '_shard_group', None)
    manifest = {'kind': 'measured_tp_ar_scope', 'ranks': ranks,
                'capacity': session.max_sequence_length, 'schedule': session.schedule,
                'prefill_schedule': 'bulk' if resident else 'token-serial',
                'mode': session.mode, 'reduce_mode': session.reduce_mode,
                'head_shard': session.head_shard,
                'mlp_shard_variant': None if group is None else group.mlp_decode_variant,
                'per_rank_ffn': None if group is None else group.per_rank_ffn,
                'partial_dtypes': [] if group is None else sorted({r.partial_dtype for r in group._ranks.values()}),
                'head_plan': None if getattr(session, '_head_plan', None) is None else vars(session._head_plan),
                'layer_graph_count': len(getattr(session, '_layer_execs', {})),
                'source_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                    for name in ('hipengine/runtime/qwen35_gguf_runner.py', 'hipengine/runtime/gguf_linear.py',
                                 'hipengine/core/memory.py', 'hipengine/distributed/tp2_generate.py',
                                 'hipengine/distributed/shard_exec.py')},
                'profile_certificate': False}
    encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    return {'manifest': manifest, 'sha256': hashlib.sha256(encoded).hexdigest()}


class NativeARAdapter:
    """Common host loop over existing product primitives, with native readbacks.

    TP1 keeps graph feedback on-device. TP2 retains its full-logit readback,
    host argmax and native StepTrace construction. Extra forcing/control reads
    are explicitly untimed correctness-only operations.
    """
    def __init__(self, owner, *, resident):
        self.owner = owner
        self.resident = resident
        self.session = owner.session if resident else owner
        self.runtime = self.session.runtime
        self.vocab_size = owner.vocab_size
        # Report the schedule the session actually drives, not the schedule the
        # adapter used to hardcode: the TP2 arm's prefill is switchable, and a
        # comparison artifact that names the wrong schedule is not provenance.
        self.bulk_prefill = (
            True if resident else bool(getattr(self.session, 'bulk_prefill_enabled', False))
        )
        self.prefill_schedule = 'bulk' if resident else (
            'bulk-tp2' if self.bulk_prefill else 'token-serial'
        )
        self.position = 0
        self.graph = None
        self.traces = []
        # Resident TP1 only: ``None`` keeps the session's own default, ``False``
        # selects its token-serial prefill. The sustained probe uses this to
        # measure the schedule-sensitivity reference without a second session.
        self.prefill_use_bulk = None

    def prepare(self):
        if not self.resident:
            self.session._ensure_graph_schedule()

    def prefill(self, tokens):
        self.session.reset()
        self.traces = []
        if self.resident:
            result = self.session.prefill(tokens, use_bulk=self.prefill_use_bulk, bulk_attention_mode='bulk', return_logits=False)
            token = int(result.token_id)
        elif self.bulk_prefill:
            # The session's own rank-local bulk prefill, not an adapter-side
            # re-implementation: this arm exists to measure that candidate.
            rows = self.session.bulk_prefill(tokens)
            token = int(np.argmax(np.asarray(rows)[-1]))
        else:
            for position, input_token in enumerate(tokens):
                logits, trace = self.session._forward_token(int(input_token), position, kind='prefill')
                self.traces.append(trace)
                token = int(np.argmax(logits))
        self.position = len(tokens)
        self.next_token = token
        self.check_token(token)
        return token

    def check_token(self, token):
        if isinstance(token, bool) or not isinstance(token, (int, np.integer)) or not 0 <= token < self.vocab_size:
            raise ValueError(f'invalid sampled/forced token: {token}')

    def begin_decode(self, count):
        if self.resident:
            self.graph = self.session.capture_decode_graph(position=self.position,
                steps_per_replay=1, max_replay_steps=count,
                attention_max_context_len=self.position + count)

    def transition(self, token, *, return_logits=False, force=False):
        self.check_token(token)
        if self.resident:
            if force:
                from hipengine.core.memory import copy_host_array_to_device
                copy_host_array_to_device(self.session._lm_out_index, np.array([token], dtype=np.int64), runtime=self.runtime)
            elif token != self.next_token:
                raise ValueError('product graph feedback token mismatch')
            self.graph.replay(1)
            sample = self.graph.read_sample(return_logits=return_logits)
            if int(self.session.position) != self.position + 1:
                raise ValueError('graph transition position mismatch')
        else:
            logits, trace = self.session._forward_token(int(token), self.position, kind='decode')
            self.traces.append(trace)
            sample = SimpleNamespace(token_id=int(np.argmax(logits)), logits=logits)
        self.position += 1
        self.next_token = int(sample.token_id)
        self.check_token(sample.token_id)
        return sample

    def force_input(self, token):
        """Set/check graph input before replay; correctness only, outside timings."""
        self.check_token(token)
        if self.resident:
            from hipengine.core.memory import copy_host_array_to_device
            from scripts.tp2_xtx_tp1_eager_stage_probe import graph_device_state
            copy_host_array_to_device(self.session._lm_out_index, np.array([token], dtype=np.int64), runtime=self.runtime)
            state = graph_device_state(self.session)
            if state != {'position': self.position, 'context': self.position+1, 'sampled_token': int(token)}:
                raise ValueError(f'forced graph input control mismatch: {state}')
        self.next_token = int(token)

    def check_transition(self, token, position):
        """Read actual device ownership/positions, never used in product timing."""
        if self.resident:
            from scripts.tp2_xtx_tp1_eager_stage_probe import graph_device_state
            state = graph_device_state(self.session)
            if state['position'] != position+1 or state['context'] != position+2:
                raise ValueError('graph device cursor did not advance')
            return {'position': position, 'input_token': int(token), 'device': state}
        from hipengine.core.device import scoped_current_device
        states = {}
        for device in self.session.devices:
            with scoped_current_device(self.runtime, device):
                self.runtime.stream_synchronize(self.session._rank_stream(device))
                pair = read_position(self.runtime, self.session._scratches[device])
                host = np.empty(1, dtype=np.int64)
                self.runtime.memcpy(host.ctypes.data, self.session._step_buffers[device]['token_buf'].ptr, host.nbytes, 2)
                if pair != [position, position+1] or int(host[0]) != token:
                    raise ValueError(f'TP2 rank {device} input/cursor mismatch')
                states[str(device)] = {'position_context': pair, 'input_token': int(host[0])}
        return {'position': position, 'input_token': int(token), 'ranks': states}

    def end_decode(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None

    def destroy_graphs(self):
        self.end_decode()
        if not self.resident:
            self.session._destroy_graphs()

    def close(self):
        self.owner.close()


def create_native_adapter(model, arm, *, capacity=200, bulk_prefill=False):
    if arm != 'tp2':
        if bulk_prefill:
            raise ValueError('bulk_prefill selects the TP2 rank-local candidate; the TP1 arm is already bulk')
        return NativeARAdapter(create_resident_control(model, capacity=capacity, capture_rows=False), resident=True)
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
    return NativeARAdapter(MlpTP2GenerationSession(model, devices=(0, 1), mode='tp2',
        max_sequence_length=capacity, bulk_prefill=bool(bulk_prefill),
        bulk_prefill_rows=capacity if bulk_prefill else None), resident=False)
