"""CPU-only fail-closed diagnostic contract; no HIP or model allocation."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts import tp2_xtx_tp1_eager_stage_probe as probe


@pytest.mark.parametrize("logits", [None, [], [1, np.nan, 2], [1, np.inf, 2],
                                    [0, 0, 0], [[1, 2, 3], [1, 2, 3]], [1, 2]])
def test_invalid_logits_fail(logits):
    with pytest.raises(ValueError):
        probe.validate_result(SimpleNamespace(token_id=1, logits=logits), 3)


@pytest.mark.parametrize("token", [-1, 3, 2**63-1, 1.5, None, True])
def test_invalid_token_fails(token):
    with pytest.raises(ValueError):
        probe.validate_result(SimpleNamespace(token_id=token, logits=np.array([1., 3., 2.])), 3)


def test_resident_single_row_logits_shape_is_valid():
    # The real resident _logits_host allocation is (1, vocab), not (vocab,).
    result = probe.validate_result(SimpleNamespace(token_id=1, logits=np.array([[1., 3., 2.]])), 3)
    assert result['logits']['shape'] == [1, 3]


def test_valid_logits_do_not_claim_unwritten():
    result = probe.validate_result(SimpleNamespace(token_id=1, logits=np.array([1., 3., 2.])), 3)
    assert result['token_id'] == 1
    assert 'sentinel_fraction' not in result['logits']


@pytest.mark.parametrize('failure', ['build', 'prefill-auto', 'eager-step', 'teardown'])
def test_guard_stops_and_persists_failure(tmp_path, failure):
    path = tmp_path / 'partial.json'
    recorder = probe.StageRecorder(path, {})
    calls = []
    def work(name):
        calls.append(name)
        current = json.loads(path.read_text())
        assert current['active_stage'] == name
        assert current['status'] == 'running'
        if name == failure:
            raise RuntimeError('injected')
        return {}
    for name in ['build', 'prefill-auto', 'eager-step', 'teardown']:
        recorder.guard(name, lambda: work(name))
    recorder.finish()
    payload = json.loads(path.read_text())
    assert payload['first_bad_stage'] == failure
    assert payload['status'] == 'failed'
    assert recorder.exit_code == 1
    assert calls[-1] == failure


def test_synchronization_failure_stops(tmp_path):
    recorder = probe.StageRecorder(tmp_path / 'a.json', {})
    def sync():
        raise RuntimeError('sync fault')
    assert not recorder.guard('prefill-auto', lambda: {}, sync=sync)
    assert not recorder.guard('step', lambda: pytest.fail('launched after fault'))


def test_incremental_artifact_before_unfinished_stage(tmp_path):
    path = tmp_path / 'a.json'
    recorder = probe.StageRecorder(path, {})
    def interrupted():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        recorder.guard('build', interrupted)
    payload = json.loads(path.read_text())
    assert payload['status'] == 'running'
    assert payload['active_stage'] == 'build'
    assert payload['stages'][-1]['ok'] is None


def test_invalid_prefill_never_reaches_step_or_close(tmp_path):
    calls = []
    class Session:
        runtime = SimpleNamespace(device_synchronize=lambda: calls.append('sync'))
        runner = SimpleNamespace(vocab_size=3)
        def reset(self): calls.append('reset')
        def prefill(self, *a, **kw):
            calls.append('prefill')
            return SimpleNamespace(token_id=1, logits=np.array([1., np.nan, 2.]))
        def step(self, *a, **kw): pytest.fail('step after bad prefill')
        def close(self): pytest.fail('close after bad prefill')
    recorder = probe.StageRecorder(tmp_path / 'a.json', {})
    probe.run_stages(Session(), [1], recorder)
    assert recorder.exit_code == 1
    assert recorder.artifact['first_bad_stage'] == 'prefill-auto'


def test_greedy_token_must_match_logits():
    with pytest.raises(ValueError, match='argmax'):
        probe.validate_result(SimpleNamespace(token_id=0, logits=np.array([1., 3., 2.])), 3)


def test_main_build_failure_emits_artifact_and_exits_nonzero(tmp_path, monkeypatch):
    class Exit(Exception):
        pass
    codes = []
    def exit_(code):
        codes.append(code)
        raise Exit()
    from hipengine.core import build
    monkeypatch.setattr(build, '_resolve_cache_root', lambda _: tmp_path / 'cache')
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *a, **k: 'test-source')
    monkeypatch.setattr(probe.os, '_exit', exit_)
    monkeypatch.setenv('HIPENGINE_GGUF_DECODE_REPACK', '0')
    path = tmp_path / 'failure.json'
    with pytest.raises(Exit):
        probe.main(['--model', str(tmp_path / 'missing.gguf'), '--json', str(path)])
    payload = json.loads(path.read_text())
    assert codes == [1]
    assert payload['first_bad_stage'] == 'build'
    assert payload['status'] == 'failed'
    assert payload['env']['HIPENGINE_GGUF_DECODE_REPACK'] == '1'
    assert len(payload['stages']) == 1


@pytest.mark.parametrize('graph_logits,passes', [([1., 3., 2.], True), ([3., 1., 2.], False)])
def test_graph_eager_comparison_is_numerical_not_finite_only(graph_logits, passes):
    eager = SimpleNamespace(token_id=1, logits=np.array([[1., 3., 2.]]))
    graph = SimpleNamespace(token_id=int(np.argmax(graph_logits)), logits=np.array([graph_logits]))
    if passes:
        assert probe.compare_graph_eager(eager, graph, 3)['top1_agreement'] == 1.0
    else:
        with pytest.raises(ValueError):
            probe.compare_graph_eager(eager, graph, 3)


@pytest.mark.parametrize('bad_device_cursor', [False, True])
def test_graph_reconstructs_same_input_position_and_closes(tmp_path, monkeypatch, bad_device_cursor):
    class Session:
        runner = SimpleNamespace(vocab_size=3)
        runtime = SimpleNamespace(device_synchronize=lambda: None)
        position = 0
        def reset(self): self.position = 0
        def result(self): return SimpleNamespace(token_id=1, logits=np.array([[1., 3., 2.]]))
        def prefill(self, tokens, **kw): self.position = len(tokens); return self.result()
        def step(self, token, **kw):
            assert token == 1
            steps.append(self.position)
            self.position += 1
            return self.result()
        def capture_decode_graph(self, **kw):
            assert kw['position'] == self.position
            session = self
            class Graph:
                def replay(self, n):
                    session.position += n
                def read_sample(self, **kw): return session.result()
                def transport_provenance(self): return {'transport': 'hipgraph'}
                def close(self): closed.append('graph')
            return Graph()
        def close(self): closed.append('session')
    steps, closed = [], []
    monkeypatch.setattr(probe, 'graph_device_state', lambda session: {
        'position': session.position + int(bad_device_cursor),
        'context': session.position + 1, 'sampled_token': 1})
    record = probe.StageRecorder(tmp_path / 'g.json', {})
    probe.run_stages(Session(), [0, 1], record, with_graph=True)
    if bad_device_cursor:
        assert record.exit_code == 1
        assert record.artifact['first_bad_stage'] == 'graph-capture'
        assert steps == [2] and closed == []
        return
    assert record.exit_code == 0
    assert steps == [2, 2, 3, 4]  # initial step, reconstruction, two matched graph inputs
    assert closed == ['graph', 'session']
    comparison = next(s for s in record.artifact['stages'] if s['stage'] == 'graph-eager-compare')
    assert comparison['input_position'] == 3
    assert comparison['output_position'] == 4


def test_graph_state_reads_device_not_host_cursor():
    import ctypes
    fields = [np.array([x], dtype=np.int64) for x in (22, 23, 12305)]
    buffers = [SimpleNamespace(ptr=x.ctypes.data) for x in fields]
    session = SimpleNamespace(position=999,
        scratch=SimpleNamespace(position_buf=buffers[0], context_buf=buffers[1]),
        _lm_out_index=buffers[2], runtime=SimpleNamespace(device_synchronize=lambda: None,
            memcpy=lambda dst, src, nbytes, kind: ctypes.memmove(dst, src, nbytes)))
    assert probe.graph_device_state(session) == {'position': 22, 'context': 23, 'sampled_token': 12305}


def test_success_includes_teardown(tmp_path):
    calls = []
    result = SimpleNamespace(token_id=1, logits=np.array([1., 3., 2.]))
    session = SimpleNamespace(runtime=SimpleNamespace(device_synchronize=lambda: None),
        runner=SimpleNamespace(vocab_size=3), reset=lambda: None,
        prefill=lambda *a, **k: result, step=lambda *a, **k: result,
        close=lambda: calls.append('close'))
    recorder = probe.StageRecorder(tmp_path / 'a.json', {})
    probe.run_stages(session, [1], recorder)
    recorder.finish()
    assert recorder.exit_code == 0
    assert calls == ['close']
    assert recorder.artifact['status'] == 'complete'
