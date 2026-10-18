"""CPU control-flow coverage for the generated-history CLI."""
from types import SimpleNamespace
import json


def test_public_request_runs_before_resident_session_releases_runner(monkeypatch, tmp_path):
    from scripts import gemma4_generated_history_gate as gate
    import numpy as np

    state = dict(public=False, closed=False)

    class LLM:
        def _get_text_generator(self):
            return SimpleNamespace(tokenizer=object())

        def generate(self, tokens, sampling):
            assert not state['closed'], 'public request used a released runner'
            state['public'] = True
            return ['text']

        def close(self):
            pass

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            assert state['public']
            return self

        def __exit__(self, *args):
            state['closed'] = True

    def trajectory(session, **kwargs):
        if kwargs['forced_input_ids'] is not None:
            assert kwargs['forced_input_ids'] == [1]
            assert kwargs['teacher_row_ids'] == [1, 2]
        return np.zeros((2, 3), dtype=np.float32), [], [
            dict(teacher_token_id=1), dict(teacher_token_id=2)]

    monkeypatch.setattr(gate, '_resolve_generator', lambda *a: (LLM(), object(), {}))
    monkeypatch.setattr(gate, 'Gemma4ResidentSession', Session)
    monkeypatch.setattr(gate, '_prompt_rows', lambda *a, **k: [dict(id='case', category='code')])
    monkeypatch.setattr(gate, '_prompt_token_ids', lambda *a, **k: [1, 2, 3])
    monkeypatch.setattr(gate, '_trajectory_with_controls', trajectory)
    monkeypatch.setattr(gate, 'gemma4_moe_expert_route_counts',
                        lambda: {'selected_gemv': int(state['public'])})
    monkeypatch.setattr(gate.sys, 'argv', ['gate', '--model', str(tmp_path/'model.gguf'),
                                         '--output-dir', str(tmp_path/'evidence')])
    assert gate.main() == 0
    result = json.loads((tmp_path/'evidence/verdict.json').read_text())
    assert result['combined']['passed']
    assert result['public_routes'] == {'selected_gemv': 1}
    assert state == dict(public=True, closed=True)
