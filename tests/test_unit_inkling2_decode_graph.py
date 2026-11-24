"""CPU tests for the opt-in whole-token decode graph's capture rules (no GPU)."""

from types import SimpleNamespace
from unittest.mock import Mock

from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention
from hipengine.runtime.inkling2_gguf import Inkling2GGUFText

# (KV capacity, query heads, KV heads, head width, relative-bias extent) per state slot, as the
# generator builds them for a 4,096-token session: one sliding-window visit and one global visit.
UNSATURATED = ((1024, 24, 8, 128, 1024), (4096, 24, 8, 128, 8192))


def _session(position, geometry=UNSATURATED):
    return SimpleNamespace(position=position, attention=SimpleNamespace(geometry=tuple(geometry), capture_headers=None))


def _model():
    model = Inkling2GGUFText.__new__(Inkling2GGUFText)
    model.config = {"log_scaling_n_floor": 8192}
    model._graph_replays = 0
    model._graph_captures = 0
    model._graph_capture_seconds = 0.0
    return model


def test_capture_is_offered_right_after_a_short_prompt():
    # The graph used to wait for every KV window to saturate, which needs about 1,024 prompt
    # tokens, so ordinary prompts never ran it.
    model = _model()
    model._capture_graph = Mock(return_value="eager result")
    session = _session(46)
    assert model._graph_step(session, None) == "eager result"
    model._capture_graph.assert_called_once_with(session, None)


def test_capture_is_offered_once_every_window_is_saturated():
    model = _model()
    model._capture_graph = Mock(return_value="eager result")
    session = _session(4095)
    assert model._graph_step(session, None) == "eager result"


def test_capture_is_refused_past_the_log_scale_floor():
    # The clip kernels bake the position they were recorded with; the factor is 1 only below it.
    model = _model()
    model._capture_graph = Mock()
    assert model._graph_step(_session(8192), None) is None
    model._capture_graph.assert_not_called()


def test_capture_is_refused_without_attention_state_or_once_disabled():
    model = _model()
    model._capture_graph = Mock()
    assert model._graph_step(SimpleNamespace(position=46, attention=None), None) is None
    assert model._graph_step(SimpleNamespace(position=46, attention=SimpleNamespace(geometry=())), None) is None
    model._graph_disabled = True
    assert model._graph_step(_session(46), None) is None
    model._capture_graph.assert_not_called()


class _Runtime:
    """Device pointers from a rising counter; every copy, event and stream call does nothing."""

    def __init__(self):
        self.next = 0x100000

    def malloc(self, nbytes):
        self.next += (nbytes + 255) // 256 * 256 + 256
        return self.next - ((nbytes + 255) // 256 * 256 + 256)

    def free(self, ptr):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _block_table_args(*, recording, positions, capacity=1024, extent=1024):
    seen = {}
    attention = Inkling2DeviceAttention(
        runtime=_Runtime(), launch=None, write=None,
        geometry=[(capacity, 4, 2, 8, extent), (capacity, 4, 2, 8, extent)],
        block_table=lambda *args, **kwargs: seen.update(args=args),
    )
    if recording:
        attention.capture_headers = [0x100000, 0x100008]
    for position in positions:
        attention.step_block_resident(0, position, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1)
    return seen["args"]


def test_eager_attention_limits_the_relative_bias_to_the_live_window():
    # query, keys, values, rel, sink, out, spans, nq, nk, dim, extent, key, value, rows
    args = _block_table_args(recording=False, positions=[0, 1, 2, 3])
    assert args[10] == 4
    args = _block_table_args(recording=False, positions=range(5001))
    assert args[10] == 1024


def test_captured_attention_bakes_the_largest_extent_the_table_holds():
    # A replay serves later positions with more live keys, so the recorded kernel must be able to
    # reach every lag the window can hold. Keys beyond the live count are never visited.
    args = _block_table_args(recording=True, positions=[0, 1, 2, 3])
    assert args[10] == 1024
    args = _block_table_args(recording=True, positions=[0, 1, 2, 3], capacity=64, extent=1024)
    assert args[10] == 64


def test_captured_attention_reads_its_header_from_the_device_slot():
    pointers = []
    attention = Inkling2DeviceAttention(
        runtime=_Runtime(), launch=None, write=None, geometry=[(16, 4, 2, 8, 16), (16, 4, 2, 8, 16)], block_table=None,
    )

    def table(*args, **kwargs):
        pointers.append((args[6].live_counts.ptr, args[6].row_positions.ptr))

    attention.block_table = table
    attention.capture_headers = [0x200000, 0x200008]
    attention.step_block_resident(1, 3, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1)
    attention.capture_headers = None
    attention.step_block_resident(1, 4, 1, 2, 3, 4, 5, 6, 7, 8, 2, 1)
    # Recording: the (live, position) pair comes from the tick kernel's slot for this visit.
    assert pointers[0] == (0x200008, 0x20000C)
    # Eager: the pair is uploaded into the shared block header for every visit.
    assert pointers[1][1] - pointers[1][0] == 4 and pointers[1][0] != 0x200008


def _replay_state(submission):
    import numpy as np

    pin = np.zeros(4, np.float32)
    return {"runtime": Mock(), "stream": 7, "pin_ptr": pin.ctypes.data, "base": 0x1000,
            "offsets": {"h": 0, "stack": 256}, "count": 1, "submission": submission}


def test_replay_launches_through_the_submission_transport_and_counts_itself():
    import numpy as np

    model = _model()
    model.projection = Mock()
    submission = Mock()
    state = _replay_state(submission)
    session = _session(3)
    session.attention.positions = {0: 3, 1: 3}
    result, blocks = model._replay_graph(session, np.ones((1, 4), np.float32), state)
    submission.launch.assert_called_once_with(7)
    state["runtime"].graph_launch.assert_not_called()
    state["runtime"].stream_synchronize.assert_called_once_with(7)
    assert session.attention.positions == {0: 4, 1: 4}
    assert model._graph_replays == 1 and model._graph_captures == 0 and model._graph_capture_seconds == 0.0
    assert model.projection.download.call_count == 2
    assert result.shape == (1, 4) and len(blocks) == 1


def test_recorded_graph_goes_to_the_submission_transport_with_the_capture_stream(monkeypatch):
    import hipengine.runtime.inkling2_gguf as runtime_module

    created = []
    submission = Mock()
    monkeypatch.setattr(runtime_module, "create_graph_submission",
                        lambda **kwargs: created.append(kwargs) or submission)
    model = _model()
    model.projection = SimpleNamespace(backend="hip_gfx1100", target_arch="gfx1100")
    state = {"runtime": Mock(), "stream": 9}
    assert model._graph_submission(state, 0x1234) is submission
    assert created == [{"backend": "hip_gfx1100", "gfx_arch": "gfx1100", "runtime": state["runtime"],
                        "graph": 0x1234, "stream": 9}]
    assert state["submission"] is submission and state["graph_object"] == 0x1234
    state["runtime"].graph_instantiate.assert_not_called()


def test_discarding_a_capture_closes_the_submission_and_the_graph():
    model = _model()
    submission, rt = Mock(), Mock()
    model._discard_capture({"runtime": rt, "submission": submission, "graph_object": 0x1234, "pin": None},
                           _session(3))
    submission.close.assert_called_once()
    rt.graph_destroy.assert_called_once_with(0x1234)
    rt.graph_exec_destroy.assert_not_called()
