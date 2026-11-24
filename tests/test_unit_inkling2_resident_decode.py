"""CPU tests for what the Inkling 2 GGUF resident decode step keeps between requests (no GPU)."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.runtime.inkling2_gguf import Inkling2GGUFText

HIDDEN = 4
ROWS = {(0, "mlp"): 2, (1, "attn"): 2, (1, "mlp"): 2, (2, "attn"): 2, (2, "msg"): 3, (2, "mlp"): 2}
VISITS = (
    SimpleNamespace(state_slot=0, weight_layer=0, message_pass=None),      # entry: mlp read query only
    SimpleNamespace(state_slot=1, weight_layer=1, message_pass=None),      # local: attention and mlp queries
    SimpleNamespace(state_slot=2, weight_layer=2, message_pass=object()),  # loop visit: adds message queries
)


def _model():
    model = Inkling2GGUFText.__new__(Inkling2GGUFText)
    model._query_bank_ptr = None
    model._query_rows = None
    model.config = {"hidden_size": HIDDEN}
    model.plan = SimpleNamespace(visits=list(VISITS))
    model.weights = Mock()
    model.weights.dense.side_effect = lambda name: np.full((2, HIDDEN), len(name), np.float32)
    model.gguf = Mock()
    model.gguf.loop_queries.side_effect = lambda visit: np.full((3, HIDDEN), 7.0, np.float32)
    model.projection = Mock()
    model.projection.resident_vector.side_effect = lambda name, host: 0x300000
    return model


def test_residual_read_queries_are_model_residency_not_step_scratch():
    # They used to be uploaded once into the step arena. That arena is shared scratch: the next
    # request's prompt blocks wrote over it, so the first request of a process was right and every
    # later one read clobbered queries.
    model = _model()
    assert model._query_bank() == 0x300000
    assert model._query_bank() == 0x300000
    model.projection.resident_vector.assert_called_once()
    assert model.projection.resident_vector.call_args.args[0] == "attn_res.residual_read_queries"
    assert not model.projection.upload.called


def test_residual_read_query_bank_layout():
    model = _model()
    model._query_bank()
    bank = model.projection.resident_vector.call_args.args[1]
    assert bank.shape == (3, 16, HIDDEN) and bank.dtype == np.float32
    np.testing.assert_array_equal(bank[0, 8:10], 22.0)   # "attn_res.mlp_queries.0"
    np.testing.assert_array_equal(bank[1, 0:2], 23.0)    # "attn_res.attn_queries.1"
    np.testing.assert_array_equal(bank[1, 8:10], 22.0)
    np.testing.assert_array_equal(bank[2, 4:7], 7.0)     # loop message queries
    assert not bank[:, 10:16].any() and not bank[:2, 4:8].any()
    assert model._query_rows == ROWS


def test_residual_read_query_block_of_more_than_four_rows_is_refused():
    model = _model()
    model.weights.dense.side_effect = lambda name: np.ones((5, HIDDEN), np.float32)
    with pytest.raises(RuntimeError, match="exceeds its slot"):
        model._query_bank()
