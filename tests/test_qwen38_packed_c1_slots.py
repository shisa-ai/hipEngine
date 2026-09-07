"""Diagnostic slot choice must transfer an existing free lease, never alias it."""
from types import SimpleNamespace as NS
import pytest


@pytest.mark.parametrize('capacity', [1, 2, 8])
def test_select_each_existing_lease_without_mutating_peers(capacity):
    from scripts.qwen38_packed_c1_slots import acquire_slot
    leases = [NS(session=NS(_resident_slot_index=i)) for i in range(capacity)]
    for slot in range(capacity):
        owner = NS(capacity=capacity, _available=list(reversed(leases)))
        chosen = acquire_slot(owner, slot)
        assert chosen is leases[slot]
        assert owner._available == [x for x in reversed(leases) if x is not chosen]
        with pytest.raises(ValueError, match='free lease'):
            acquire_slot(owner, slot)


@pytest.mark.parametrize('slot', [-1, 2, 8])
def test_slot_choice_rejects_out_of_capacity(slot):
    from scripts.qwen38_packed_c1_slots import acquire_slot
    owner = NS(capacity=2, _available=[])
    with pytest.raises(ValueError, match='capacity'):
        acquire_slot(owner, slot)


def test_duplicate_lease_slot_fails_without_mutation():
    from scripts.qwen38_packed_c1_slots import acquire_slot
    leases = [NS(session=NS(_resident_slot_index=1)) for _ in range(2)]
    owner = NS(capacity=2, _available=leases.copy())
    with pytest.raises(ValueError, match='free lease'):
        acquire_slot(owner, 1)
    assert owner._available == leases


def test_owner_session_implicit_slot_zero():
    from scripts.qwen38_packed_c1_slots import acquire_slot
    lease = NS(session=NS())
    owner = NS(capacity=1, _available=[lease])
    assert acquire_slot(owner, 0) is lease


@pytest.mark.parametrize('slot', [None, 1])
def test_capture_scopes_and_restores_real_owner_hook(tmp_path, slot):
    from scripts.qwen38_packed_c1_logits import PackedC1Capture
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner as Owner
    original = Owner._acquire_lease
    capture = PackedC1Capture(tmp_path / 'capture', slot=slot)
    leases = [NS(session=NS(_resident_slot_index=i)) for i in range(2)]
    owner = NS(capacity=2, _available=list(reversed(leases)))
    try:
        capture.install()
        assert (Owner._acquire_lease is original) == (slot is None)
        chosen = Owner._acquire_lease(owner)
        assert chosen is leases[0 if slot is None else slot]
        assert len(owner._available) == 1
    finally:
        capture.close(success=False)
    assert Owner._acquire_lease is original
