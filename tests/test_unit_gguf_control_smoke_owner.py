"""Host ownership checks for the GGUF control comparison driver."""

from types import SimpleNamespace

import pytest

from scripts import execution_profile_gguf_control_smoke as smoke


@pytest.mark.parametrize("fail", [False, True])
def test_route_owner_restores_fields_without_process_mutation(monkeypatch, fail):
    owner = SimpleNamespace(router_f32w_coop=None,
                            router_f32w_persistent_counter=False,
                            q8_t16_rowtile_all=True)
    session = SimpleNamespace(runner=owner)
    before = vars(owner).copy()
    monkeypatch.setenv(smoke._ROUTER_COOP_ENV, "caller")
    try:
        with smoke._route_owner(session, smoke._STRICT_ROUTE_ENV):
            assert all(value is False for value in vars(owner).values())
            with smoke._route_owner(session, smoke._PRODUCTION_ROUTE_ENV):
                assert owner.router_f32w_coop is True
                assert owner.router_f32w_persistent_counter is True
                assert owner.q8_t16_rowtile_all is True
            assert all(value is False for value in vars(owner).values())
            if fail:
                raise RuntimeError("comparison failure")
    except RuntimeError:
        assert fail
    assert vars(owner) == before
    import os
    assert os.environ[smoke._ROUTER_COOP_ENV] == "caller"
