"""Explicit c1 ownership control for tests with synthetic resident weights."""

def select_request_owned_c1(monkeypatch):
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpResidentServingRunner

    def prepare_c1(owner):
        if owner._ar_batch is None and owner._ar_batch_mode is None:
            owner._ar_batch_mode = "qwen4exp_c1_request_owned_step"

    monkeypatch.setattr(Qwen4ExpResidentServingRunner, "_prepare_ar_batch", prepare_c1)
