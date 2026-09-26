"""Unit tier: the TP2 bulk workspace contract must fail closed.

This harness is a mechanical contract checker, so its verdicts have to be as
trustworthy as the numbers they summarize. Four ways it could wave a broken run
through are pinned here:

* a later repeat that is non-finite or differently shaped than the first;
* an over-capacity prompt that raised the wrong exception (or the right type
  with an unrelated message) being recorded as a valid refusal;
* a variant with no lengths at all passing vacuously;
* a revisit that only matched on row count while the underlying workspace was
  released and rebuilt.

No device contact: the argument refusals are pure, and the verdict functions
consume synthetic ladder records. ``argmax`` is explicitly diagnostic-only, so
nothing here treats it as a correctness oracle.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tp2_bulk_workspace_contract.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("tp2_bulk_workspace_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_script()


# -- synthetic record builders ------------------------------------------------


def _facts(*, shape: tuple[int, int] = (1, 100), finite: bool = True) -> dict:
    elements = int(shape[0]) * int(shape[1])
    return {
        "shape": list(shape),
        "dtype": "float32",
        "elements": elements,
        "nonfinite_elements": 0 if finite else elements,
        "all_finite": finite,
        "last_row_finite": finite,
        "last_row_nonfinite": 0 if finite else int(shape[1]),
        "last_row_argmax_diagnostic": 0 if finite else None,
    }


def _alloc(seed: int = 1) -> dict:
    return {"0": {"hidden": [seed, seed + 1], "logits": seed + 2}}


def _entry(
    length: int,
    *,
    revisit: bool = False,
    rows_before: int = 0,
    rows_after: int | None = None,
    alloc_before: dict | None = None,
    alloc_after: dict | None = None,
    finite: bool = True,
    shape: tuple[int, int] = (1, 100),
    repeats: int = 3,
    decode_tokens: int = 2,
    decode_steps: int | None = None,
    generated_count: int | None = None,
) -> dict:
    if rows_after is None:
        rows_after = length
    if alloc_before is None:
        alloc_before = _alloc(1)
    if alloc_after is None:
        alloc_after = alloc_before
    return {
        "length": length,
        "revisit": revisit,
        "workspace_rows_before": rows_before,
        "workspace_rows_after_first": rows_after,
        "workspace_allocation_before": alloc_before,
        "workspace_allocation_after": alloc_after,
        "finite": finite,
        "logits": _facts(shape=shape, finite=finite),
        "last_row_argmax_diagnostic": 0,
        "repeat_outputs": [
            _facts(shape=shape, finite=finite) for _ in range(max(0, repeats - 1))
        ],
        "generate_decode_steps": decode_tokens if decode_steps is None else decode_steps,
        "generated_token_count": (
            decode_tokens if generated_count is None else generated_count
        ),
    }


def _valid_prompt_sized(*, repeats: int = 3, decode_tokens: int = 2) -> dict:
    lengths = [52, 128, 512, 1024]
    entries: list[dict] = []
    alloc = _alloc(1)
    rows = 0
    for length in lengths:
        entries.append(
            _entry(
                length,
                rows_before=rows,
                rows_after=length,
                alloc_before=alloc,
                alloc_after=alloc,
                repeats=repeats,
                decode_tokens=decode_tokens,
            )
        )
        rows = length
    entries.append(
        _entry(
            52,
            revisit=True,
            rows_before=rows,
            rows_after=rows,
            alloc_before=alloc,
            alloc_after=alloc,
            repeats=repeats,
            decode_tokens=decode_tokens,
        )
    )
    return {
        "variant": "prompt-sized",
        "bulk_prefill_rows_requested": None,
        "lengths": entries,
    }


def _refusal(
    length: int,
    capacity: int,
    *,
    raised: bool = True,
    capacity_refusal: bool = True,
    rows_before: int | None = None,
    rows_after: int | None = None,
    alloc_before: dict | None = None,
    alloc_after: dict | None = None,
    poisoned_before: bool = False,
    poisoned_after: bool = False,
) -> dict:
    if rows_before is None:
        rows_before = capacity
    if rows_after is None:
        rows_after = rows_before
    if alloc_before is None:
        alloc_before = _alloc(1)
    if alloc_after is None:
        alloc_after = alloc_before
    return {
        "length": length,
        "workspace_rows_before": rows_before,
        "workspace_allocation_before": alloc_before,
        "session_poisoned_before": poisoned_before,
        "refused_over_pinned_capacity": {
            "raised": raised,
            "type": "ValueError",
            "message": (
                f"prompt of {length} tokens exceeds the bulk prefill capacity "
                f"{capacity}; chunked bulk prefill is not implemented"
            ),
            "expected_type": "ValueError",
            "expected_capacity": capacity,
            "capacity_refusal": capacity_refusal,
        },
        "workspace_rows_after_first": rows_after,
        "workspace_allocation_after": alloc_after,
        "session_poisoned_after": poisoned_after,
    }


def _valid_pinned_512(*, repeats: int = 3, decode_tokens: int = 2) -> dict:
    pinned = 512
    entries: list[dict] = []
    alloc = _alloc(1)
    rows = 0
    for length in [52, 128, 511, 512]:
        entries.append(
            _entry(
                length,
                rows_before=rows,
                rows_after=length,
                alloc_before=alloc,
                alloc_after=alloc,
                repeats=repeats,
                decode_tokens=decode_tokens,
            )
        )
        rows = length
    for length in [513, 1024]:
        entries.append(_refusal(length, pinned))
    entries.append(
        _entry(
            52,
            revisit=True,
            rows_before=rows,
            rows_after=rows,
            alloc_before=alloc,
            alloc_after=alloc,
            repeats=repeats,
            decode_tokens=decode_tokens,
        )
    )
    return {
        "variant": "pinned-512",
        "bulk_prefill_rows_requested": pinned,
        "lengths": entries,
    }


def _checks(record: dict, *, decode_tokens: int = 2, repeats: int = 3, capacity: int = 1027) -> dict:
    return contract.evaluate_variant_checks(
        record, decode_tokens=decode_tokens, repeats=repeats, capacity=capacity
    )


def _all_passed(record: dict, **kwargs) -> bool:
    checks = _checks(record, **kwargs)
    return all(value for value in checks.values() if isinstance(value, bool))


# -- argument contract --------------------------------------------------------


def test_parse_lengths_rejects_an_empty_list() -> None:
    with pytest.raises(Exception):
        contract.parse_lengths(" , ")


def test_parse_lengths_rejects_zero() -> None:
    with pytest.raises(Exception):
        contract.parse_lengths("0")


def test_main_rejects_empty_variants() -> None:
    with pytest.raises(SystemExit):
        contract.main(["--variants", ""])


def test_main_rejects_nonpositive_decode_tokens() -> None:
    with pytest.raises(SystemExit):
        contract.main(["--variants", "prompt-sized", "--decode-tokens", "0"])


def test_main_rejects_nonpositive_repeats() -> None:
    with pytest.raises(SystemExit):
        contract.main(["--variants", "prompt-sized", "--repeats", "0"])


def test_main_rejects_unknown_variants() -> None:
    with pytest.raises(SystemExit):
        contract.main(["--variants", "prompt-sized,nope"])


# -- capacity-refusal classification -----------------------------------------


def test_exact_capacity_refusal_is_recognised() -> None:
    error = ValueError(
        "prompt of 513 tokens exceeds the bulk prefill capacity 512; "
        "chunked bulk prefill is not implemented"
    )
    assert contract._is_capacity_refusal(error, 513, 512)


def test_wrong_exception_type_is_not_a_capacity_refusal() -> None:
    error = RuntimeError(
        "prompt of 513 tokens exceeds the bulk prefill capacity 512; "
        "chunked bulk prefill is not implemented"
    )
    assert not contract._is_capacity_refusal(error, 513, 512)


def test_unrelated_value_error_is_not_a_capacity_refusal() -> None:
    assert not contract._is_capacity_refusal(ValueError("bad token"), 513, 512)


def test_capacity_refusal_with_the_wrong_capacity_is_not_accepted() -> None:
    error = ValueError(
        "prompt of 513 tokens exceeds the bulk prefill capacity 4096; "
        "chunked bulk prefill is not implemented"
    )
    assert not contract._is_capacity_refusal(error, 513, 512)


class _FakeSession:
    def __init__(self, error: BaseException | None) -> None:
        self._error = error
        self.calls = 0

    def bulk_prefill(self, prompt, *, logits_rows=1):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return None


def test_probe_propagates_an_unrelated_exception() -> None:
    session = _FakeSession(RuntimeError("harness blew up"))
    with pytest.raises(RuntimeError, match="harness blew up"):
        contract._probe_capacity_refusal(session, (1, 2, 3), 3, 2)


def test_probe_records_an_unrelated_value_error_as_not_a_refusal() -> None:
    session = _FakeSession(ValueError("bad token"))
    refusal = contract._probe_capacity_refusal(session, (1, 2, 3), 3, 2)
    assert refusal["raised"] is True
    assert refusal["capacity_refusal"] is False


def test_probe_records_the_exact_capacity_refusal() -> None:
    error = ValueError(
        "prompt of 3 tokens exceeds the bulk prefill capacity 2; "
        "chunked bulk prefill is not implemented"
    )
    session = _FakeSession(error)
    refusal = contract._probe_capacity_refusal(session, (1, 2, 3), 3, 2)
    assert refusal["raised"] is True
    assert refusal["capacity_refusal"] is True


def test_probe_records_a_missing_refusal() -> None:
    refusal = contract._probe_capacity_refusal(_FakeSession(None), (1, 2, 3), 3, 2)
    assert refusal["raised"] is False
    assert refusal["capacity_refusal"] is False


# -- synthetic verdicts -------------------------------------------------------


def test_valid_prompt_sized_record_passes() -> None:
    assert _all_passed(_valid_prompt_sized())


def test_valid_pinned_512_record_with_refusals_passes() -> None:
    assert _all_passed(_valid_pinned_512())


def test_empty_selection_cannot_pass() -> None:
    record = {"variant": "prompt-sized", "bulk_prefill_rows_requested": None, "lengths": []}
    checks = _checks(record)
    assert checks["has_lengths"] is False
    assert _all_passed(record) is False


def test_nonfinite_first_output_fails() -> None:
    record = _valid_prompt_sized()
    record["lengths"][1]["finite"] = False
    record["lengths"][1]["logits"]["all_finite"] = False
    checks = _checks(record)
    assert checks["every_length_finite"] is False
    assert _all_passed(record) is False


def test_nonfinite_repeat_output_fails() -> None:
    record = _valid_prompt_sized()
    record["lengths"][2]["repeat_outputs"][1]["all_finite"] = False
    checks = _checks(record)
    assert checks["every_repeat_output_finite_and_shaped"] is False
    assert checks["every_length_finite"] is True  # the first call was fine
    assert _all_passed(record) is False


def test_malformed_repeat_shape_fails() -> None:
    record = _valid_prompt_sized()
    record["lengths"][0]["repeat_outputs"][0]["shape"] = [2, 100]
    checks = _checks(record)
    assert checks["every_repeat_output_finite_and_shaped"] is False
    assert _all_passed(record) is False


def test_missing_repeat_fails() -> None:
    record = _valid_prompt_sized()
    record["lengths"][0]["repeat_outputs"] = []
    checks = _checks(record)
    assert checks["every_repeat_output_finite_and_shaped"] is False
    assert _all_passed(record) is False


def test_malformed_first_shape_fails() -> None:
    record = _valid_prompt_sized()
    record["lengths"][0]["logits"]["shape"] = [512, 100]
    checks = _checks(record)
    assert checks["every_length_shape_expected"] is False
    assert _all_passed(record) is False


def test_wrong_refusal_type_fails() -> None:
    record = _valid_pinned_512()
    refusal = record["lengths"][4]["refused_over_pinned_capacity"]
    refusal["type"] = "RuntimeError"
    refusal["capacity_refusal"] = False
    checks = _checks(record)
    assert checks["over_pinned_lengths_refused"] is False
    assert _all_passed(record) is False


def test_wrong_refusal_message_fails() -> None:
    record = _valid_pinned_512()
    record["lengths"][5]["refused_over_pinned_capacity"]["capacity_refusal"] = False
    checks = _checks(record)
    assert checks["over_pinned_lengths_refused"] is False
    assert _all_passed(record) is False


def test_refusal_that_grew_workspace_fails() -> None:
    record = _valid_pinned_512()
    record["lengths"][4]["workspace_rows_after_first"] = 2048
    checks = _checks(record)
    assert checks["refusals_preserve_workspace_rows"] is False
    assert _all_passed(record) is False


def test_refusal_that_reallocated_workspace_fails() -> None:
    record = _valid_pinned_512()
    record["lengths"][4]["workspace_allocation_after"] = _alloc(99)
    checks = _checks(record)
    assert checks["refusals_preserve_workspace_allocation"] is False
    assert _all_passed(record) is False


def test_refusal_that_poisoned_session_fails() -> None:
    record = _valid_pinned_512()
    record["lengths"][5]["session_poisoned_after"] = True
    checks = _checks(record)
    assert checks["refusals_leave_session_healthy"] is False
    assert _all_passed(record) is False


def test_refusal_missing_state_evidence_fails_closed() -> None:
    record = _valid_pinned_512()
    refusal = record["lengths"][4]
    del refusal["workspace_rows_before"]
    del refusal["workspace_allocation_before"]
    del refusal["session_poisoned_before"]
    checks = _checks(record)
    assert checks["refusals_preserve_workspace_rows"] is False
    assert checks["refusals_preserve_workspace_allocation"] is False
    assert checks["refusals_leave_session_healthy"] is False
    assert _all_passed(record) is False


def test_missing_expected_refusal_fails() -> None:
    record = _valid_pinned_512()
    # The 1024 prompt should have been refused; recording it as a successful
    # call must not slip through as "no refusals were expected".
    record["lengths"][5] = _entry(1024, rows_before=512, rows_after=1024)
    checks = _checks(record)
    assert checks["refused_lengths_match_expected"] is False
    assert _all_passed(record) is False


def test_incomplete_generate_fails() -> None:
    record = _valid_prompt_sized()
    record["lengths"][0]["generate_decode_steps"] = 1
    checks = _checks(record)
    assert checks["every_length_generate_completed"] is False
    assert _all_passed(record) is False


def test_generated_token_count_must_match_decode_tokens() -> None:
    record = _valid_prompt_sized()
    record["lengths"][0]["generated_token_count"] = 0
    checks = _checks(record)
    assert checks["every_length_generate_completed"] is False
    assert _all_passed(record) is False


def test_revisit_with_changed_rows_fails() -> None:
    record = _valid_prompt_sized()
    record["lengths"][-1]["workspace_rows_after_first"] = 2048
    checks = _checks(record)
    assert checks["revisit_rows_unchanged"] is False
    assert _all_passed(record) is False


def test_revisit_reallocated_at_same_rows_fails() -> None:
    """Same row count, different buffers: a release/rebuild must not pass."""

    record = _valid_prompt_sized()
    record["lengths"][-1]["workspace_allocation_after"] = _alloc(99)
    checks = _checks(record)
    assert checks["revisit_rows_unchanged"] is True
    assert checks["revisit_allocation_unchanged"] is False
    assert _all_passed(record) is False


def test_summarize_checks_marks_empty_variants_failed() -> None:
    result = {"variants": {}}
    contract.summarize_checks(result, decode_tokens=2, repeats=3, capacity=1027)
    assert result["all_checks_passed"] is False


def test_summarize_checks_passes_valid_variants() -> None:
    result = {"variants": {"prompt-sized": _valid_prompt_sized()}}
    contract.summarize_checks(result, decode_tokens=2, repeats=3, capacity=1027)
    assert result["all_checks_passed"] is True


# -- oracle scope -------------------------------------------------------------


def test_argmax_is_not_a_correctness_check() -> None:
    source = SCRIPT.read_text()
    body = source.split("def evaluate_variant_checks", 1)[1].split("\ndef ", 1)[0]
    assert "argmax" not in body
    assert "ORACLE_SCOPE" in source
    assert "no numerical" in contract.ORACLE_SCOPE


def test_exact_command_uses_supplied_argv() -> None:
    import shlex

    command = contract._exact_command(["--repeats", "1", "--variants", "prompt-sized"])
    assert shlex.split(command) == [
        sys.executable,
        str(SCRIPT.resolve()),
        "--repeats",
        "1",
        "--variants",
        "prompt-sized",
    ]


def test_main_records_the_supplied_argv() -> None:
    source = SCRIPT.read_text()
    assert "_exact_command(argv)" in source


def test_capacity_refusal_handler_does_not_catch_bare_exception() -> None:
    source = SCRIPT.read_text()
    body = source.split("def _probe_capacity_refusal", 1)[1].split("\ndef ", 1)[0]
    assert "except ValueError" in body
    assert "except Exception" not in body


# -- fake-session integration over run_variant --------------------------------


class _FakeGeneration:
    def __init__(self, decode_tokens: int) -> None:
        from types import SimpleNamespace

        self.step_traces = [SimpleNamespace(kind="prefill", total_s=0.001)]
        for _ in range(decode_tokens):
            self.step_traces.append(SimpleNamespace(kind="decode", total_s=0.001))
        self.token_ids = tuple(range(1, decode_tokens + 1))


class _RecordingSession:
    """Minimal bulk-prefill session good enough to drive ``run_variant``.

    It models the one behaviour the harness cares about: the workspace grows to
    the largest prompt seen and is otherwise left alone, and an over-capacity
    prompt raises the exact ``ValueError`` without touching it.
    """

    instances: list["_RecordingSession"] = []

    def __init__(
        self,
        model,
        *,
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=0,
        bulk_prefill=True,
        bulk_prefill_rows=None,
    ) -> None:
        self.devices = tuple(devices)
        self.capacity = (
            int(bulk_prefill_rows)
            if bulk_prefill_rows is not None
            else int(max_sequence_length)
        )
        self._bulk_rows = 0
        self._bulk_hidden: dict = {}
        self._bulk_logits_buf: dict = {}
        self._poisoned = False
        self.calls = 0
        self.closed = False
        _RecordingSession.instances.append(self)

    def _allocate(self, rows: int) -> None:
        from types import SimpleNamespace

        for device in self.devices:
            self._bulk_hidden[device] = (1000 + rows, 2000 + rows)
            self._bulk_logits_buf[device] = SimpleNamespace(ptr=3000 + rows)
        self._bulk_rows = rows

    def bulk_prefill(self, token_ids, *, logits_rows=1):
        self.calls += 1
        rows = len(token_ids)
        if rows > self.capacity:
            raise ValueError(
                f"prompt of {rows} tokens exceeds the bulk prefill capacity "
                f"{self.capacity}; chunked bulk prefill is not implemented"
            )
        if self._bulk_rows < rows:
            self._allocate(rows)
        return [[0.0] * 8 for _ in range(int(logits_rows or 1))]

    def generate(self, prompt_token_ids, *, max_new_tokens=32):
        return _FakeGeneration(int(max_new_tokens))

    def close(self) -> None:
        self.closed = True


def _null_scoped(*args, **kwargs):
    from contextlib import nullcontext

    return nullcontext()


def _fake_runtime():
    from types import SimpleNamespace

    return SimpleNamespace(mem_get_info=lambda: (0, 1))


@pytest.fixture(autouse=True)
def _reset_fake_sessions():
    _RecordingSession.instances.clear()
    yield
    _RecordingSession.instances.clear()


def _patch_session(monkeypatch):
    import hipengine.distributed.tp2_generate as tp2_generate

    monkeypatch.setattr(tp2_generate, "MlpTP2GenerationSession", _RecordingSession)


def test_run_variant_repeats_one_measures_only_the_first_call(monkeypatch) -> None:
    """``--repeats 1`` must record zero repeats, not one extra measured call."""

    _patch_session(monkeypatch)
    record = contract.run_variant(
        model=Path("unused.gguf"),
        devices=(0, 1),
        lengths=[4],
        decode_tokens=2,
        repeats=1,
        token_id=1,
        variant="prompt-sized",
        capacity=16,
        scoped=_null_scoped,
        runtime=_fake_runtime(),
    )
    # One warmup plus the first call and the revisit; no repeat call.
    assert _RecordingSession.instances[-1].calls == 3
    entry = record["lengths"][0]
    assert entry["repeat_outputs"] == []
    assert entry["steady_samples_s"] == []
    assert entry["steady_median_s"] is None
    checks = contract.evaluate_variant_checks(
        record, decode_tokens=2, repeats=1, capacity=16
    )
    assert checks["every_repeat_output_finite_and_shaped"] is True
    assert all(value for value in checks.values() if isinstance(value, bool))


def test_run_variant_refusal_preserves_allocation_and_health(monkeypatch) -> None:
    """An over-pinned prompt must leave rows, pointers, and health unchanged."""

    _patch_session(monkeypatch)
    record = contract.run_variant(
        model=Path("unused.gguf"),
        devices=(0, 1),
        lengths=[4, 600],
        decode_tokens=2,
        repeats=2,
        token_id=1,
        variant="pinned-512",
        capacity=1024,
        scoped=_null_scoped,
        runtime=_fake_runtime(),
    )
    session = _RecordingSession.instances[-1]
    assert session._poisoned is False
    refusal = next(
        e for e in record["lengths"] if "refused_over_pinned_capacity" in e
    )
    assert refusal["refused_over_pinned_capacity"]["capacity_refusal"] is True
    assert refusal["workspace_rows_before"] == refusal["workspace_rows_after_first"]
    assert (
        refusal["workspace_allocation_before"]
        == refusal["workspace_allocation_after"]
    )
    assert refusal["session_poisoned_before"] is False
    assert refusal["session_poisoned_after"] is False
    checks = contract.evaluate_variant_checks(
        record, decode_tokens=2, repeats=2, capacity=1024
    )
    assert all(value for value in checks.values() if isinstance(value, bool))
