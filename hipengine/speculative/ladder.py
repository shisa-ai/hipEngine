"""Target-verifier layer-ladder diagnostics.

The production DFlash verifier must eventually run one native target forward over
``[root, candidates...]`` rows and make the selected row's state commit-ready.
This module provides the deterministic comparison contract used while that path
is assembled layer by layer: compare a serial c=1 replay against a bulk
verify-row execution at each layer-family boundary and report the first failing
row/tensor.

The synthetic harness is intentionally small and torch-free.  It is not a model
implementation; it exercises the same topology requirements as DFlash chain
verification (root rows, parent rows, candidate depths, per-row state selection,
and terminal logits) so runtime/GPU implementations can plug their real stage
snapshots into the same comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Sequence

from hipengine.speculative.interfaces import TargetVerifyBatch

Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class TargetVerifyStateRows:
    """Named per-row state snapshot captured at one verifier ladder stage."""

    name: str
    rows: Matrix

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("state name must be non-empty")
        _validate_matrix(self.rows, f"state {self.name}")

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class TargetVerifyStageSnapshot:
    """Bulk/serial verifier outputs at one layer-family boundary."""

    stage: str
    family: str
    layer_index: int | None
    hidden_rows: Matrix
    logits_rows: Matrix = ()
    state_rows: tuple[TargetVerifyStateRows, ...] = ()

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage must be non-empty")
        if not self.family:
            raise ValueError("family must be non-empty")
        _validate_matrix(self.hidden_rows, "hidden_rows")
        rows = len(self.hidden_rows)
        if self.logits_rows:
            _validate_matrix(self.logits_rows, "logits_rows")
            if len(self.logits_rows) != rows:
                raise ValueError("logits row count must match hidden row count")
        seen: set[str] = set()
        for state in self.state_rows:
            if state.name in seen:
                raise ValueError(f"duplicate state snapshot {state.name!r}")
            seen.add(state.name)
            if state.row_count != rows:
                raise ValueError(f"state {state.name!r} row count must match hidden row count")

    @property
    def row_count(self) -> int:
        return len(self.hidden_rows)

    def state_for_row(self, name: str, row: int) -> tuple[float, ...]:
        for state in self.state_rows:
            if state.name == name:
                return state.rows[row]
        raise KeyError(name)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "family": self.family,
            "layer_index": self.layer_index,
            "rows": self.row_count,
            "hidden_width": len(self.hidden_rows[0]) if self.hidden_rows else 0,
            "logits_width": len(self.logits_rows[0]) if self.logits_rows else 0,
            "states": [state.name for state in self.state_rows],
        }


@dataclass(frozen=True, slots=True)
class TargetVerifyLadderMismatch:
    """First row/column mismatch at a ladder stage."""

    stage: str
    family: str
    layer_index: int | None
    tensor: str
    row: int
    column: int
    serial_value: float
    bulk_value: float
    abs_diff: float
    tolerance: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "family": self.family,
            "layer_index": self.layer_index,
            "tensor": self.tensor,
            "row": self.row,
            "column": self.column,
            "serial_value": self.serial_value,
            "bulk_value": self.bulk_value,
            "abs_diff": self.abs_diff,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class TargetVerifyLadderStageComparison:
    """Comparator result for one serial-vs-bulk stage pair."""

    stage: str
    family: str
    layer_index: int | None
    max_abs: float
    max_rel: float
    passed: bool
    mismatch: TargetVerifyLadderMismatch | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "family": self.family,
            "layer_index": self.layer_index,
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "passed": self.passed,
            "mismatch": None if self.mismatch is None else self.mismatch.to_json_dict(),
        }


@dataclass(frozen=True, slots=True)
class TargetVerifyLayerLadderResult:
    """Full layer-ladder comparison for one target-verification batch."""

    serial: tuple[TargetVerifyStageSnapshot, ...]
    bulk: tuple[TargetVerifyStageSnapshot, ...]
    comparisons: tuple[TargetVerifyLadderStageComparison, ...]
    atol: float
    rtol: float

    @property
    def passed(self) -> bool:
        return all(comparison.passed for comparison in self.comparisons)

    @property
    def first_failure(self) -> TargetVerifyLadderMismatch | None:
        for comparison in self.comparisons:
            if comparison.mismatch is not None:
                return comparison.mismatch
        return None

    def selectable_state(self, *, row: int, stage: str, state: str, bulk: bool = True) -> tuple[float, ...]:
        snapshots = self.bulk if bulk else self.serial
        for snapshot in snapshots:
            if snapshot.stage == stage:
                return snapshot.state_for_row(state, row)
        raise KeyError(stage)

    def terminal_logits(self, *, bulk: bool = True) -> Matrix:
        snapshots = self.bulk if bulk else self.serial
        for snapshot in reversed(snapshots):
            if snapshot.logits_rows:
                return snapshot.logits_rows
        raise ValueError("ladder result has no terminal logits")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "atol": self.atol,
            "rtol": self.rtol,
            "first_failure": None if self.first_failure is None else self.first_failure.to_json_dict(),
            "comparisons": [comparison.to_json_dict() for comparison in self.comparisons],
            "stages": [snapshot.to_json_dict() for snapshot in self.bulk],
        }


def compare_target_verify_ladder(
    serial: Sequence[TargetVerifyStageSnapshot],
    bulk: Sequence[TargetVerifyStageSnapshot],
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> TargetVerifyLayerLadderResult:
    """Compare serial c=1 and bulk verifier snapshots stage by stage."""

    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol/rtol must be non-negative")
    serial_snapshots = tuple(serial)
    bulk_snapshots = tuple(bulk)
    if len(serial_snapshots) != len(bulk_snapshots):
        raise ValueError("serial and bulk ladders must have the same number of stages")
    comparisons = []
    for serial_stage, bulk_stage in zip(serial_snapshots, bulk_snapshots, strict=True):
        comparisons.append(_compare_stage(serial_stage, bulk_stage, atol=atol, rtol=rtol))
    return TargetVerifyLayerLadderResult(
        serial=serial_snapshots,
        bulk=bulk_snapshots,
        comparisons=tuple(comparisons),
        atol=float(atol),
        rtol=float(rtol),
    )


def synthetic_chain_target_verify_ladder(
    batch: TargetVerifyBatch,
    *,
    hidden_size: int = 8,
    vocab_size: int = 24,
    layer_limit: int | None = None,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> TargetVerifyLayerLadderResult:
    """Run the deterministic chain-verifier ladder and compare serial vs bulk rows.

    This helper is a correctness fixture for the DFlash chain topology.  It
    validates that a bulk topological execution can expose the same hidden rows,
    logits, and selectable per-row state as independent serial c=1 replays for
    fixed synthetic candidates.
    """

    serial = synthetic_chain_target_verify_snapshots(
        batch,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        layer_limit=layer_limit,
        execution="serial",
    )
    bulk = synthetic_chain_target_verify_snapshots(
        batch,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        layer_limit=layer_limit,
        execution="bulk",
    )
    return compare_target_verify_ladder(serial, bulk, atol=atol, rtol=rtol)


def synthetic_chain_target_verify_snapshots(
    batch: TargetVerifyBatch,
    *,
    hidden_size: int = 8,
    vocab_size: int = 24,
    layer_limit: int | None = None,
    execution: str = "bulk",
) -> tuple[TargetVerifyStageSnapshot, ...]:
    """Return deterministic serial or bulk snapshots for a verify-chain batch."""

    _check_synthetic_args(batch, hidden_size=hidden_size, vocab_size=vocab_size)
    stages = _stage_specs(layer_limit)
    if execution == "bulk":
        return _synthetic_bulk_snapshots(batch, stages, hidden_size=hidden_size, vocab_size=vocab_size)
    if execution == "serial":
        return _synthetic_serial_snapshots(batch, stages, hidden_size=hidden_size, vocab_size=vocab_size)
    raise ValueError("execution must be 'serial' or 'bulk'")


def _compare_stage(
    serial: TargetVerifyStageSnapshot,
    bulk: TargetVerifyStageSnapshot,
    *,
    atol: float,
    rtol: float,
) -> TargetVerifyLadderStageComparison:
    if (serial.stage, serial.family, serial.layer_index) != (bulk.stage, bulk.family, bulk.layer_index):
        raise ValueError("serial and bulk stages must have matching stage/family/layer_index")
    tensors: list[tuple[str, Matrix, Matrix]] = [("hidden", serial.hidden_rows, bulk.hidden_rows)]
    if bool(serial.logits_rows) != bool(bulk.logits_rows):
        raise ValueError("serial and bulk stages must both include or omit logits")
    if serial.logits_rows:
        tensors.append(("logits", serial.logits_rows, bulk.logits_rows))
    serial_states = {state.name: state.rows for state in serial.state_rows}
    bulk_states = {state.name: state.rows for state in bulk.state_rows}
    if serial_states.keys() != bulk_states.keys():
        raise ValueError("serial and bulk stages must expose the same state names")
    for name in sorted(serial_states):
        tensors.append((f"state:{name}", serial_states[name], bulk_states[name]))

    max_abs = 0.0
    max_rel = 0.0
    first_mismatch: TargetVerifyLadderMismatch | None = None
    for tensor_name, serial_rows, bulk_rows in tensors:
        diff = _compare_matrix_values(
            serial_rows,
            bulk_rows,
            tensor_name=tensor_name,
            stage=serial,
            atol=atol,
            rtol=rtol,
        )
        max_abs = max(max_abs, diff.max_abs)
        max_rel = max(max_rel, diff.max_rel)
        if first_mismatch is None and diff.first_mismatch is not None:
            first_mismatch = diff.first_mismatch
    return TargetVerifyLadderStageComparison(
        stage=serial.stage,
        family=serial.family,
        layer_index=serial.layer_index,
        max_abs=max_abs,
        max_rel=max_rel,
        passed=first_mismatch is None,
        mismatch=first_mismatch,
    )


@dataclass(frozen=True, slots=True)
class _MatrixDiff:
    max_abs: float
    max_rel: float
    first_mismatch: TargetVerifyLadderMismatch | None


def _compare_matrix_values(
    serial: Matrix,
    bulk: Matrix,
    *,
    tensor_name: str,
    stage: TargetVerifyStageSnapshot,
    atol: float,
    rtol: float,
) -> _MatrixDiff:
    if len(serial) != len(bulk):
        raise ValueError(f"{tensor_name} row count mismatch")
    max_abs = 0.0
    max_rel = 0.0
    first: TargetVerifyLadderMismatch | None = None
    for row_index, (serial_row, bulk_row) in enumerate(zip(serial, bulk, strict=True)):
        if len(serial_row) != len(bulk_row):
            raise ValueError(f"{tensor_name} width mismatch at row {row_index}")
        for column, (serial_value, bulk_value) in enumerate(zip(serial_row, bulk_row, strict=True)):
            expected = float(serial_value)
            observed = float(bulk_value)
            abs_diff = abs(expected - observed)
            rel_diff = abs_diff / max(abs(expected), 1.0e-12)
            tolerance = float(atol) + float(rtol) * abs(expected)
            max_abs = max(max_abs, abs_diff)
            max_rel = max(max_rel, rel_diff)
            if first is None and abs_diff > tolerance:
                first = TargetVerifyLadderMismatch(
                    stage=stage.stage,
                    family=stage.family,
                    layer_index=stage.layer_index,
                    tensor=tensor_name,
                    row=row_index,
                    column=column,
                    serial_value=expected,
                    bulk_value=observed,
                    abs_diff=abs_diff,
                    tolerance=tolerance,
                )
    return _MatrixDiff(max_abs=max_abs, max_rel=max_rel, first_mismatch=first)


def _check_synthetic_args(batch: TargetVerifyBatch, *, hidden_size: int, vocab_size: int) -> None:
    if batch.mode != "verify_chain":
        raise ValueError("synthetic chain ladder requires mode='verify_chain'")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")


def _stage_specs(layer_limit: int | None) -> tuple[tuple[str, str, int | None], ...]:
    stages: tuple[tuple[str, str, int | None], ...] = (
        ("embedding_position", "embedding/position", 0),
        ("input_rmsnorm", "rmsnorm", 1),
        ("linear_attn_conv_gdn", "linear_attention", 2),
        ("full_attn_kv", "full_attention", 3),
        ("moe", "moe", 4),
        ("final_norm_lm_head", "terminal", None),
    )
    if layer_limit is None:
        return stages
    limit = int(layer_limit)
    if limit <= 0:
        raise ValueError("layer_limit must be positive")
    if limit > len(stages):
        raise ValueError("layer_limit exceeds synthetic ladder stages")
    return stages[:limit]


def _synthetic_serial_snapshots(
    batch: TargetVerifyBatch,
    stages: tuple[tuple[str, str, int | None], ...],
    *,
    hidden_size: int,
    vocab_size: int,
) -> tuple[TargetVerifyStageSnapshot, ...]:
    per_stage_hidden: list[list[tuple[float, ...]]] = [[] for _ in stages]
    per_stage_logits: list[list[tuple[float, ...]]] = [[] for _ in stages]
    per_stage_states: list[dict[str, list[tuple[float, ...]]]] = [dict() for _ in stages]
    for row in range(batch.rows):
        row_outputs = _run_synthetic_serial_path(batch, row, stages, hidden_size=hidden_size, vocab_size=vocab_size)
        for index, output in enumerate(row_outputs):
            per_stage_hidden[index].append(output.hidden)
            if output.logits:
                per_stage_logits[index].append(output.logits)
            for name, values in output.states:
                per_stage_states[index].setdefault(name, []).append(values)
    return tuple(
        TargetVerifyStageSnapshot(
            stage=stage,
            family=family,
            layer_index=layer_index,
            hidden_rows=tuple(per_stage_hidden[index]),
            logits_rows=tuple(per_stage_logits[index]),
            state_rows=tuple(
                TargetVerifyStateRows(name=name, rows=tuple(rows))
                for name, rows in sorted(per_stage_states[index].items())
            ),
        )
        for index, (stage, family, layer_index) in enumerate(stages)
    )


@dataclass(frozen=True, slots=True)
class _RowStageOutput:
    hidden: tuple[float, ...]
    logits: tuple[float, ...] = ()
    states: tuple[tuple[str, tuple[float, ...]], ...] = ()


def _run_synthetic_serial_path(
    batch: TargetVerifyBatch,
    row: int,
    stages: tuple[tuple[str, str, int | None], ...],
    *,
    hidden_size: int,
    vocab_size: int,
) -> tuple[_RowStageOutput, ...]:
    path = _row_path(batch, row)
    request_id = batch.row_to_request[row]
    linear_state = _root_state(request_id, hidden_size, salt=11)
    kv_key = _root_state(request_id, hidden_size, salt=17)
    kv_value = _root_state(request_id, hidden_size, salt=23)
    final_outputs: tuple[_RowStageOutput, ...] = ()
    for path_row in path:
        outputs = _run_synthetic_token(
            batch,
            path_row,
            stages,
            linear_state=linear_state,
            kv_key=kv_key,
            kv_value=kv_value,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        final_outputs = outputs
        for output in outputs:
            state_map = dict(output.states)
            if "linear_recurrent" in state_map:
                linear_state = state_map["linear_recurrent"]
            if "kv_key" in state_map:
                kv_key = state_map["kv_key"]
            if "kv_value" in state_map:
                kv_value = state_map["kv_value"]
    return final_outputs


def _row_path(batch: TargetVerifyBatch, row: int) -> tuple[int, ...]:
    path: list[int] = []
    current = int(row)
    seen: set[int] = set()
    while current >= 0:
        if current in seen:
            raise ValueError("target verify parent rows contain a cycle")
        seen.add(current)
        path.append(current)
        current = int(batch.parent_rows[current])
    path.reverse()
    return tuple(path)


def _synthetic_bulk_snapshots(
    batch: TargetVerifyBatch,
    stages: tuple[tuple[str, str, int | None], ...],
    *,
    hidden_size: int,
    vocab_size: int,
) -> tuple[TargetVerifyStageSnapshot, ...]:
    hidden_rows = [_embedding(batch.tokens[row], batch.positions[row], row, hidden_size) for row in range(batch.rows)]
    snapshots: list[TargetVerifyStageSnapshot] = []
    if _include_stage(stages, "embedding_position"):
        snapshots.append(_snapshot(stages, "embedding_position", hidden_rows))
    if _include_stage(stages, "input_rmsnorm"):
        hidden_rows = [_rmsnorm(row, salt=3) for row in hidden_rows]
        snapshots.append(_snapshot(stages, "input_rmsnorm", hidden_rows))
    if _include_stage(stages, "linear_attn_conv_gdn"):
        next_rows: list[tuple[float, ...]] = []
        linear_states: list[tuple[float, ...]] = []
        conv_states: list[tuple[float, ...]] = []
        for row, hidden in enumerate(hidden_rows):
            parent = batch.parent_rows[row]
            parent_state = _root_state(batch.row_to_request[row], hidden_size, salt=11) if parent < 0 else linear_states[parent]
            next_hidden, conv_state, linear_state = _linear_attn_stage(hidden, parent_state, batch.draft_depths[row], batch.positions[row])
            next_rows.append(next_hidden)
            conv_states.append(conv_state)
            linear_states.append(linear_state)
        hidden_rows = next_rows
        snapshots.append(
            _snapshot(
                stages,
                "linear_attn_conv_gdn",
                hidden_rows,
                states=(
                    TargetVerifyStateRows("linear_conv", tuple(conv_states)),
                    TargetVerifyStateRows("linear_recurrent", tuple(linear_states)),
                ),
            )
        )
    if _include_stage(stages, "full_attn_kv"):
        next_rows = []
        kv_keys: list[tuple[float, ...]] = []
        kv_values: list[tuple[float, ...]] = []
        for row, hidden in enumerate(hidden_rows):
            parent = batch.parent_rows[row]
            parent_key = _root_state(batch.row_to_request[row], hidden_size, salt=17) if parent < 0 else kv_keys[parent]
            parent_value = _root_state(batch.row_to_request[row], hidden_size, salt=23) if parent < 0 else kv_values[parent]
            next_hidden, key, value = _full_attn_stage(hidden, parent_key, parent_value, batch.positions[row])
            next_rows.append(next_hidden)
            kv_keys.append(key)
            kv_values.append(value)
        hidden_rows = next_rows
        snapshots.append(
            _snapshot(
                stages,
                "full_attn_kv",
                hidden_rows,
                states=(
                    TargetVerifyStateRows("kv_key", tuple(kv_keys)),
                    TargetVerifyStateRows("kv_value", tuple(kv_values)),
                ),
            )
        )
    if _include_stage(stages, "moe"):
        next_rows = []
        experts = []
        for row, hidden in enumerate(hidden_rows):
            next_hidden, expert = _moe_stage(hidden, batch.tokens[row], row)
            next_rows.append(next_hidden)
            experts.append((float(expert),))
        hidden_rows = next_rows
        snapshots.append(_snapshot(stages, "moe", hidden_rows, states=(TargetVerifyStateRows("selected_expert", tuple(experts)),)))
    if _include_stage(stages, "final_norm_lm_head"):
        hidden_rows = [_rmsnorm(row, salt=7) for row in hidden_rows]
        logits = [_lm_head_logits(row, vocab_size) for row in hidden_rows]
        snapshots.append(
            _snapshot(
                stages,
                "final_norm_lm_head",
                hidden_rows,
                logits=logits,
                states=(TargetVerifyStateRows("final_hidden", tuple(hidden_rows)),),
            )
        )
    return tuple(snapshots)


def _run_synthetic_token(
    batch: TargetVerifyBatch,
    row: int,
    stages: tuple[tuple[str, str, int | None], ...],
    *,
    linear_state: tuple[float, ...],
    kv_key: tuple[float, ...],
    kv_value: tuple[float, ...],
    hidden_size: int,
    vocab_size: int,
) -> tuple[_RowStageOutput, ...]:
    outputs: list[_RowStageOutput] = []
    hidden = _embedding(batch.tokens[row], batch.positions[row], row, hidden_size)
    if _include_stage(stages, "embedding_position"):
        outputs.append(_RowStageOutput(hidden=hidden))
    if _include_stage(stages, "input_rmsnorm"):
        hidden = _rmsnorm(hidden, salt=3)
        outputs.append(_RowStageOutput(hidden=hidden))
    if _include_stage(stages, "linear_attn_conv_gdn"):
        hidden, conv_state, linear_state = _linear_attn_stage(hidden, linear_state, batch.draft_depths[row], batch.positions[row])
        outputs.append(
            _RowStageOutput(
                hidden=hidden,
                states=(
                    ("linear_conv", conv_state),
                    ("linear_recurrent", linear_state),
                ),
            )
        )
    if _include_stage(stages, "full_attn_kv"):
        hidden, kv_key, kv_value = _full_attn_stage(hidden, kv_key, kv_value, batch.positions[row])
        outputs.append(
            _RowStageOutput(
                hidden=hidden,
                states=(
                    ("kv_key", kv_key),
                    ("kv_value", kv_value),
                ),
            )
        )
    if _include_stage(stages, "moe"):
        hidden, expert = _moe_stage(hidden, batch.tokens[row], row)
        outputs.append(_RowStageOutput(hidden=hidden, states=(("selected_expert", (float(expert),)),)))
    if _include_stage(stages, "final_norm_lm_head"):
        hidden = _rmsnorm(hidden, salt=7)
        logits = _lm_head_logits(hidden, vocab_size)
        outputs.append(_RowStageOutput(hidden=hidden, logits=logits, states=(("final_hidden", hidden),)))
    return tuple(outputs)


def _include_stage(stages: tuple[tuple[str, str, int | None], ...], stage: str) -> bool:
    return any(item[0] == stage for item in stages)


def _snapshot(
    stages: tuple[tuple[str, str, int | None], ...],
    stage_name: str,
    hidden: Sequence[Sequence[float]],
    *,
    logits: Sequence[Sequence[float]] = (),
    states: tuple[TargetVerifyStateRows, ...] = (),
) -> TargetVerifyStageSnapshot:
    for stage, family, layer_index in stages:
        if stage == stage_name:
            return TargetVerifyStageSnapshot(
                stage=stage,
                family=family,
                layer_index=layer_index,
                hidden_rows=_matrix(hidden),
                logits_rows=_matrix(logits) if logits else (),
                state_rows=states,
            )
    raise KeyError(stage_name)


def _embedding(token: int, position: int, row: int, hidden_size: int) -> tuple[float, ...]:
    return tuple(
        (((int(token) + 3) * (dim + 1) + (int(position) + 5) * (dim + 2) + (row + 7) * 3) % 101) / 50.0 - 1.0
        for dim in range(hidden_size)
    )


def _root_state(request_id: int, hidden_size: int, *, salt: int) -> tuple[float, ...]:
    return tuple((((int(request_id) + salt) * (dim + 5)) % 89) / 80.0 - 0.55 for dim in range(hidden_size))


def _rmsnorm(row: Sequence[float], *, salt: int) -> tuple[float, ...]:
    denom = sqrt(sum(float(value) * float(value) for value in row) / len(row) + 1.0e-6)
    return tuple((float(value) / denom) * (1.0 + (((index + salt) % 5) - 2) * 0.01) for index, value in enumerate(row))


def _linear_attn_stage(
    hidden: Sequence[float],
    parent_state: Sequence[float],
    depth: int,
    position: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    conv = tuple(0.7 * float(h) + 0.2 * float(p) + 0.01 * (index + 1) for index, (h, p) in enumerate(zip(hidden, parent_state, strict=True)))
    recurrent = tuple(
        0.55 * conv_value + 0.35 * float(parent) + 0.015 * (int(depth) + 1) + 0.001 * ((int(position) + index) % 7)
        for index, (conv_value, parent) in enumerate(zip(conv, parent_state, strict=True))
    )
    out = tuple(0.8 * conv_value + 0.2 * recurrent_value for conv_value, recurrent_value in zip(conv, recurrent, strict=True))
    return out, conv, recurrent


def _full_attn_stage(
    hidden: Sequence[float],
    parent_key: Sequence[float],
    parent_value: Sequence[float],
    position: int,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    key = tuple(0.6 * float(value) + 0.25 * float(parent) + 0.002 * (int(position) % 11) for value, parent in zip(hidden, parent_key, strict=True))
    value = tuple(
        0.5 * float(hidden_value) + 0.3 * float(parent) + 0.01 * ((index + int(position)) % 3)
        for index, (hidden_value, parent) in enumerate(zip(reversed(tuple(hidden)), parent_value, strict=True))
    )
    out = tuple(float(h) + 0.125 * k - 0.0625 * v for h, k, v in zip(hidden, key, value, strict=True))
    return out, key, value


def _moe_stage(hidden: Sequence[float], token: int, row: int) -> tuple[tuple[float, ...], int]:
    expert = (int(token) + int(row)) % 4
    scale = 0.035 * (expert + 1)
    return tuple(float(value) + scale * _silu(float(value)) for value in hidden), expert


def _lm_head_logits(hidden: Sequence[float], vocab_size: int) -> tuple[float, ...]:
    logits = []
    for vocab in range(vocab_size):
        acc = 0.0
        for index, value in enumerate(hidden):
            weight = (((vocab + 1) * (index + 3)) % 17 - 8) / 32.0
            acc += float(value) * weight
        logits.append(acc + ((vocab % 5) - 2) * 0.01)
    return tuple(logits)


def _silu(value: float) -> float:
    return value / (1.0 + exp(-value))


def _matrix(rows: Sequence[Sequence[float]]) -> Matrix:
    return tuple(tuple(float(value) for value in row) for row in rows)


def _validate_matrix(rows: Matrix, name: str) -> None:
    if not rows:
        raise ValueError(f"{name} must contain at least one row")
    width = len(rows[0])
    if width <= 0:
        raise ValueError(f"{name} rows must be non-empty")
    for row in rows:
        if len(row) != width:
            raise ValueError(f"{name} rows must have a consistent width")


__all__ = [
    "TargetVerifyLadderMismatch",
    "TargetVerifyLadderStageComparison",
    "TargetVerifyLayerLadderResult",
    "TargetVerifyStageSnapshot",
    "TargetVerifyStateRows",
    "compare_target_verify_ladder",
    "synthetic_chain_target_verify_ladder",
    "synthetic_chain_target_verify_snapshots",
]
