"""Target verify group row padding to admitted rowtile shapes.

Physical C2 groups whose total verify rows fall below the backend's admitted
production rowtile rows (gfx1100: rows6) currently fall back to the shared-B
256-row padded tile at ~5.1x cycle cost. Group padding appends inactive
candidate rows owned by the last request so the physical launch rides the
qualified rowtile. These contracts pin the padding semantics: padded graphs
stay valid, and the padded accept path is bit-identical to the unpadded one.
"""

from __future__ import annotations

import pytest

from hipengine.speculative.frontier import (
    CandidateGraph,
    pad_candidate_graph_rows,
)
from hipengine.speculative.interfaces import TargetVerifyBatch


def _chain_graph(
    request_ids: tuple[int, ...],
    counts: tuple[int, ...],
    *,
    root_position: int = 40,
) -> CandidateGraph:
    tokens: list[int] = []
    owners: list[int] = []
    parents: list[int] = []
    depths: list[int] = []
    offsets = [0]
    offset = 0
    for request_id, count in zip(request_ids, counts, strict=True):
        for depth in range(1, count + 1):
            tokens.append(request_id * 100 + depth)
            owners.append(request_id)
            parents.append(-1 if depth == 1 else offset + depth - 2)
            depths.append(depth)
        offset += count
        offsets.append(offset)
    return CandidateGraph(
        provider_key="test-provider",
        method_key="mtp2",
        policy_fingerprint="test-policy:v1",
        cycle_id=7,
        transaction_id=7,
        request_ids=request_ids,
        resident_slots=tuple(range(len(request_ids))),
        root_positions=tuple(root_position + index for index in range(len(request_ids))),
        row_offsets=tuple(offsets),
        row_to_request=tuple(owners),
        parent_candidate_rows=tuple(parents),
        draft_depths=tuple(depths),
        active_mask=(True,) * len(tokens),
        candidate_tokens=tuple(tokens),
    )


def _target_batch(graph: CandidateGraph) -> TargetVerifyBatch:
    return TargetVerifyBatch.from_draft(
        graph.to_draft_batch(),
        root_tokens=tuple(1000 + index for index in range(len(graph.request_ids))),
        root_positions=graph.root_positions,
    )


def _accept_payload(batch: TargetVerifyBatch, top1: list[int]) -> tuple:
    accept = batch.accept_from_top1(
        tuple(top1),
        transaction_id=7,
        remaining_decode=(8,) * len(batch.request_ids),
    )
    return (
        accept.accepted_counts,
        accept.accepted_tokens,
        accept.selected_candidate_rows,
        accept.correction_or_bonus_tokens,
    )


class TestPadCandidateGraphRows:
    def test_pads_append_inactive_rows_owned_by_last_request(self) -> None:
        graph = _chain_graph((11, 12), (2, 2))
        padded = pad_candidate_graph_rows(
            graph,
            pad_rows=2,
            pad_token_id=0,
        )
        assert padded.candidate_rows == 6
        assert padded.row_offsets == (0, 2, 6)
        assert padded.row_to_request == (11, 11, 12, 12, 12, 12)
        assert padded.active_mask == (True, True, True, True, False, False)
        assert padded.candidate_tokens[:4] == graph.candidate_tokens
        assert padded.candidate_tokens[4:] == (0, 0)
        assert padded.root_positions == graph.root_positions
        assert padded.request_ids == graph.request_ids
        assert padded.draft_depths[:4] == graph.draft_depths

    def test_pad_accept_from_top1_is_identical_to_unpadded(self) -> None:
        graph = _chain_graph((11, 12), (2, 2))
        batch = _target_batch(graph)
        padded_graph = pad_candidate_graph_rows(graph, pad_rows=2, pad_token_id=0)
        padded_batch = _target_batch(padded_graph)
        assert padded_batch.rows == batch.rows + 2
        # Cover accept/reject/partial outcomes for both requests.
        for accepted_first, accepted_second in ((2, 2), (1, 0), (0, 1), (0, 0), (2, 0)):
            base_top1 = [0] * batch.rows
            for request_index, accepted in enumerate((accepted_first, accepted_second)):
                root_row = batch.root_rows[request_index]
                base_top1[root_row] = 5_000 + request_index
                cursor = root_row
                for depth in range(1, accepted + 1):
                    candidate_row = next(
                        row
                        for row in batch.candidate_rows
                        if batch.row_to_request[row] == batch.request_ids[request_index]
                        and batch.draft_depths[row] == depth
                    )
                    base_top1[cursor] = batch.tokens[candidate_row]
                    cursor = candidate_row
            padded_top1 = list(base_top1)
            # Pad rows carry arbitrary target top-1 values; they must never
            # influence the accept outcome.
            padded_top1 += [123_456, 654_321]
            assert _accept_payload(padded_batch, padded_top1) == _accept_payload(
                batch, base_top1
            ), (accepted_first, accepted_second)

    def test_pad_rejects_nonpositive_pad_rows(self) -> None:
        graph = _chain_graph((11,), (2,))
        with pytest.raises(ValueError):
            pad_candidate_graph_rows(graph, pad_rows=0, pad_token_id=0)
        with pytest.raises(ValueError):
            pad_candidate_graph_rows(graph, pad_rows=-1, pad_token_id=0)

    def test_pad_rejects_negative_pad_token(self) -> None:
        graph = _chain_graph((11,), (2,))
        with pytest.raises(ValueError):
            pad_candidate_graph_rows(graph, pad_rows=1, pad_token_id=-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


class TestPhysicalGroupPadRows:
    def test_pads_to_next_admitted_multiple(self) -> None:
        from hipengine.speculative.frontier import physical_group_pad_rows

        # counts=(6,), max 24: pad physical rows up to the next multiple of 6.
        assert physical_group_pad_rows((6,), 2, 2, 24) == 2   # 4 -> 6
        assert physical_group_pad_rows((6,), 2, 4, 24) == 0   # 6 exact
        assert physical_group_pad_rows((6,), 2, 6, 24) == 4   # 8 -> 12
        assert physical_group_pad_rows((6,), 3, 6, 24) == 3   # 9 -> 12
        assert physical_group_pad_rows((6,), 4, 8, 24) == 0   # 12 exact
        assert physical_group_pad_rows((6,), 5, 10, 24) == 3  # 15 -> 18
        assert physical_group_pad_rows((6,), 4, 12, 24) == 2  # 16 -> 18
        assert physical_group_pad_rows((6,), 4, 14, 24) == 0  # 18 exact

    def test_exact_count_bypasses_padding_without_changing_other_widths(self) -> None:
        from hipengine.speculative.frontier import physical_group_pad_rows

        assert physical_group_pad_rows((6,), 2, 6, 24) == 4
        assert physical_group_pad_rows(
            (6,), 2, 6, 24, exact_counts=(8,)
        ) == 0
        assert physical_group_pad_rows(
            (6,), 1, 3, 24, exact_counts=(8,)
        ) == 2
        assert physical_group_pad_rows(
            (6,), 3, 6, 24, exact_counts=(8,)
        ) == 3

    def test_no_pad_when_multiple_exceeds_capacity(self) -> None:
        from hipengine.speculative.frontier import physical_group_pad_rows

        # 20 rows -> next multiple 24 exceeds a 22-row cap: no pad.
        assert physical_group_pad_rows((6,), 7, 15, 22) == 0
        # Empty admitted counts disable padding entirely.
        assert physical_group_pad_rows((), 2, 2, 24) == 0
