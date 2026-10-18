"""Frozen G3 long-context task generator/evaluator for the DMS selector campaign.

This module owns the G3 task-quality machinery defined in
``docs/campaigns/DMS-SELECTOR-IMPROVEMENT.md`` Section 4 (gate table) and the
free-running diagnostic reporting required by the same section.  It is
deliberately pure: the standard library only, no model, GPU, capture, sidecar,
or tokenizer dependency.  Tokenization is injected as a callable so unit tests
and the CLI share one frozen generation plan.

Frozen structure (do not retune to rescue a failed candidate):

- **24 cases per length** — three task families (exact retrieval, two-hop
  key/value lookup, variable-state tracking) x four language categories
  (code, general English, Japanese, mixed Japanese/English) x two placements
  (early evictable history, recent protected region).
- **Exact target token length** per case, with recorded dependency and query
  token positions.  Early dependencies lie outside the protected window at
  the prefill cut (``T-1-position > 256``); recent dependencies lie within
  128 tokens of the cut.
- **Fixed unique answer nonces** derived from the suite seed; the answer token
  sequence must not occur in the filler.  Filler is sliced from sequences of
  the one assigned split only (source-disjoint by construction).  Cases that
  reuse filler source material are reported as correlated, not independent.
- **G3** — candidate correct count must be at least the dense-teacher count
  overall and independently per task family, language category, placement,
  and tested length.  The frozen baseline DMS arm is reported separately and
  never gates.  All dense failures are retained; hard cases are never deleted.
- **Free-running diagnostic** — first divergence, comparable-prefix length and
  fixed-length token match rate on ordinary sealed prompts.  These metrics are
  non-binding diagnostics and never substitute for G3 tasks.

Final answers live only in the task manifest and the evaluator; answer
keys are never supplied as a separate selector feature or control signal.
Each case's answer occurs exactly once, inside its dependency statement,
where the task semantics require it; the selector otherwise sees only the
ordinary prompt token representation, never a leaked answer key.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Sequence as SequenceType

from hipengine.benchmark.dms_campaign import CATEGORIES

TASK_MANIFEST_KIND = "hipengine_qwen38_dms_campaign_task_manifest"
TASK_RUN_KIND = "hipengine_qwen38_dms_campaign_task_run"
FREE_RUNNING_RUN_KIND = "hipengine_qwen38_dms_campaign_free_running_run"

TASK_FAMILIES = ("retrieval", "two_hop", "variable_state")
PLACEMENTS = ("early", "recent")
SUITES = ("development", "qualification", "final")
SUPPORTED_TARGET_TOKENS = (32768, 131072)
MIN_TARGET_TOKENS = 1024

# Early dependencies must be outside the protected window W256 at the prefill
# cut: (T-1 - dep_end) > EARLY_MIN_DISTANCE.
EARLY_MIN_DISTANCE = 256
# Recent dependencies must lie within this many tokens of the cut.
RECENT_WINDOW_TOKENS = 128

RUN_ARMS = ("dense", "no_evict", "sidecar")
DMS_ARMS = ("no_evict", "sidecar")
DENSE_ARM = "dense"

DEFAULT_MAX_ANSWER_TOKENS = 32
DEFAULT_FREE_RUNNING_TOKENS = 256

_FILLER_AVOID_ATTEMPTS = 128

# ---------------------------------------------------------------------------
# Case identity and answer nonces


def case_coordinates() -> tuple[tuple[str, str, str], ...]:
    """Return the frozen 24-case coordinate matrix in canonical order."""
    return tuple(
        (family, category, placement)
        for family in TASK_FAMILIES
        for category in CATEGORIES
        for placement in PLACEMENTS
    )


def case_id(suite: str, target_tokens: int, family: str, category: str, placement: str) -> str:
    return f"{suite}-{int(target_tokens)}-{family}-{category}-{placement}"


def _hex_digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def case_seed(seed: int, suite: str, target_tokens: int, family: str, category: str, placement: str) -> int:
    return int(_hex_digest(seed, suite, target_tokens, family, category, placement, "seed"), 16)


def _nonce(seed_int: int, label: str) -> str:
    """Deterministic unique-looking nonce like ``K7QF-2M9X``."""
    digest = hashlib.sha256(f"{seed_int}|{label}".encode("utf-8")).hexdigest()
    return f"{digest[:4].upper()}-{digest[4:8].upper()}"


def case_entities(family: str, seed_int: int) -> dict[str, str]:
    """Deterministic per-case entity/answer nonces for one task family."""
    if family == "retrieval":
        return {
            "entity": _nonce(seed_int, "entity"),
            "answer": _nonce(seed_int, "answer"),
        }
    if family == "two_hop":
        return {
            "node": _nonce(seed_int, "node"),
            "gateway": _nonce(seed_int, "gateway"),
            "answer": _nonce(seed_int, "answer"),
        }
    if family == "variable_state":
        return {
            "state_1": _nonce(seed_int, "state1"),
            "state_2": _nonce(seed_int, "state2"),
            "answer": _nonce(seed_int, "answer"),
        }
    raise ValueError(f"unknown task family {family!r}; expected one of {TASK_FAMILIES}")


# ---------------------------------------------------------------------------
# Frozen instruction templates (semantically explicit, deterministic)


_TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "retrieval": {
        "general_en": (
            "MEMO: The access key for relay station {entity} is {answer}.",
            "What is the access key for relay station {entity}? "
            "Answer with only the exact key and no other text.",
        ),
        "general_ja": (
            "メモ:中継局{entity}のアクセスキーは{answer}である。",
            "中継局{entity}のアクセスキーは何か。キーのみをそのまま答え、他の文字は書くな。",
        ),
        "mixed_ja_en": (
            "MEMO: 中継局 relay station {entity} のアクセスキーは {answer} です。",
            "relay station {entity} のアクセスキーは何ですか。Answer with only the exact key, "
            "他のテキストは書くな。",
        ),
        "code": (
            'const ACCESS_KEY_RELAY_{entity} = "{answer}";',
            "// Print the value of ACCESS_KEY_RELAY_{entity}. "
            "Output only the exact string value and nothing else.",
        ),
    },
    "two_hop": {
        "general_en": (
            "MEMO: The gateway for node {node} is {gateway}. "
            "The access key for gateway {gateway} is {answer}.",
            "What is the access key for the gateway of node {node}? "
            "Answer with only the exact key and no other text.",
        ),
        "general_ja": (
            "メモ:ノード{node}のゲートウェイは{gateway}である。"
            "ゲートウェイ{gateway}のアクセスキーは{answer}である。",
            "ノード{node}のゲートウェイのアクセスキーは何か。キーのみをそのまま答えよ。",
        ),
        "mixed_ja_en": (
            "MEMO: ノード node {node} のゲートウェイは {gateway} です。"
            "ゲートウェイ gateway {gateway} のアクセスキーは {answer} です。",
            "node {node} のゲートウェイのアクセスキー is what? "
            "Answer with only the exact key, 他のテキストは書くな。",
        ),
        "code": (
            'const GATEWAY_OF_NODE_{node} = "{gateway}";\n'
            'const KEY_OF_GATEWAY_{gateway} = "{answer}";',
            "// Print the key of the gateway of node {node}. "
            "Output only the exact key value and nothing else.",
        ),
    },
    "variable_state": {
        "general_en": (
            "LOG: At checkpoint 3 the cargo state is {state_1}. "
            "At checkpoint 7 the cargo state is {state_2}. "
            "At checkpoint 11 the cargo state is {answer}.",
            "What is the cargo state at the final checkpoint? "
            "Answer with only the exact state value and no other text.",
        ),
        "general_ja": (
            "記録:チェックポイント3の貨物状態は{state_1}である。"
            "チェックポイント7の貨物状態は{state_2}である。"
            "チェックポイント11の貨物状態は{answer}である。",
            "最終チェックポイントの貨物状態は何か。状態の値のみをそのまま答えよ。",
        ),
        "mixed_ja_en": (
            "LOG: チェックポイント3の cargo state は {state_1} です。"
            "チェックポイント7の cargo state は {state_2} です。"
            "チェックポイント11の cargo state は {answer} です。",
            "最終チェックポイントの cargo state は何ですか。"
            "Answer with only the exact state value, 他のテキストは書くな。",
        ),
        "code": (
            'state = "{state_1}";  // checkpoint 3\n'
            'state = "{state_2}";  // checkpoint 7\n'
            'state = "{answer}";  // checkpoint 11',
            "// Print the cargo state at the final checkpoint. "
            "Output only the exact state value and nothing else.",
        ),
    },
}


def case_texts(family: str, category: str, entities: dict[str, str]) -> tuple[str, str]:
    """Return the frozen (dependency_text, query_text) pair for one case."""
    if family not in _TEMPLATES:
        raise ValueError(f"unknown task family {family!r}")
    if category not in _TEMPLATES[family]:
        raise ValueError(f"unknown category {category!r}")
    dependency_format, query_format = _TEMPLATES[family][category]
    return dependency_format.format(**entities), query_format.format(**entities)


# ---------------------------------------------------------------------------
# Answer parsing


def parse_answer(generated_text: str, answer_text: str) -> dict[str, Any]:
    """Parse one generation for the exact case answer nonce.

    G3 is exact-answer paired scoring; top-1 agreement never substitutes for
    this verdict.
    """
    text = str(generated_text)
    normalized = text.strip()
    answer = str(answer_text)
    contains_index = text.find(answer)
    correct = normalized == answer
    return {
        "answer_found": contains_index >= 0,
        "answer_index": contains_index if contains_index >= 0 else None,
        "normalized_answer": normalized,
        "verdict": "correct" if correct else "incorrect",
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Filler selection (source-disjoint by construction: one assigned split)


def _contains_sublist(haystack: SequenceType[int], needle: SequenceType[int]) -> bool:
    if not needle:
        return False
    first = int(needle[0])
    limit = len(haystack) - len(needle)
    needle_tuple = tuple(int(t) for t in needle)
    for start in range(limit + 1):
        if int(haystack[start]) != first:
            continue
        if tuple(int(t) for t in haystack[start : start + len(needle)]) == needle_tuple:
            return True
    return False


def _category_stream(pool: SequenceType[dict[str, Any]], category: str):
    entries = [entry for entry in pool if str(entry["category"]) == category]
    if not entries:
        raise ValueError(f"no filler sources for category {category!r} in the assigned split")
    tokens: list[int] = []
    boundaries: list[tuple[dict[str, Any], int, int]] = []
    for entry in entries:
        entry_tokens = [int(t) for t in entry["token_ids"]]
        if not entry_tokens:
            raise ValueError(f"filler source {entry['sequence_id']!r} has no tokens")
        boundaries.append((entry, len(tokens), len(entry_tokens)))
        tokens.extend(entry_tokens)
    return entries, tokens, boundaries


def select_filler(
    tokens: SequenceType[int],
    boundaries: SequenceType[tuple[dict[str, Any], int, int]],
    *,
    length: int,
    avoid_ids: SequenceType[int],
    rng: random.Random,
    max_attempts: int = _FILLER_AVOID_ATTEMPTS,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Pick a deterministic filler window of exactly ``length`` tokens.

    The answer token sequence must not occur in the chosen window; the start
    offset advances deterministically until the window is clean, and the
    function refuses rather than emitting a contaminated case.
    """
    if length < 0:
        raise ValueError("filler length must be non-negative")
    if len(tokens) < length:
        raise ValueError(
            f"filler pool has {len(tokens)} tokens, fewer than the required {length}"
        )
    if length == 0:
        return [], []
    span = len(tokens) - length
    start = rng.randrange(span + 1)
    window: list[int] = []
    for _ in range(max(1, max_attempts)):
        window = [int(t) for t in tokens[start : start + length]]
        if not _contains_sublist(window, avoid_ids):
            break
        start = start + 1 if start < span else 0
    else:
        raise ValueError(
            "cannot select filler that avoids the answer token sequence; "
            "the assigned split's sources repeat the answer nonce"
        )
    end = start + length
    slices: list[dict[str, Any]] = []
    for entry, offset, entry_length in boundaries:
        slice_start = max(start, offset)
        slice_end = min(end, offset + entry_length)
        if slice_start < slice_end:
            slices.append(
                {
                    "sequence_id": str(entry["sequence_id"]),
                    "source_id": str(entry.get("source_id", "")),
                    "normalized_text_sha256": str(entry.get("normalized_text_sha256", "")),
                    "start": slice_start - offset,
                    "end": slice_end - offset - 1,
                    "length": slice_end - slice_start,
                }
            )
    if not slices:
        raise ValueError("filler window recorded no source slices")
    return window, slices


# ---------------------------------------------------------------------------
# Case construction and validation


def build_case(
    *,
    index: int,
    suite: str,
    target_tokens: int,
    family: str,
    category: str,
    placement: str,
    seed: int,
    pool: SequenceType[dict[str, Any]],
    tokenize: Callable[[str], SequenceType[int]],
) -> dict[str, Any]:
    """Build one frozen G3 case with an exact target token length."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected one of {SUITES}")
    if family not in TASK_FAMILIES:
        raise ValueError(f"unknown task family {family!r}")
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement {placement!r}")
    target_tokens = int(target_tokens)
    if target_tokens < MIN_TARGET_TOKENS:
        raise ValueError(
            f"target_tokens {target_tokens} below the supported minimum {MIN_TARGET_TOKENS}"
        )

    seed_int = case_seed(seed, suite, target_tokens, family, category, placement)
    entities = case_entities(family, seed_int)
    dependency_text, query_text = case_texts(family, category, entities)
    if entities["answer"] in query_text:
        raise ValueError("query template leaks the answer nonce")
    dependency_ids = [int(t) for t in tokenize(dependency_text)]
    query_ids = [int(t) for t in tokenize(query_text)]
    answer_ids = [int(t) for t in tokenize(entities["answer"])]

    if placement == "recent" and len(dependency_ids) + len(query_ids) > RECENT_WINDOW_TOKENS:
        raise ValueError(
            "recent placement requires dependency + query to fit inside "
            f"{RECENT_WINDOW_TOKENS} tokens; got {len(dependency_ids)} + {len(query_ids)}"
        )
    filler_length = target_tokens - len(dependency_ids) - len(query_ids)
    if filler_length <= 0:
        raise ValueError(
            f"target_tokens {target_tokens} too small for dependency+query blocks"
        )

    _, tokens, boundaries = _category_stream(pool, category)
    rng = random.Random(seed_int)
    window, slices = select_filler(
        tokens, boundaries, length=filler_length, avoid_ids=answer_ids, rng=rng
    )
    filler_split = target_tokens // 4 if placement == "early" else filler_length
    filler_first = window[:filler_split]
    filler_second = window[filler_split:]
    token_ids = filler_first + dependency_ids + filler_second + query_ids

    dep_start = filler_split
    dep_end = filler_split + len(dependency_ids) - 1
    query_start = target_tokens - len(query_ids)
    query_end = target_tokens - 1
    distance_from_cut = target_tokens - 1 - dep_end
    if placement == "early":
        placement_passed = distance_from_cut > EARLY_MIN_DISTANCE
    else:
        placement_passed = (target_tokens - 1 - dep_start) < RECENT_WINDOW_TOKENS

    case = {
        "case_id": case_id(suite, target_tokens, family, category, placement),
        "case_index": int(index),
        "suite": suite,
        "family": family,
        "category": category,
        "placement": placement,
        "target_tokens": target_tokens,
        "seed": seed_int,
        "entities": entities,
        "answer_text": entities["answer"],
        "answer_token_ids": answer_ids,
        "answer_token_ids_sha256": _hex_digest("answer-tokens", target_tokens, family, category, placement, answer_ids),
        "dependency_text": dependency_text,
        "dependency_token_ids": dependency_ids,
        "query_text": query_text,
        "query_token_ids": query_ids,
        "dependency_position": {"start": dep_start, "end": dep_end},
        "query_position": {"start": query_start, "end": query_end},
        "token_ids": token_ids,
        "token_ids_sha256": _hex_digest("case-tokens", token_ids),
        "filler_token_ids": window,
        "filler_token_ids_sha256": _hex_digest("filler-tokens", window),
        "filler_slices": slices,
        "answer_absent_from_filler": not _contains_sublist(window, answer_ids),
        "placement_check": {
            "placement": placement,
            "dep_start": dep_start,
            "dep_end": dep_end,
            "distance_from_cut": distance_from_cut,
            "early_requirement": f"distance_from_cut > {EARLY_MIN_DISTANCE} (outside W256 at the prefill cut)",
            "recent_requirement": f"dependency block within {RECENT_WINDOW_TOKENS} tokens of the cut",
            "passed": placement_passed,
        },
    }
    problems = validate_case(case)
    if problems:
        raise ValueError("; ".join(problems))
    return case


def validate_case(case: dict[str, Any]) -> list[str]:
    """Validate every frozen case invariant; return a list of problems."""
    problems: list[str] = []
    token_ids = [int(t) for t in case["token_ids"]]
    target_tokens = int(case["target_tokens"])
    if len(token_ids) != target_tokens:
        problems.append(f"token length {len(token_ids)} != target {target_tokens}")
    dep = case["dependency_position"]
    query = case["query_position"]
    dep_ids = [int(t) for t in case["dependency_token_ids"]]
    query_ids = [int(t) for t in case["query_token_ids"]]
    if token_ids[dep["start"] : dep["end"] + 1] != dep_ids:
        problems.append("recorded dependency position does not match the prompt tokens")
    if token_ids[query["start"] : query["end"] + 1] != query_ids:
        problems.append("recorded query position does not match the prompt tokens")
    if query["end"] != target_tokens - 1 or query["start"] != target_tokens - len(query_ids):
        problems.append("query block is not at the end of the prompt")
    if not (0 <= dep["start"] <= dep["end"] < query["start"]):
        problems.append("dependency block overlaps or follows the query block")
    distance = target_tokens - 1 - dep["end"]
    if case["placement"] == "early" and not distance > EARLY_MIN_DISTANCE:
        problems.append(f"early dependency only {distance} tokens from the cut (need > {EARLY_MIN_DISTANCE})")
    if case["placement"] == "recent":
        if not (target_tokens - 1 - dep["start"]) < RECENT_WINDOW_TOKENS:
            problems.append(
                f"recent dependency starts {(target_tokens - 1 - dep['start'])} tokens from the cut "
                f"(need < {RECENT_WINDOW_TOKENS})"
            )
    if not case["answer_absent_from_filler"]:
        problems.append("answer token sequence occurs in the filler")
    if _contains_sublist(query_ids, case["answer_token_ids"]):
        problems.append("answer token sequence occurs in the query block")
    if case["answer_text"] in case["query_text"]:
        problems.append("answer nonce leaks into the query text")
    return problems


def build_task_manifest(
    *,
    pool: SequenceType[dict[str, Any]],
    suite: str,
    target_tokens: int,
    seed: int,
    tokenize: Callable[[str], SequenceType[int]],
    data_manifest: dict[str, Any],
    model: dict[str, Any],
    schema_version: int = 1,
) -> dict[str, Any]:
    """Build the immutable 24-case G3 task manifest for one suite and length."""
    coordinates = case_coordinates()
    if len(coordinates) != 24:
        raise AssertionError("frozen G3 matrix must contain exactly 24 cases")
    cases = [
        build_case(
            index=index,
            suite=suite,
            target_tokens=target_tokens,
            family=family,
            category=category,
            placement=placement,
            seed=seed,
            pool=pool,
            tokenize=tokenize,
        )
        for index, (family, category, placement) in enumerate(coordinates)
    ]
    answers = [case["answer_text"] for case in cases]
    if len(set(answers)) != len(answers):
        raise ValueError("answer nonces are not unique across the 24 cases")
    manifest = {
        "schema_version": int(schema_version),
        "kind": TASK_MANIFEST_KIND,
        "immutable": True,
        "suite": suite,
        "seed": int(seed),
        "target_tokens": int(target_tokens),
        "task_families": list(TASK_FAMILIES),
        "categories": list(CATEGORIES),
        "placements": list(PLACEMENTS),
        "case_count": len(cases),
        "data_manifest": dict(data_manifest),
        "model": dict(model),
        "cases": cases,
        "filler_correlations": filler_correlations(cases),
        "answer_policy": {
            "parser": "generated text stripped of surrounding whitespace must equal the answer nonce",
            "answers_held_by": "evaluator task manifest only",
            "no_separate_selector_answer_signal": True,
            "top1_is_not_task_scoring": True,
        },
        "frozen_gate": "g3",
    }
    return manifest


# ---------------------------------------------------------------------------
# Correlated filler prefixes are not independent prompts


def _common_prefix_length(a: SequenceType[int], b: SequenceType[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def filler_correlations(cases: SequenceType[dict[str, Any]]) -> dict[str, Any]:
    """Group cases whose filler reuses the same source material.

    Cases that share filler sources or long repeated filler prefixes are
    correlated samples, not independent prompts; the manifest reports the
    group structure instead of counting every case as independent.
    """
    count = len(cases)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    source_sets: list[set[str]] = []
    for case in cases:
        source_sets.append(
            {str(s["sequence_id"]) for s in case["filler_slices"]}
        )
    for i in range(count):
        for j in range(i + 1, count):
            if source_sets[i] & source_sets[j]:
                union(i, j)
                continue
            filler_i = [int(t) for t in cases[i]["filler_token_ids"]]
            filler_j = [int(t) for t in cases[j]["filler_token_ids"]]
            shared = _common_prefix_length(filler_i, filler_j)
            if shared >= 64 and shared >= min(len(filler_i), len(filler_j)) // 2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    correlated = []
    for members in groups.values():
        if len(members) < 2:
            continue
        shared_sources = set.intersection(*(source_sets[m] for m in members))
        prefix = cases[members[0]]["filler_token_ids"]
        shared_prefix = len(prefix)
        for member in members[1:]:
            shared_prefix = _common_prefix_length(prefix, cases[member]["filler_token_ids"])
            prefix = cases[members[0]]["filler_token_ids"][:shared_prefix]
        correlated.append(
            {
                "members": sorted(str(cases[m]["case_id"]) for m in members),
                "shared_source_ids": sorted(shared_sources),
                "shared_prefix_tokens": int(shared_prefix),
            }
        )
    return {
        "case_count": count,
        "independent_case_groups": len(groups),
        "correlated_groups": correlated,
        "note": (
            "cases reusing filler source material or long repeated filler "
            "prefixes are correlated samples, not independent prompts"
        ),
    }


# ---------------------------------------------------------------------------
# G3 comparison


def _scope_bucket_keys(case: dict[str, Any]) -> dict[str, str]:
    return {
        "family": str(case["family"]),
        "category": str(case["category"]),
        "placement": str(case["placement"]),
        "target_tokens": str(int(case["target_tokens"])),
    }


def result_scope_counts(result: dict[str, Any]) -> dict[str, Any]:
    """Count correct cases overall and per G3 scope of one run result."""
    cases = list(result["cases"])
    scopes: dict[str, dict[str, dict[str, int]]] = {
        "overall": {"all": {"correct": 0, "total": 0}},
        "family": {},
        "category": {},
        "placement": {},
        "target_tokens": {},
    }
    for case in cases:
        correct = 1 if bool(case["correct"]) else 0

        def bump(scope: str, key: str) -> None:
            bucket = scopes[scope].setdefault(key, {"correct": 0, "total": 0})
            bucket["correct"] += correct
            bucket["total"] += 1

        bump("overall", "all")
        for scope, key in _scope_bucket_keys(case).items():
            bump(scope, key)
    return scopes


def compare_g3(
    dense_result: dict[str, Any],
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen G3 gate to paired task-run result files.

    G3 — the candidate's correct count must be at least the dense teacher's
    count overall and independently per task family, language category,
    placement, and tested length.  The frozen baseline DMS arm is reported
    separately and never gates.  Identity or hash mismatches reject the
    comparison outright.
    """
    errors: list[str] = []
    for label, result, arm, require_metadata in (
        ("dense", dense_result, DENSE_ARM, False),
        ("baseline", baseline_result, "sidecar", True),
        ("candidate", candidate_result, "sidecar", True),
    ):
        if str(result.get("kind")) != TASK_RUN_KIND:
            errors.append(f"{label} result kind {result.get('kind')!r} is not a task run")
        if str(result.get("arm")) != arm:
            errors.append(f"{label} result arm {result.get('arm')!r} != required {arm!r}")
        if require_metadata and not bool((result.get("metadata") or {}).get("sha256")):
            errors.append(f"{label} result is a DMS arm but carries no metadata hash")
        if not require_metadata and bool((result.get("metadata") or {}).get("sha256")):
            errors.append(f"{label} arm must not carry metadata")
    dense_manifest = (dense_result.get("task_manifest") or {}).get("sha256")
    for label, result in (("baseline", baseline_result), ("candidate", candidate_result)):
        if (result.get("task_manifest") or {}).get("sha256") != dense_manifest:
            errors.append(f"{label} result task-manifest hash differs from the dense result")
    dense_model = (dense_result.get("model") or {}).get("sha256")
    for label, result in (("baseline", baseline_result), ("candidate", candidate_result)):
        if (result.get("model") or {}).get("sha256") != dense_model:
            errors.append(f"{label} result model hash differs from the dense result")
    dense_evaluator = dense_result.get("evaluator") or {}
    for label, result in (("baseline", baseline_result), ("candidate", candidate_result)):
        evaluator = result.get("evaluator") or {}
        for field in ("library_sha256", "script_sha256"):
            if not dense_evaluator.get(field) or evaluator.get(field) != dense_evaluator.get(field):
                errors.append(f"{label} result evaluator {field} differs from the dense result")

    dense_rows = list(dense_result.get("cases") or [])
    baseline_rows = list(baseline_result.get("cases") or [])
    candidate_rows = list(candidate_result.get("cases") or [])
    for label, rows in (
        ("dense", dense_rows),
        ("baseline", baseline_rows),
        ("candidate", candidate_rows),
    ):
        ids = [str(case.get("case_id", "")) for case in rows]
        if not ids or any(not case_id for case_id in ids):
            errors.append(f"{label} result carries no cases or an empty case ID")
        if len(ids) != len(set(ids)):
            errors.append(f"{label} result contains duplicate case IDs")

    dense_cases = {str(c["case_id"]): c for c in dense_rows}
    baseline_cases = {str(c["case_id"]): c for c in baseline_rows}
    candidate_cases = {str(c["case_id"]): c for c in candidate_rows}
    for label, cases in (("baseline", baseline_cases), ("candidate", candidate_cases)):
        if set(cases) != set(dense_cases):
            errors.append(f"{label} result case set differs from the dense result")
            continue
        for case_id, dense_case in dense_cases.items():
            case = cases[case_id]
            for field in (
                "family",
                "category",
                "placement",
                "target_tokens",
                "answer_token_ids_sha256",
                "prompt_token_ids_sha256",
            ):
                if case.get(field) != dense_case.get(field):
                    errors.append(
                        f"{label} result case {case_id} field {field} differs from dense"
                    )

    if errors:
        return {
            "gate": "g3",
            "passed": False,
            "status": "rejected_identity",
            "errors": errors,
            "note": "identity/hash mismatch rejects the comparison before any scoring",
        }

    dense_counts = result_scope_counts(dense_result)
    candidate_counts = result_scope_counts(candidate_result)
    baseline_counts = result_scope_counts(baseline_result)

    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for scope in ("overall", "family", "category", "placement", "target_tokens"):
        keys = sorted(set(dense_counts[scope]) | set(candidate_counts[scope]))
        for key in keys:
            dense_correct = dense_counts[scope].get(key, {"correct": 0})["correct"]
            candidate_correct = candidate_counts[scope].get(key, {"correct": 0})["correct"]
            passed = candidate_correct >= dense_correct
            entry = {
                "scope": scope,
                "key": key,
                "dense_correct": dense_correct,
                "candidate_correct": candidate_correct,
                "passed": passed,
            }
            checks.append(entry)
            if not passed:
                failures.append(entry)

    dense_failures = [
        {
            "case_id": case_id,
            "family": case["family"],
            "category": case["category"],
            "placement": case["placement"],
            "target_tokens": case["target_tokens"],
            "parser_verdict": case.get("parser", {}).get("verdict"),
            "generated_text": case.get("generated_text"),
        }
        for case_id, case in sorted(dense_cases.items())
        if not bool(case["correct"])
    ]
    paired_deltas = [
        {
            "case_id": case_id,
            "family": dense_cases[case_id]["family"],
            "category": dense_cases[case_id]["category"],
            "placement": dense_cases[case_id]["placement"],
            "target_tokens": dense_cases[case_id]["target_tokens"],
            "dense_correct": bool(dense_cases[case_id]["correct"]),
            "baseline_correct": bool(baseline_cases[case_id]["correct"]),
            "candidate_correct": bool(candidate_cases[case_id]["correct"]),
            "delta": int(bool(candidate_cases[case_id]["correct"]))
            - int(bool(dense_cases[case_id]["correct"])),
        }
        for case_id in sorted(dense_cases)
    ]
    tested_lengths = sorted({int(case["target_tokens"]) for case in dense_cases.values()})
    return {
        "gate": "g3",
        "passed": not failures,
        "status": "passed" if not failures else "failed_quality",
        "errors": [],
        "rule": (
            "candidate correct count >= dense teacher count overall and per task family, "
            "language category, placement, and tested length"
        ),
        "tested_lengths": tested_lengths,
        "dense_counts": dense_counts,
        "candidate_counts": candidate_counts,
        "baseline_counts": baseline_counts,
        "baseline_note": "frozen baseline DMS arm reported separately; never gates G3",
        "scope_checks": checks,
        "failures": failures,
        "dense_failures": dense_failures,
        "paired_case_deltas": paired_deltas,
    }


# ---------------------------------------------------------------------------
# Free-running diagnostic (non-binding)


def free_running_metrics(
    reference_tokens: SequenceType[int],
    candidate_tokens: SequenceType[int],
    *,
    fixed_length: int = DEFAULT_FREE_RUNNING_TOKENS,
) -> dict[str, Any]:
    """First divergence / comparable prefix / fixed-length token match.

    These are non-binding diagnostics on ordinary sealed prompts; they never
    substitute for G3 paired task scoring.
    """
    reference = [int(t) for t in reference_tokens]
    candidate = [int(t) for t in candidate_tokens]
    prefix = _common_prefix_length(reference, candidate)
    if prefix == len(reference) and prefix == len(candidate):
        first_divergence = None
    else:
        first_divergence = prefix
    fixed_length = max(1, int(fixed_length))
    matches = sum(
        1
        for index in range(fixed_length)
        if index < len(reference) and index < len(candidate) and reference[index] == candidate[index]
    )
    return {
        "first_divergence": first_divergence,
        "comparable_prefix_tokens": int(prefix),
        "fixed_length": fixed_length,
        "fixed_length_matches": matches,
        "fixed_length_token_match_rate": matches / fixed_length,
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
        "non_binding": True,
        "note": "free-running diagnostics never substitute for G3 task scoring",
    }
