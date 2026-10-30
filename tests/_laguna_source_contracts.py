"""Scoped Laguna source contracts that replace broad whole-file source pins.

RF-11 retired the last ``_POST_MERGE_SOURCE_SHA256`` re-baselines. Those pins
hashed shared files (``laguna_gguf_runner.py``, ``laguna_moe.py`` and the
campaign REDs themselves) that every Qwen3.8, execution-profile and row-batch
campaign edits, so an unrelated line move failed a contract it could not
affect. Each function here asserts the semantic owner a pin was standing in
for: which capability is read, where the dispatch boundary is, and who owns
the allocation or scope. Comments, formatting and unrelated additions are
invisible; a moved policy, dispatch boundary or ownership guard fails with a
message naming the file and the contract.

The functions take source *text* so a mutation test can prove both directions:
an unrelated edit is accepted and a protected edit is rejected. Live callers
pass ``Path.read_text()`` for the file the old hash covered.
"""

from __future__ import annotations

import ast
from typing import Any, Iterable

RUNNER = "hipengine/runtime/laguna_gguf_runner.py"
MOE = "hipengine/runtime/laguna_moe.py"
H8A_GPU_RED = "tests/test_gpu_laguna_h8a_resident_q5_global_cache.py"
H8B_UNIT_RED = "tests/test_unit_laguna_h8b_scoped_activation_pack_reuse.py"
H7U_LIVE_RED = "tests/test_live_laguna_moe_gpu.py"

H8A_CAPABILITY = "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE"
H8B_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE"
H7U_CAPABILITY = "LAGUNA_MOE_GROUP_COMPACT_MODE"
H8A_OWNER_CLASS = "LagunaQ5F32ResidentGlobalCache"
H8A_PLANE_COUNT = 24


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _parse(source: str, label: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise AssertionError(f"{label}: source is not parseable: {exc}") from exc


def _dotted(node: ast.AST) -> str:
    """Render a name/attribute/call path without formatting or comments."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        args = ",".join(_dotted(arg) for arg in node.args)
        kwargs = ",".join(
            f"{kw.arg}={_dotted(kw.value)}" for kw in node.keywords
        )
        inner = ",".join(part for part in (args, kwargs) if part)
        return f"{_dotted(node.func)}({inner})"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        items = ",".join(_dotted(item) for item in node.elts)
        return f"[{items}]"
    if isinstance(node, ast.Dict):
        return "{" + ",".join(
            f"{_dotted(key)}:{_dotted(value)}"
            for key, value in zip(node.keys, node.values)
        ) + "}"
    if isinstance(node, ast.Compare):
        left = _dotted(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            operator = {
                ast.Is: "is",
                ast.IsNot: "is not",
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.In: "in",
                ast.NotIn: "not in",
            }.get(type(op), type(op).__name__)
            left = f"{left} {operator} {_dotted(comparator)}"
        return left
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"-{_dotted(node.operand)}"
    if isinstance(node, ast.BinOp):
        operator = {
            ast.Mult: "*",
            ast.Add: "+",
            ast.Sub: "-",
            ast.Div: "/",
            ast.FloorDiv: "//",
        }.get(type(node.op), type(node.op).__name__)
        return f"({_dotted(node.left)} {operator} {_dotted(node.right)})"
    return ast.dump(node)


def _find_function(scope: ast.AST, name: str, label: str) -> Any:
    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{label}: missing function {name}()")


def _find_class(tree: ast.Module, name: str, label: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{label}: missing class {name}")


def _module_constants(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level `NAME = <value>` bindings, by name."""
    return _body_constants(tree.body)


def _class_constant(node: ast.ClassDef, name: str) -> ast.AST | None:
    return _body_constants(node.body).get(name)


def _body_constants(body: list[ast.stmt]) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for node in body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _const(node: ast.AST, label: str, what: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise AssertionError(
            f"{label}: {what} is not a literal: {_dotted(node)}"
        ) from exc


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and _dotted(call.func) == name
    ]


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _parameters(function: Any) -> dict[str, ast.arg]:
    args = function.args
    params: dict[str, ast.arg] = {}
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        params[arg.arg] = arg
    return params


def _defaults(function: Any) -> dict[str, ast.AST | None]:
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    defaulted = positional[len(positional) - len(args.defaults) :]
    defaults: dict[str, ast.AST | None] = {
        arg.arg: value for arg, value in zip(defaulted, args.defaults)
    }
    for arg, value in zip(args.kwonlyargs, args.kw_defaults):
        defaults[arg.arg] = value
    for arg in positional:
        defaults.setdefault(arg.arg, None)
    return defaults


def _assignments_to(node: ast.AST, target: str) -> list[ast.Assign]:
    found = []
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Assign):
            continue
        if any(_dotted(item) == target for item in candidate.targets):
            found.append(candidate)
    return found


def _parent_map(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _capability_reads(
    tree: ast.Module, capability: str, label: str
) -> list[ast.Call]:
    """Every `backend_package_capability(..., capability, ...)` call.

    Also rejects a capability literal used anywhere else, so a second policy
    path (env var, `getattr`, hard-coded `True`) cannot bypass the resolver.
    """
    reads = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and _dotted(call.func) == "backend_package_capability"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == capability
    ]
    parents = _parent_map(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and node.value == capability):
            continue
        parent = parents.get(node)
        if not (
            isinstance(parent, ast.Call)
            and _dotted(parent.func) == "backend_package_capability"
            and node in parent.args[1:2]
        ):
            raise AssertionError(
                f"{label}: {capability} is read outside its "
                "backend_package_capability resolver"
            )
    assert reads, f"{label}: {capability} has no backend_package_capability read"
    return reads


# --------------------------------------------------------------------------- #
# H8A: resident global-Q5 F32 cache ownership
# --------------------------------------------------------------------------- #
def assert_h8a_runner_owner(source: str) -> None:
    """H8A's protected semantics inside the shared Laguna runner.

    One capability resolves the policy, one allocation guard owns the 24
    planes, and only the session that allocated the cache frees it.
    """
    label = RUNNER
    tree = _parse(source, label)
    reads = _capability_reads(tree, H8A_CAPABILITY, label)
    resolver = _find_function(
        tree, "resolve_laguna_q5_f32_resident_global_cache", label
    )
    assert len(reads) == 2, (
        f"{label}: {H8A_CAPABILITY} must be read by the session resolver and the "
        f"allocation precondition only, found {len(reads)} reads"
    )
    assert any(call in list(ast.walk(resolver)) for call in reads), (
        f"{label}: resolve_laguna_q5_f32_resident_global_cache no longer reads "
        f"{H8A_CAPABILITY}"
    )
    for call in reads:
        default = _keyword(call, "default")
        positional = call.args[2] if len(call.args) > 2 else None
        default = default if default is not None else positional
        assert default is not None and _const(default, label, "capability default") is False, (
            f"{label}: {H8A_CAPABILITY} must default to False"
        )
    resolver_returns = [
        node.value
        for node in ast.walk(resolver)
        if isinstance(node, ast.Return) and node.value is not None
    ]
    assert resolver_returns and all(
        isinstance(value, ast.Call)
        and _dotted(value.func) == "bool"
        and len(value.args) == 1
        and value.args[0] in reads
        and not value.keywords
        for value in resolver_returns
    ), f"{label}: the H8A resolver must return the capability value as bool()"

    owner = _find_class(tree, H8A_OWNER_CLASS, label)
    allocate = _find_function(owner, "allocate", label)
    guards = [
        node
        for node in ast.walk(allocate)
        if isinstance(node, ast.Compare) and _dotted(node.left) in {
            "len(targets)",
            "len(raw_ptrs)",
        }
    ]
    assert {_dotted(node.left) for node in guards} == {"len(targets)", "len(raw_ptrs)"}, (
        f"{label}: allocate must guard both target and raw-pointer counts"
    )
    for node in guards:
        assert len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq), (
            f"{label}: allocate must reject non-24 plane counts"
        )
        assert [_const(item, label, "plane count") for item in node.comparators] == [
            H8A_PLANE_COUNT
        ], (
            f"{label}: {_dotted(node.left)} must stay pinned to "
            f"{H8A_PLANE_COUNT} resident planes"
        )
    assert _calls_named(allocate, "runtime.device_synchronize"), (
        f"{label}: allocate must synchronize before publishing the plane map"
    )
    assert _calls_named(allocate, "free"), (
        f"{label}: allocate must free partially built planes on failure"
    )
    raises_allocation_error = any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and _dotted(node.exc.func)
        == "LagunaQ5F32ResidentGlobalCacheAllocationError"
        for node in ast.walk(allocate)
    )
    assert raises_allocation_error, (
        f"{label}: allocate must fail closed with "
        "LagunaQ5F32ResidentGlobalCacheAllocationError"
    )
    layers = _class_constant(owner, "_LAYERS")
    assert layers is not None, (
        f"{label}: {H8A_OWNER_CLASS} lost its global-attention layer class"
    )
    assert _dotted(layers) == "tuple(range(0,48,4))", (
        f"{label}: {H8A_OWNER_CLASS} must own every fourth global-attention layer"
    )

    session = _find_class(tree, "LagunaGGUFResidentSession", label)
    init = _find_function(session, "__init__", label)
    policy_assignments = _assignments_to(
        init, "self.use_q5_f32_resident_global_cache"
    )
    assert len(policy_assignments) == 1, (
        f"{label}: the session must derive the H8A policy exactly once"
    )
    assert _dotted(policy_assignments[0].value) == (
        "resolve_laguna_q5_f32_resident_global_cache(self.backend)"
    ), (
        f"{label}: the session H8A policy must come from the capability resolver, "
        f"found {_dotted(policy_assignments[0].value)}"
    )
    allocation_guards = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and {
            _dotted(value) for value in node.test.values
        }
        >= {"self.use_q5_f32_resident_global_cache", "self._owns_weights"}
    ]
    assert len(allocation_guards) == 1, (
        f"{label}: resident-cache allocation must be gated by the H8A policy and "
        "self._owns_weights"
    )
    guard = allocation_guards[0]
    allocation_tries = [
        node
        for node in ast.walk(guard)
        if isinstance(node, ast.Try)
        and _calls_named(node, f"{H8A_OWNER_CLASS}.allocate")
    ]
    assert len(allocation_tries) == 1, (
        f"{label}: the H8A guard must own one allocate() transaction"
    )
    allocation_try = allocation_tries[0]
    assert any(
        isinstance(handler.type, ast.Name)
        and handler.type.id == "LagunaQ5F32ResidentGlobalCacheAllocationError"
        for handler in allocation_try.handlers
    ), f"{label}: allocation failure must be contained by the H8A guard"
    success_assignments = [
        node
        for statement in allocation_try.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Assign)
    ]
    assert any(
        any(
            _dotted(target) == "self._owns_q5_f32_resident_cache"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        for node in success_assignments
    ), f"{label}: only a successful allocation may take cache ownership"

    close = _find_function(session, "_close", label)
    release_guards = [
        node
        for node in ast.walk(close)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and {
            _dotted(value) for value in node.test.values
        }
        >= {"self.q5_f32_resident_cache is not None"}
    ]
    owned_release_guards = [
        node for node in release_guards
        if "self._owns_q5_f32_resident_cache"
        in {_dotted(value) for value in node.test.values}
    ]
    assert owned_release_guards, (
        f"{label}: _close must free the resident cache only when the session owns it"
    )
    releases = _calls_named(close, "cache.free")
    assert any(
        call in list(ast.walk(statement))
        for guard in owned_release_guards
        for statement in guard.body
        for call in releases
    ), f"{label}: the owned cache release must stay inside its ownership guard"


# --------------------------------------------------------------------------- #
# H8B: scoped activation-pack reuse
# --------------------------------------------------------------------------- #
def assert_h8b_shared_pack_scope(source: str) -> None:
    """H8B's protected semantics inside the shared Laguna MoE launcher.

    One scope owns pack reuse for the shared branch, it is parameter-driven,
    and both shared launch sites forward the caller's bit unchanged.
    """
    label = MOE
    tree = _parse(source, label)
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "hipengine.kernels.activation_pack"
        and any(alias.name == "activation_pack_reuse_scope" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imported, f"{label}: activation_pack_reuse_scope import moved"

    scope_calls = _calls_named(tree, "activation_pack_reuse_scope")
    assert len(scope_calls) == 1, (
        f"{label}: exactly one pack-reuse scope may exist, found "
        f"{len(scope_calls)}"
    )
    shared = _find_function(tree, "_launch_laguna_shared_rows", label)
    assert scope_calls[0] in list(ast.walk(shared)), (
        f"{label}: the shared-expert launcher must own the pack-reuse scope"
    )
    parents = _parent_map(tree)
    scope = parents.get(scope_calls[0])
    scope = parents.get(scope) if scope is not None else None
    assert isinstance(scope, ast.With), (
        f"{label}: shared gate/up launches must stay inside the reuse context"
    )
    launches = _calls_named(shared, "launch_gguf_linear")
    gate_up = [call for call in launches if call.args and _dotted(call.args[0]) in {"shared_gate", "shared_up"}]
    scoped_nodes = {node for statement in scope.body for node in ast.walk(statement)}
    assert len(gate_up) == 2 and all(call in scoped_nodes for call in gate_up), (
        f"{label}: shared gate/up launches must stay inside the reuse context"
    )
    enabled = _keyword(scope_calls[0], "enabled")
    assert enabled is not None and _dotted(enabled) == "use_activation_pack_reuse", (
        f"{label}: the scope must follow use_activation_pack_reuse, found "
        f"{None if enabled is None else _dotted(enabled)}"
    )
    shared_defaults = _defaults(shared)
    assert "use_activation_pack_reuse" in _parameters(shared), (
        f"{label}: _launch_laguna_shared_rows must take use_activation_pack_reuse"
    )
    assert shared_defaults["use_activation_pack_reuse"] is None, (
        f"{label}: the shared launcher must not invent a pack-reuse default"
    )

    rows = _find_function(tree, "run_laguna_moe_rows", label)
    defaults = _defaults(rows)
    assert "use_activation_pack_reuse" in defaults, (
        f"{label}: run_laguna_moe_rows must take use_activation_pack_reuse"
    )
    assert _const(
        defaults["use_activation_pack_reuse"], label, "pack-reuse default"
    ) is False, f"{label}: run_laguna_moe_rows must default pack reuse off"
    forwards = _calls_named(rows, "_launch_laguna_shared_rows")
    assert len(forwards) == 2, (
        f"{label}: both shared launch sites must remain, found {len(forwards)}"
    )
    for call in forwards:
        forwarded = _keyword(call, "use_activation_pack_reuse")
        assert forwarded is not None and _dotted(forwarded) == (
            "use_activation_pack_reuse"
        ), (
            f"{label}: a shared launch site dropped its pack-reuse forwarding "
            f"({_dotted(call)})"
        )


# --------------------------------------------------------------------------- #
# H7U: stable parallel MoE group-compaction policy
# --------------------------------------------------------------------------- #
def assert_h7u_group_compact_resolver(source: str) -> None:
    """H7U's protected semantics: the backend-qualified compaction policy."""
    label = MOE
    tree = _parse(source, label)
    constants = _module_constants(tree)

    modes = constants.get("_GROUP_COMPACT_MODES")
    assert isinstance(modes, ast.Call) and _dotted(modes.func) == "frozenset", (
        f"{label}: _GROUP_COMPACT_MODES must stay a frozenset literal"
    )
    assert modes.args and isinstance(modes.args[0], ast.Set), (
        f"{label}: _GROUP_COMPACT_MODES lost its literal mode set"
    )
    assert {
        _const(element, label, "compaction mode") for element in modes.args[0].elts
    } == {"serial", "parallel"}, (
        f"{label}: _GROUP_COMPACT_MODES must stay {{serial, parallel}}"
    )
    baseline = constants.get("_BASELINE_GROUP_COMPACT_MODE")
    assert baseline is not None and _const(
        baseline, label, "baseline compaction mode"
    ) == "serial", f"{label}: the serial rollback baseline must stay the default"

    reads = _capability_reads(tree, H7U_CAPABILITY, label)
    assert len(reads) == 1, (
        f"{label}: {H7U_CAPABILITY} must have one dispatch read, found "
        f"{len(reads)}"
    )
    resolver = _find_function(tree, "resolve_laguna_group_compact_mode", label)
    assert reads[0] in list(ast.walk(resolver)), (
        f"{label}: resolve_laguna_group_compact_mode no longer reads "
        f"{H7U_CAPABILITY}"
    )
    default = _keyword(reads[0], "default")
    if default is None and len(reads[0].args) > 2:
        default = reads[0].args[2]
    assert default is not None and _dotted(default) == (
        "_BASELINE_GROUP_COMPACT_MODE"
    ), f"{label}: the resolver must default to _BASELINE_GROUP_COMPACT_MODE"

    selection = _assignments_to(resolver, "selected")
    assert len(selection) == 1, f"{label}: one compaction selection must own the explicit override"
    choice = selection[0].value
    assert (
        isinstance(choice, ast.IfExp)
        and _dotted(choice.test) == "requested is None"
        and choice.body is reads[0]
        and _dotted(choice.orelse) == "str(requested)"
    ), f"{label}: explicit requested mode must override the backend capability"

    parameters = _parameters(resolver)
    assert "requested" in parameters, (
        f"{label}: the explicit serial/parallel override parameter is gone"
    )
    assert _const(
        _defaults(resolver)["requested"], label, "requested default"
    ) is None, f"{label}: the override must default to None"

    fail_closed = [
        node
        for node in ast.walk(resolver)
        if isinstance(node, ast.If)
        and any(
            isinstance(compare, ast.Compare)
            and _dotted(compare.left) == "parsed"
            and any(isinstance(op, ast.NotIn) for op in compare.ops)
            and _dotted(compare.comparators[0]) == "_GROUP_COMPACT_MODES"
            for compare in ast.walk(node.test)
        )
    ]
    assert fail_closed, (
        f"{label}: the resolver must reject modes outside _GROUP_COMPACT_MODES"
    )
    assert any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and _dotted(node.exc.func) == "ValueError"
        for node in ast.walk(fail_closed[0])
    ), f"{label}: an unknown compaction mode must raise ValueError"
    assert any(
        isinstance(node, ast.Return) and node.value is not None
        and _dotted(node.value) == "parsed"
        for node in ast.walk(resolver)
    ), f"{label}: the resolver must return the validated mode"


# --------------------------------------------------------------------------- #
# Campaign REDs the source-default contracts depend on
# --------------------------------------------------------------------------- #
_REQUIRED = "missing {module} test {name}()"


def _require_tests(tree: ast.Module, label: str, names: Iterable[str]) -> None:
    present = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in names:
        assert name in present, _REQUIRED.format(module=label, name=name)


def assert_h8a_gpu_red(source: str) -> None:
    """The H8A GPU RED still covers allocation, rollback, dispatch and parity."""
    label = H8A_GPU_RED
    tree = _parse(source, label)
    constants = _module_constants(tree)
    assert _const(constants["_PLANE_COUNT"], label, "_PLANE_COUNT") == (
        H8A_PLANE_COUNT
    ), f"{label}: the GPU RED no longer covers {H8A_PLANE_COUNT} resident planes"
    assert _dotted(constants["_RESIDENT_BYTES"]) == (
        "(_PLANE_COUNT * _PLANE_BYTES)"
    ), f"{label}: _RESIDENT_BYTES no longer derives from the plane count"
    assert _const(constants["_SOURCE_CAPABILITY"], label, "_SOURCE_CAPABILITY") == (
        H8A_CAPABILITY
    ), f"{label}: the GPU RED no longer targets {H8A_CAPABILITY}"
    assert _const(constants["_OWNER_CLASS"], label, "_OWNER_CLASS") == (
        H8A_OWNER_CLASS
    ), f"{label}: the GPU RED no longer targets {H8A_OWNER_CLASS}"
    topology = _const(constants["_EXPECTED_TOPOLOGY"], label, "_EXPECTED_TOPOLOGY")
    assert topology["setup_coltile16_producers"] == H8A_PLANE_COUNT
    assert topology["request_activation_packs"] == H8A_PLANE_COUNT
    assert topology["request_h7g_consumers"] == H8A_PLANE_COUNT
    assert topology["request_coltile16_producers"] == 0
    _require_tests(
        tree,
        label,
        (
            "test_h8a_frozen_target_current_sources_topology_and_admission_contract",
            "test_h8a_composite_registry_preflight_and_gfx1151_fail_closed",
            "test_h8a_owner_allocates_publishes_and_frees_exact_24_plane_map",
            "test_h8a_owner_failed_setup_rolls_back_without_publishing_partial_map",
            "test_h8a_source_selected_shared_session_dispatch_and_unshared_fallback",
            "test_h8a_complete_resident_plane_and_all_row_outputs_match_h7g",
        ),
    )
    allocation = _find_function(
        tree,
        "test_h8a_owner_allocates_publishes_and_frees_exact_24_plane_map",
        label,
    )
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "allocation_count"
        for node in ast.walk(allocation)
    ), f"{label}: the allocation RED no longer checks the owned plane count"
    rollback = _find_function(
        tree,
        "test_h8a_owner_failed_setup_rolls_back_without_publishing_partial_map",
        label,
    )
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "free_order"
        for node in ast.walk(rollback)
    ), f"{label}: the rollback RED no longer checks reverse-order frees"
    parity = _find_function(
        tree,
        "test_h8a_complete_resident_plane_and_all_row_outputs_match_h7g",
        label,
    )
    assert any(
        isinstance(node, ast.Name) and node.id == "resident_dev"
        for node in ast.walk(parity)
    ), f"{label}: the parity RED no longer allocates the resident plane"
    assert any(
        isinstance(node, ast.Name) and node.id == "_PLANE_BYTES"
        for node in ast.walk(parity)
    ), f"{label}: the parity RED no longer uses the resident plane size"


def assert_h8b_unit_red(source: str) -> None:
    """The H8B unit RED still covers scope publication and complete ownership."""
    label = H8B_UNIT_RED
    tree = _parse(source, label)
    constants = _module_constants(tree)
    assert _const(constants["_SOURCE_CAPABILITY"], label, "_SOURCE_CAPABILITY") == (
        H8B_CAPABILITY
    ), f"{label}: the unit RED no longer targets {H8B_CAPABILITY}"
    classes = _const(constants["_EXPECTED_CLASSES"], label, "_EXPECTED_CLASSES")
    assert set(classes) == {
        "full_attention_qkv",
        "swa_kv",
        "shared_q5_gate_up",
        "dense_q5_gate_up",
        "layer47_shared_q6_gate_up",
    }, f"{label}: the unit RED lost a pack-reuse class axis"
    assert sum(groups for groups, _calls, _removed in classes.values()) == 95, (
        f"{label}: the unit RED no longer covers the 95 recurrence groups"
    )
    assert sum(removed for _groups, _calls, removed in classes.values()) == 107, (
        f"{label}: the unit RED no longer covers the 107 removed pack calls"
    )
    topology = _const(constants["_EXPECTED_TOPOLOGY"], label, "_EXPECTED_TOPOLOGY")
    assert topology["packs_after"] == 223 and topology["packs_before"] == 330
    assert topology["dispatches_after"] == 2_155
    _require_tests(
        tree,
        label,
        (
            "test_h8b_frozen_target_sources_state_topology_and_admission_contract",
            "test_h8b_package_and_runtime_owner_follow_source_default",
            "test_h8b_scope_publishes_only_after_success_and_isolates_every_key_axis",
            "test_h8b_q5_q6_composites_share_exact_pack_without_changing_arithmetic",
            "test_h8b_runtime_owns_all_attention_dense_and_shared_groups_together",
        ),
    )
    isolation = _find_function(
        tree,
        "test_h8b_scope_publishes_only_after_success_and_isolates_every_key_axis",
        label,
    )
    assert _calls_named(isolation, "activation_pack_reuse_scope"), (
        f"{label}: the isolation RED no longer exercises the reuse scope"
    )
    ownership = _find_function(
        tree,
        "test_h8b_runtime_owns_all_attention_dense_and_shared_groups_together",
        label,
    )
    assert _calls_named(ownership, "_launch_laguna_shared_rows") or any(
        isinstance(node, ast.Attribute) and node.attr == "_launch_laguna_shared_rows"
        for node in ast.walk(ownership)
    ), f"{label}: the ownership RED no longer inspects the shared launcher"


def assert_h7u_live_mode_red(source: str) -> None:
    """The live MoE RED still proves the backend-qualified compaction default."""
    label = H7U_LIVE_RED
    tree = _parse(source, label)
    test = _find_function(
        tree, "test_laguna_group_compact_mode_is_backend_qualified", label
    )
    comparisons: set[tuple[tuple[Any, ...], Any]] = set()
    for node in ast.walk(test):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        compare = node.test
        if not isinstance(compare.left, ast.Call):
            continue
        if _dotted(compare.left.func) != "resolve_laguna_group_compact_mode":
            continue
        if len(compare.ops) != 1 or not isinstance(compare.ops[0], ast.Eq):
            continue
        arguments = tuple(
            _const(argument, label, "resolver argument")
            for argument in compare.left.args
        )
        comparisons.add((arguments, _const(compare.comparators[0], label, "mode")))
    required = {
        (("hip_gfx1100",), "parallel"),
        (("hip_gfx1100", "serial"), "serial"),
        (("hip_gfx1100", "parallel"), "parallel"),
        (("hip_gfx1151",), "parallel"),
        (("hip_gfx1151", "serial"), "serial"),
        (("hip_gfx1151", "parallel"), "parallel"),
    }
    missing = required - comparisons
    assert not missing, (
        f"{label}: the live MoE RED dropped backend-qualified mode assertions: "
        f"{sorted(missing)}"
    )
    raises = [
        node
        for node in ast.walk(test)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and _dotted(item.context_expr.func) == "pytest.raises"
            and item.context_expr.args
            and _dotted(item.context_expr.args[0]) == "ValueError"
            for item in node.items
        )
    ]
    assert raises, f"{label}: the live MoE RED no longer rejects unknown modes"
    rejected = {
        _dotted(call.args[1])
        for node in raises
        for call in _calls_named(node, "resolve_laguna_group_compact_mode")
        if len(call.args) > 1
    }
    assert "'atomic'" in rejected, (
        f"{label}: the live MoE RED no longer rejects the 'atomic' mode"
    )
