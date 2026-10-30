"""Bounded source-literal reader for backend kernel package declarations.

GGUF admission and the quant-route audit read backend capability constants out
of ``hipengine/kernels/<backend>/`` source. Neither may import the backend
package -- registration needs the HIP toolchain -- and neither may evaluate
arbitrary code. This reader resolves, without importing or executing anything:

* literal assignments, including annotated ``NAME: type = ...`` declarations;
* the container wrappers the backend packages put around one literal argument
  (``frozenset({...})``, ``set``, ``tuple``, ``list``, and the empty
  zero-argument form);
* ordered local literal aliases (``ALIAS = NAME`` assigned earlier in the same
  module);
* explicit same-package import aliases and re-exports: absolute
  ``from hipengine.kernels.<backend> import policy_x as _policy_x``, relative
  ``from . import policy_x`` / ``from .policy_x import NAME as NAME``, and
  ``import hipengine.kernels.<backend>.policy_x as _policy_x`` followed by
  ``NAME = _policy_x.NAME``. Every import binding is a module-level name, so
  ``from .policy import NAME as NAME`` is reported as a re-export.

Everything else is unreadable and reports :data:`NONLITERAL_STATUS` -- never
guessed and never evaluated. That covers import cycles (including a module
that re-exports from itself), module aliases bound outside the backend
package, missing or unparseable modules, non-identifier module paths (so a
declaration cannot escape the backend directory), arbitrary calls, malformed
or shadowed container wrappers, and any other expression.

Bindings are read in source order. A later import rebinds its local name even
when the import is rejected as external, so an earlier literal or same-package
module alias never survives its rebinding. A top-level method call
(``NAMES.append(...)``) or subscript assignment (``NAMES[0] = ...``) mutates
an earlier container literal, so that name stops being a literal too.

Only module-level statements are read. A declaration inside a conditional or
``try`` block is not a module constant and is treated as absent, exactly like
a missing name.

A backend package may also declare the ordered modules that own its capability
constants (``POLICY_MODULE_NAMES = ("policy_a", ...)``). The root read merges
those owners' own uppercase module constants, so a name the package no longer
re-binds still reads as the value its owner declares. The merge is bounded: it
happens only at the package root, only for a literal tuple of plain module
identifiers, only for names the package itself does not bind, and only within
the same cycle and depth limits as every other module read. A missing,
unreadable, nonliteral, or malformed declaration contributes no owners, and the
package's own binding wins even when that binding is nonliteral.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LITERAL_STATUS",
    "NONLITERAL_STATUS",
    "POLICY_MODULE_NAMES_DECLARATION",
    "SourceAssignment",
    "default_source_root",
    "read_backend_assignments",
]


LITERAL_STATUS = "literal"
NONLITERAL_STATUS = "nonliteral"

# Root declaration naming the ordered modules that own a backend's constants.
POLICY_MODULE_NAMES_DECLARATION = "POLICY_MODULE_NAMES"

# ``<root>/hipengine/kernels/<backend>`` holds one backend package.
_PACKAGE_PARENT = ("hipengine", "kernels")

# Safe container constructors the backend packages use around a literal.
_LITERAL_CALL_WRAPPERS = {
    "frozenset": frozenset,
    "set": set,
    "tuple": tuple,
    "list": list,
}

# Acyclic re-export chains are two levels deep today (initializer -> policy
# module -> shared geometry module). The bound keeps a pathological chain from
# turning the cold path into a source walk; deeper chains read as nonliteral.
_MAX_MODULE_DEPTH = 8


class _UnreadableSource(Exception):
    """A module path that must not be read (non-identifier or escaping)."""


@dataclass(frozen=True)
class SourceAssignment:
    """One module-level assignment: its bounded value and resolution status."""

    value: object | None
    status: str

    @property
    def is_literal(self) -> bool:
        return self.status == LITERAL_STATUS


_NONLITERAL = SourceAssignment(None, NONLITERAL_STATUS)


def default_source_root() -> Path:
    """Repository root the packaged backend sources live under."""

    return Path(__file__).resolve().parents[2]


def read_backend_assignments(
    backend: str, *, root: Path | None = None
) -> dict[str, SourceAssignment]:
    """Read one backend package's module-level bindings from source.

    Returns every module-level name bound by an assignment or an import. A
    binding is ``literal`` only when its value resolved through the bounded
    reader; re-exports of a readable same-package constant are literal, while
    external imports, module objects, and every unreadable expression are
    ``nonliteral``.

    ``backend`` must be a plain package name; anything else reads as an empty
    package, so a caller cannot turn a backend key into a filesystem path.
    ``root`` is the directory containing ``hipengine/`` and defaults to this
    checkout. A missing, unreadable, or unparseable package reads as ``{}`` --
    the fail-closed "declares nothing" answer, never an exception.
    """

    name = str(backend)
    if not name.isidentifier():
        return {}
    base = default_source_root() if root is None else Path(root)
    return _SourceResolver(base, name).root_assignments()


class _SourceResolver:
    """Sequential, cycle-rejecting reader for one backend package."""

    def __init__(self, root: Path, backend: str) -> None:
        self._backend = backend
        self._backend_dir = root.joinpath(*_PACKAGE_PARENT, backend)
        self._modules: dict[tuple[str, ...], dict[str, SourceAssignment]] = {}
        self._parsing: set[tuple[str, ...]] = set()
        self._module_exists_cache: dict[tuple[str, ...], bool] = {}
        self._cycles = 0

    def root_assignments(self) -> dict[str, SourceAssignment]:
        return self._merge_declared_owners(self._parse_module(()))

    def _merge_declared_owners(
        self, values: dict[str, SourceAssignment]
    ) -> dict[str, SourceAssignment]:
        """Add the uppercase constants owned by the declared policy modules.

        Fail-closed on purpose: an absent, nonliteral, non-tuple, or
        non-identifier declaration yields no owners, and a name the package
        already binds is never replaced (the package binding wins even when it
        is nonliteral). Reads stay cold and bounded by the ordinary module
        cycle and depth guards.
        """

        declared = values.get(POLICY_MODULE_NAMES_DECLARATION)
        if declared is None or not declared.is_literal:
            return values
        owners = declared.value
        if not isinstance(owners, tuple):
            return values
        if not all(isinstance(owner, str) and owner.isidentifier() for owner in owners):
            return values
        merged = dict(values)
        for owner in owners:
            for name, assignment in self._parse_module((owner,)).items():
                if name.isupper() and name not in merged:
                    merged[name] = assignment
        return merged

    # -- module files -----------------------------------------------------

    def _module_path(self, parts: tuple[str, ...]) -> Path:
        for part in parts:
            if not isinstance(part, str) or not part.isidentifier():
                raise _UnreadableSource(part)
        if parts:
            path = self._backend_dir.joinpath(*parts).with_suffix(".py")
        else:
            path = self._backend_dir / "__init__.py"
        resolved = path.resolve()
        try:
            resolved.relative_to(self._backend_dir.resolve())
        except ValueError as exc:  # pragma: no cover - identifiers cannot escape
            raise _UnreadableSource(path) from exc
        return resolved

    def _module_exists(self, parts: tuple[str, ...]) -> bool:
        cached = self._module_exists_cache.get(parts)
        if cached is not None:
            return cached
        try:
            exists = self._module_path(parts).is_file()
        except _UnreadableSource:
            exists = False
        self._module_exists_cache[parts] = exists
        return exists

    # -- module parsing ---------------------------------------------------

    def _parse_module(self, parts: tuple[str, ...]) -> dict[str, SourceAssignment]:
        cached = self._modules.get(parts)
        if cached is not None:
            return cached
        if parts in self._parsing or len(self._parsing) >= _MAX_MODULE_DEPTH:
            # Import cycle (or an unbounded chain): unreadable, not guessed.
            self._cycles += 1
            return {}
        before = self._cycles
        self._parsing.add(parts)
        try:
            result = self._read_module(parts)
        finally:
            self._parsing.discard(parts)
        if self._cycles == before:
            self._modules[parts] = result
        return result

    def _read_module(self, parts: tuple[str, ...]) -> dict[str, SourceAssignment]:
        try:
            source = self._module_path(parts).read_text()
        except (_UnreadableSource, OSError):
            return {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        return self._resolve_body(tree, parts)

    def _resolve_body(
        self, tree: ast.Module, parts: tuple[str, ...]
    ) -> dict[str, SourceAssignment]:
        values: dict[str, SourceAssignment] = {}
        attribute_bases: dict[str, tuple[str, ...]] = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                base = self._import_from_base(parts, node)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    self._forget(local, values, attribute_bases)
                    if base is None:
                        # External import: the name is bound, not readable.
                        values[local] = _NONLITERAL
                        continue
                    submodule = base + (alias.name,)
                    attribute_bases[local] = submodule
                    values[local] = (
                        _NONLITERAL
                        if self._module_exists(submodule)
                        else self._resolve_constant(base, alias.name)
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    self._forget(local, values, attribute_bases)
                    target = (
                        self._same_package_module(alias.name)
                        if alias.asname is not None
                        else None
                    )
                    if target is not None:
                        attribute_bases[local] = target
                    # An imported module object is a binding, not a literal.
                    values[local] = _NONLITERAL
            elif isinstance(node, ast.Assign):
                value = self._resolve_expression(node.value, values, attribute_bases)
                for target in node.targets:
                    self._bind(target, value, values, attribute_bases)
            elif isinstance(node, ast.AnnAssign):
                if node.value is None:
                    continue
                value = self._resolve_expression(node.value, values, attribute_bases)
                self._bind(node.target, value, values, attribute_bases)
            elif isinstance(node, ast.AugAssign):
                self._bind(node.target, _NONLITERAL, values, attribute_bases)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self._invalidate_mutated_binding(node.value, values, attribute_bases)
        return values

    @staticmethod
    def _forget(
        name: str,
        values: dict[str, SourceAssignment],
        attribute_bases: dict[str, tuple[str, ...]],
    ) -> None:
        """Drop every earlier binding of ``name`` before it is rebound."""

        values.pop(name, None)
        attribute_bases.pop(name, None)

    @staticmethod
    def _root_name(node: ast.expr) -> str | None:
        while isinstance(node, ast.Subscript):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def _bind(
        self,
        target: ast.expr,
        value: SourceAssignment,
        values: dict[str, SourceAssignment],
        attribute_bases: dict[str, tuple[str, ...]],
    ) -> None:
        if isinstance(target, ast.Name):
            self._forget(target.id, values, attribute_bases)
            values[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Destructuring is not one value per name: never guessed.
            for element in target.elts:
                if isinstance(element, ast.Name):
                    self._forget(element.id, values, attribute_bases)
                    values[element.id] = _NONLITERAL
        else:
            root = self._root_name(target)
            if root is not None:
                # In-place mutation: the earlier literal value is stale.
                self._forget(root, values, attribute_bases)
                values[root] = _NONLITERAL

    def _invalidate_mutated_binding(
        self,
        call: ast.Call,
        values: dict[str, SourceAssignment],
        attribute_bases: dict[str, tuple[str, ...]],
    ) -> None:
        """A top-level ``NAME.method(...)`` call may mutate NAME's value."""

        func = call.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            return
        name = func.value.id
        target = attribute_bases.get(name)
        if target is not None and self._module_exists(target):
            # A call on a module object does not rebind the module name.
            return
        if name in values:
            values[name] = _NONLITERAL

    # -- imports ----------------------------------------------------------

    def _same_package_module(self, dotted: str) -> tuple[str, ...] | None:
        """Module parts inside this backend package, or ``None`` if external."""

        parts = tuple(str(dotted).split("."))
        if (
            len(parts) < 3
            or parts[0] != _PACKAGE_PARENT[0]
            or parts[1] != _PACKAGE_PARENT[1]
            or parts[2] != self._backend
        ):
            return None
        rest = parts[3:]
        if not all(part.isidentifier() for part in rest):
            return None
        return rest

    def _import_from_base(
        self, parts: tuple[str, ...], node: ast.ImportFrom
    ) -> tuple[str, ...] | None:
        if node.level == 0:
            if node.module is None:
                return None
            return self._same_package_module(node.module)
        base = list(parts[:-1])
        if node.level - 1 > len(base):
            return None
        base = base[: len(base) - (node.level - 1)]
        if node.module:
            tail = tuple(node.module.split("."))
            if not all(part.isidentifier() for part in tail):
                return None
            base.extend(tail)
        return tuple(base)

    # -- expressions ------------------------------------------------------

    def _resolve_expression(
        self,
        node: ast.expr,
        values: dict[str, SourceAssignment],
        attribute_bases: dict[str, tuple[str, ...]],
    ) -> SourceAssignment:
        if isinstance(node, ast.Name):
            return values.get(node.id, _NONLITERAL)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            target = attribute_bases.get(node.value.id)
            if target is not None and self._module_exists(target):
                return self._resolve_constant(target, node.attr)
            return _NONLITERAL
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _LITERAL_CALL_WRAPPERS
            and (node.func.id in values or node.func.id in attribute_bases)
        ):
            # A rebound name is not the builtin container constructor.
            return _NONLITERAL
        return _literal_value(node)

    def _resolve_constant(
        self, parts: tuple[str, ...], name: str
    ) -> SourceAssignment:
        if not name.isidentifier():
            return _NONLITERAL
        entry = self._parse_module(parts).get(name)
        return _NONLITERAL if entry is None else entry


def _literal_value(node: ast.expr) -> SourceAssignment:
    """Resolve one expression bounded to literals and safe container wrappers.

    Anything else is unreadable: the value is never guessed and the expression
    is never evaluated. A wrapper whose argument is a literal but whose
    construction fails (``frozenset(1)``, ``set({[]})``) is unreadable too.
    """

    try:
        return SourceAssignment(ast.literal_eval(node), LITERAL_STATUS)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        wrapper = _LITERAL_CALL_WRAPPERS.get(node.func.id)
        if wrapper is not None and not node.keywords:
            if not node.args:
                return _wrap_literal(wrapper, ())
            if len(node.args) == 1:
                try:
                    inner = ast.literal_eval(node.args[0])
                except (ValueError, TypeError):
                    return _NONLITERAL
                return _wrap_literal(wrapper, (inner,))
    return _NONLITERAL


def _wrap_literal(wrapper, args: tuple[object, ...]) -> SourceAssignment:
    try:
        return SourceAssignment(wrapper(*args), LITERAL_STATUS)
    except (TypeError, ValueError):
        return _NONLITERAL
