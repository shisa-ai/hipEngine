"""Functions a later definition in the same scope replaces, before anything reads them.

Module, class and function bodies run top to bottom, so a second `def name` in
the same body rebinds the first one and every read afterwards gets the second.
The first body is then unreachable *through its own name*, which makes it a
removal candidate — not proof that nothing else holds a reference to it.

The report is deliberately narrow: one row per name and scope, naming both
source locations, the signature and body of each definition, what each body
starts with, and where the two bodies first diverge, so the reviewer compares
them instead of trusting a verdict. A group is skipped where the repeat can be
intentional or where the earlier binding is still reachable:

- any definition of the name is decorated or is a class. `@property`/`@x.setter`,
  `@overload`, `@register(...)`, `functools.cache` and click commands rebind or
  register on purpose, and a class body executes at definition time and can
  register, patch a base or mutate the module. A decorated `def` or a `class`
  sharing the name with plain `def`s rejects the whole group: the name then has
  a binding this check cannot read as a plain replacement.
- a definition is not a direct statement of the scope. Inside
  `if`/`try`/`for`/`with`/`match` a second `def` is an alternative — a
  `TYPE_CHECKING` stub, an `ImportError` fallback, a platform switch — not a
  replacement.
- anything between the two definitions reads or rebinds the name, or the later
  definition's own signature uses it. `alias = f`, `handlers.append(f)`,
  `prepare(f)`, `del f`, `if flag: def f`, `f = other`, `import f` and
  `def g(cb=f)` all keep the earlier binding reachable or add a binding this
  check does not model; `def g() -> f` reads it at definition time. A read after
  the pair does not count: by then the name is the later function.

That window is read from the parse tree, not from line numbers: the statements
between the two definitions and the later definition's own parameters, defaults,
annotations and return type. A signature that wraps onto the line its body
starts on is therefore still seen.

Identity is the file, the scope, the name and the shape of both definitions, so
moving or reformatting either one keeps the row and its decision, while changing
either body or signature is a new row. Positions sit under the `lines` evidence
key, which the decision hash treats as volatile, so a shift above a definition
does not age its triage.

What this does not establish is that the earlier body is dead. A name reached
without a `Name` node is invisible here — `globals()["f"]`, `getattr(module,
"f")`, `sys.modules[...]`, `eval`, `locals()`, a `from x import *` fed by a
string in `__all__`, a decorator that patches the class after its body finished,
or a rebinding after the pair. Groups with three or more definitions are skipped:
intermediate definitions can capture earlier bindings. Default expressions and
annotations can also have side effects even if a function body is never called.
This is an observation about one scope, and triage decides.
"""

from __future__ import annotations

import ast
import re

from ..core import Row, digest, slug
from ..inventory import corpus
from . import finding, register

DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
BINDINGS = DEFS + (ast.ClassDef,)
DEF_LINE = re.compile(r"^([ \t]*)(?:async[ \t]+)?def[ \t]+(\w+)", re.M)


def _maybe_repeats(text: str) -> bool:
    """Cheap gate before parsing: does one block define one name twice?

    Statements of one block share an indent column, so two definitions in one
    body always report the same width. Tabs expand the way Python expands them,
    and a file this gate lets through is still judged by its parse tree, so a
    mixed-indent file costs a parse rather than a finding.
    """
    seen: dict[tuple[int, str], int] = {}
    for match in DEF_LINE.finditer(text):
        key = (len(match.group(1).expandtabs(8)), match.group(2))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            return True
    return False


def _bodies(node: ast.AST, scope: str = ""):
    """`(scope name, body)` for every body that runs top to bottom."""
    yield scope, node.body
    for child in node.body:
        if isinstance(child, BINDINGS):
            yield from _bodies(child, f"{scope}.{child.name}" if scope else child.name)


def _digest(*nodes: ast.AST) -> str:
    """Identity of parsed nodes: formatting, comments and position do not count."""
    return digest(*(ast.dump(node, include_attributes=False) for node in nodes))


def _signature(node: ast.AST) -> str:
    """The definition-time half of a `def`: parameters, annotations, return type."""
    return _digest(node.args, *([node.returns] if node.returns is not None else []),
                   *getattr(node, "type_params", ()))


def _code_body(node: ast.AST) -> list[ast.stmt]:
    """The statements that do the work: a leading docstring is not one of them."""
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]
    return body


def _flat(text: str) -> str:
    """One line, so a statement's shape survives a table cell."""
    return " ".join(text.split())[:140]


def _summary(node: ast.AST) -> str:
    """The first statement after the docstring: where the body starts."""
    body = _code_body(node)
    return _flat(ast.unparse(body[0])) if body else ""


def _divergence(first: ast.AST, last: ast.AST) -> tuple[str, str]:
    """The first statement where the two bodies disagree, one line each."""
    earlier = [ast.unparse(statement) for statement in _code_body(first)]
    later = [ast.unparse(statement) for statement in _code_body(last)]
    for mine, theirs in zip(earlier, later):
        if mine != theirs:
            return _flat(mine), _flat(theirs)
    extra = min(len(earlier), len(later))
    if len(earlier) == len(later):
        return "", ""
    return (_flat(earlier[extra]) if len(earlier) > extra else "",
            _flat(later[extra]) if len(later) > extra else "")


def _repeats(body: list[ast.stmt]) -> list[tuple[str, list[int]]]:
    """`(name, body indices)` for exactly two plain `def` bindings.

    A decorated `def` or a `class` of the same name anywhere in the body rejects
    the group, even when the two plain `def`s are the ones a reader would pair up.
    """
    defined: dict[str, list[tuple[int, ast.AST]]] = {}
    for index, statement in enumerate(body):
        if isinstance(statement, BINDINGS):
            defined.setdefault(statement.name, []).append((index, statement))
    return [(name, [index for index, _node in members])
            for name, members in defined.items()
            if len(members) == 2
            and not any(not isinstance(node, DEFS) or node.decorator_list
                        for _index, node in members)]


def _touches(nodes, name: str) -> bool:
    """Does any of these nodes read or rebind `name`?

    Either one makes the pair ambiguous: a read keeps the earlier binding
    reachable, and a rebind — a conditional `def`, an import, an assignment —
    adds a binding this check does not model.
    """
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == name:
                return True
            if isinstance(child, BINDINGS) and child.name == name:
                return True
            if isinstance(child, ast.alias) and (child.asname or child.name.split(".")[0]) == name:
                return True
            if isinstance(child, ast.ExceptHandler) and child.name == name:
                return True
    return False


def _header(node: ast.AST) -> list[ast.AST]:
    """The definition-time parts of a `def`: decorators, parameters, return type."""
    return ([node.args] + ([node.returns] if node.returns is not None else [])
            + list(node.decorator_list) + list(getattr(node, "type_params", ())))


def _row(path: str, scope: str, name: str, first: ast.AST, last: ast.AST) -> Row:
    label = f"{scope}.{name}" if scope else name
    same_signature = _signature(first) == _signature(last)
    same_body = _digest(*first.body) == _digest(*last.body)
    same_code = _digest(*_code_body(first)) == _digest(*_code_body(last))
    mine, theirs = _divergence(first, last)
    row = finding(
        "shadowed-definition",
        f"{path}:{slug(label, 40)}-{digest(label, _digest(first), _digest(last))[:6]}",
        f"{label} is defined twice in one scope: the later definition replaces the first"[:120],
        f"{path}:{first.lineno}",
        fix="Compare the two definitions and keep the one that runs: delete the earlier body "
            "if nothing outside this scope reaches it, or fold it into the later one if it "
            "was meant to extend it. Check dynamic access — `globals()`, `getattr`, a "
            "registry holding the function — and definition-time effects in defaults and "
            "annotations before deleting.",
        why="a second `def` in the same scope rebinds the first before anything in the scope "
            "reads it, so the earlier body is a removal candidate, not a proven dead path",
        evidence={
            "scope": scope or "<module>",
            "name": name,
            "lines": {"earlier": first.lineno, "later": last.lineno},
            "earlier_definition": _digest(first),
            "later_definition": _digest(last),
            "signature_identical": same_signature,
            "bodies_identical": same_body,
            "earlier_starts_with": _summary(first),
            "later_starts_with": _summary(last),
            "earlier_difference": mine,
            "later_difference": theirs,
        },
    )
    row.signals.append("the two signatures are identical" if same_signature
                       else "the two signatures differ")
    if same_body:
        row.signals.append("the two bodies are identical")
    elif same_code:
        row.signals.append("the two bodies differ only in their docstring")
    else:
        row.signals.append("the two bodies differ")
    return row


@register("shadowed-definition")
def shadowed_definition() -> tuple[list[Row], dict]:
    """An undecorated `def` that a later definition in the same scope replaces."""
    rows, parsed = [], 0
    for path, text in sorted(corpus().items()):
        if not path.endswith(".py") or not _maybe_repeats(text):
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue        # a file that does not parse yields no structure, not noise
        parsed += 1
        for scope, body in _bodies(tree):
            for name, indices in _repeats(body):
                first, last = body[indices[0]], body[indices[-1]]
                between = body[indices[0] + 1:indices[-1]]
                if _touches(between, name) or _touches(_header(last), name):
                    continue
                rows.append(_row(path, scope, name, first, last))
    return rows, {"scanned": "**/*.py", "parsed": parsed}
