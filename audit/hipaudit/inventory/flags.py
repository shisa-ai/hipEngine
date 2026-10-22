"""HIPENGINE_* environment flags.

841 distinct names. `scripts/check_envs_docs.py` already proves every one is
*mentioned* in docs/ENVS.md; that is a different question from whether a flag is
alive, whether it should be a default, and what would retire it.

Per `AGENTS.md` "Flags are a cost, not a feature", a warranted flag defaults to
the production behaviour and carries a removal condition in docs/REFACTOR.md.
This extractor reports which flags do neither.
"""

from __future__ import annotations

import ast
import functools
import re

from ..core import REPO_ROOT, Row
from . import corpus, doc_corpus, register, tokens

NAME = re.compile(r"\bHIPENGINE_[A-Z0-9_]+\b")
#  Best-effort default detection at the read site.
DEFAULT_TRUE = re.compile(r"default\s*=\s*True")
DEFAULT_FALSE = re.compile(r"default\s*=\s*False")
#  `os.environ.get("X")` with no default, later compared against falsey strings,
#  is the default-off idiom in this tree.
BARE_GET = re.compile(r"""environ\.get\(\s*["']HIPENGINE_[A-Z0-9_]+["']\s*\)""")


#  How the runtime uses a name. Triage kept getting this wrong by grepping for
#  `environ.get("NAME")` and missing the tree's usual shapes: a module constant
#  (`_X_ENV = "HIPENGINE_X"`) handed to `_env_flag(_X_ENV, True)`, a helper that
#  takes the name as a parameter, or a table of names. The row now states it.
READ_ENV = "env"            # read from the environment (directly or via a constant)
READ_INDIRECT = "indirect"  # handed to a helper or table that may read it
READ_MENTION = "mention"    # only strings, comments, docstrings or prints
BOUND = "bound"             # written by the runtime: a profile binding or `environ[X] = ...`
BINDER = re.compile(r"(?:^|_)(?:bind|set|put|export)(?:_|$)|setenv$")
SHELL_OR_C_READ = re.compile(r"""getenv\(\s*["']{name}["']|\$\{{?{name}\b""")
TRUTHY = {"1", "true", "on", "yes"}
FALSY = {"", "0", "false", "off", "no"}


def _is_env_binder(func: ast.AST) -> bool:
    text = ast.unparse(func).lower()
    last = text.rsplit(".", 1)[-1]
    return (text.endswith(("environ.pop", "environ.setdefault", "environ.update"))
            or ("env" in last and bool(BINDER.search(last))))


def _is_env_reader(func: ast.AST) -> bool:
    text = ast.unparse(func).lower()
    return "env" in text.rsplit(".", 1)[-1] or text.endswith(("environ.get", "getenv"))


def _default(call: ast.Call, literals: dict[str, ast.Constant]) -> str:
    """`on`/`off`/`value`/`unknown` from a read call's default argument."""
    supplied = call.args[1] if len(call.args) > 1 else next(
        (k.value for k in call.keywords if k.arg == "default"), None)
    if isinstance(supplied, ast.Name) and supplied.id in literals:
        supplied = literals[supplied.id]
    if supplied is None:
        return "off" if ast.unparse(call.func).endswith(("environ.get", "getenv")) else "unknown"
    if isinstance(supplied, ast.Constant):
        value = supplied.value
        if value is True or (isinstance(value, str) and value.strip().lower() in TRUTHY):
            return "on"
        if value is False or value is None or (isinstance(value, str) and value.strip().lower() in FALSY):
            return "off"
        return "value"
    return "unknown"


@functools.lru_cache(maxsize=None)
def module_reads(path: str, text: str) -> dict[str, list[tuple[str, int, str]]]:
    """`{flag: [(kind, line, default)]}` for one runtime module."""
    out: dict[str, list[tuple[str, int, str]]] = {}
    if not path.endswith(".py"):
        for name in set(NAME.findall(text)):
            pattern = re.compile(SHELL_OR_C_READ.pattern.format(name=re.escape(name)))
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    out.setdefault(name, []).append((READ_ENV, number, "unknown"))
        return out
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return out
    constants: dict[str, str] = {}
    literals: dict[str, ast.Constant] = {}      # module-level `NAME = <literal>`, for defaults
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    literals[target.id] = node.value
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else [])
        value = getattr(node, "value", None)
        if targets and isinstance(value, ast.Constant) and isinstance(value.value, str) \
                and NAME.fullmatch(value.value):
            for target in targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    constants[target.id if isinstance(target, ast.Name) else target.attr] = value.value

    def flag_of(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and NAME.fullmatch(node.value):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        if isinstance(node, ast.Attribute):
            return constants.get(node.attr)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            binder = _is_env_binder(node.func)
            reader = not binder and _is_env_reader(node.func)
            for arg in [*node.args, *(k.value for k in node.keywords)]:
                name = flag_of(arg)
                if name:
                    kind = BOUND if binder else READ_ENV if reader else READ_INDIRECT
                    out.setdefault(name, []).append(
                        (kind, node.lineno,
                         _default(node, literals) if reader and node.args and arg is node.args[0]
                         else "unknown"))
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value).endswith("environ"):
            name = flag_of(node.slice)
            if name:
                kind = BOUND if isinstance(node.ctx, ast.Store) else READ_ENV
                out.setdefault(name, []).append((kind, node.lineno, "unknown"))
        elif isinstance(node, ast.Compare) and any(
                isinstance(c, ast.Name | ast.Attribute) and ast.unparse(c).endswith("environ")
                for c in node.comparators):
            name = flag_of(node.left)
            if name:
                out.setdefault(name, []).append((READ_ENV, node.lineno, "unknown"))
        elif isinstance(node, (ast.Dict, ast.Tuple, ast.List, ast.Set)):
            for item in [*getattr(node, "values", []), *getattr(node, "elts", [])]:
                name = flag_of(item)
                #  `__all__ = ("HIPENGINE_X_ENV", ...)` lists a Python name, not a flag.
                if name and name not in identifiers:
                    out.setdefault(name, []).append((READ_INDIRECT, getattr(item, "lineno", 0), "unknown"))
    return out


def runtime_reads(name: str, code: dict[str, str]) -> list[tuple[str, str, str]]:
    """`[(kind, path:line, default)]` for every runtime use of `name`, sorted."""
    found = []
    for path, text in code.items():
        if not path.startswith("hipengine/") or name not in text:
            continue
        for kind, line, default in module_reads(path, text).get(name, []):
            found.append((kind, f"{path}:{line}", default))
    return sorted(set(found), key=lambda r: (r[0] != READ_ENV, r[1]))


def read_sites(name: str, code: dict[str, str]) -> tuple[dict[str, list[str]], str]:
    """`{root: [path:line]}` plus every source line that names the flag."""
    by_root: dict[str, list[str]] = {}
    context: list[str] = []
    for path, text in code.items():
        if name not in text:
            continue
        root = path.split("/", 1)[0]
        for number, line in enumerate(text.splitlines(), 1):
            if name in line:
                by_root.setdefault(root, []).append(f"{path}:{number}")
                if root == "hipengine":
                    context.append(line.strip())
    return by_root, "\n".join(context)


@register("flags")
def extract() -> tuple[list[Row], dict]:
    code = corpus()
    docs = doc_corpus()
    ledger_text = docs.get("docs/REFACTOR.md", "")
    envs_text = docs.get("docs/ENVS.md", "")

    names: set[str] = set()
    for text in code.values():
        names.update(NAME.findall(text))
    names.discard("HIPENGINE_")

    rows: list[Row] = []
    for name in sorted(names):
        sites, context = read_sites(name, code)
        runtime = sites.get("hipengine", [])
        reads = runtime_reads(name, code)
        kinds = {kind for kind, _, _ in reads}
        bound = BOUND in kinds
        kinds.discard(BOUND)
        how = (READ_ENV if READ_ENV in kinds else READ_INDIRECT if kinds or bound
               else READ_MENTION if runtime else "none")
        signals: list[str] = []
        if how == READ_ENV:
            signals.append("read from the environment under hipengine/")
        elif how == READ_INDIRECT:
            signals.append("handed to a helper or table under hipengine/, not read directly "
                           "— confirm what reads it")
        if bound:
            signals.append("set by the runtime (a profile binding or an environ write)")
        if how == READ_MENTION:
            signals.append("named under hipengine/ but never read from the environment there "
                           "— only constants, strings or comments")

        if not runtime:
            where = ", ".join(sorted(sites)) or "nowhere"
            signals.append(f"no read site under hipengine/ — only {where}")
        if set(sites) <= {"tests"}:
            signals.append("read only from tests/")
        if set(sites) <= {"benchmarks", "scripts"}:
            signals.append("read only from harnesses, never by the runtime")

        #  A flag read as default-on in one module and default-off in another is a
        #  split default: the same name means different things on different routes.
        declared = {d for kind, _, d in reads if kind == READ_ENV and d != "unknown"}
        if declared:
            default = ("split" if {"on", "off"} <= declared else "value" if "value" in declared
                       else declared.pop())
        else:
            on = bool(DEFAULT_TRUE.search(context))
            off = bool(DEFAULT_FALSE.search(context) or BARE_GET.search(context))
            default = ("on" if on and not off else "off" if off and not on
                       else "split" if on and off else "unknown")
        if default == "split":
            signals.append(
                "conflicting defaults across runtime read sites — default-on in one module "
                "and default-off in another")
        elif default == "off":
            signals.append("reads as default-off")

        if name not in ledger_text:
            signals.append("no docs/REFACTOR.md entry — no recorded removal condition")
        documented = name in envs_text
        if not documented:
            signals.append("not documented in docs/ENVS.md")

        if len(runtime) > 6:
            signals.append(f"read at {len(runtime)} runtime sites — branching is spread out")

        rows.append(Row(
            kind="flag",
            key=name,
            title=name,
            location=(runtime or [s for v in sites.values() for s in v] or [""])[0],
            evidence={
                "default": default,
                "sites_by_root": {k: len(v) for k, v in sorted(sites.items())},
                "runtime_sites": runtime[:8],
                "runtime_read": how,
                "runtime_bound": bound,
                "runtime_reads": [f"{where} {kind} default={d}" for kind, where, d in reads[:6]],
                "in_refactor_ledger": name in ledger_text,
                "in_envs_doc": documented,
            },
            signals=signals,
            hints={"anchor": "env", "refs": [name], "tokens": tokens(name.replace("_", " "))},
        ))

    return rows, {"names": len(rows), "roots": "hipengine, scripts, tests, benchmarks"}
