"""HIPENGINE_* environment flags.

841 distinct names. `scripts/check_envs_docs.py` already proves every one is
*mentioned* in docs/ENVS.md; that is a different question from whether a flag is
alive, whether it should be a default, and what would retire it.

Per `AGENTS.md` "Flags are a cost, not a feature", a warranted flag defaults to
the production behaviour and carries a removal condition in docs/REFACTOR.md.
This extractor reports which flags do neither.
"""

from __future__ import annotations

import re

from ..core import REPO_ROOT, Row
from . import corpus, doc_corpus, register

NAME = re.compile(r"\bHIPENGINE_[A-Z0-9_]+\b")
#  Best-effort default detection at the read site.
DEFAULT_TRUE = re.compile(r"default\s*=\s*True")
DEFAULT_FALSE = re.compile(r"default\s*=\s*False")
#  `os.environ.get("X")` with no default, later compared against falsey strings,
#  is the default-off idiom in this tree.
BARE_GET = re.compile(r"""environ\.get\(\s*["']HIPENGINE_[A-Z0-9_]+["']\s*\)""")


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
        signals: list[str] = []

        if not runtime:
            where = ", ".join(sorted(sites)) or "nowhere"
            signals.append(f"no read site under hipengine/ — only {where}")
        if set(sites) <= {"tests"}:
            signals.append("read only from tests/")
        if set(sites) <= {"benchmarks", "scripts"}:
            signals.append("read only from harnesses, never by the runtime")

        #  A flag read as default-on in one module and default-off in another is a
        #  split default: the same name means different things on different routes.
        on = bool(DEFAULT_TRUE.search(context))
        off = bool(DEFAULT_FALSE.search(context) or BARE_GET.search(context))
        default = "on" if on and not off else "off" if off and not on else "split" if on and off else "unknown"
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
                "in_refactor_ledger": name in ledger_text,
                "in_envs_doc": documented,
            },
            signals=signals,
        ))

    return rows, {"names": len(rows), "roots": "hipengine, scripts, tests, benchmarks"}
