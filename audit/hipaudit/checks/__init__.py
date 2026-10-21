"""Code checks: scan the tree and queue work that can actually be done.

The inventory catalogues debt somebody already wrote down. These checks look at
the **code** and find things with a concrete fix, so each finding names the edit
that closes it.

Findings share the triage store with the inventory, so a `wontfix` recorded here
is honoured everywhere, and a finding survives rescans the same way an inventory
row does.
"""

from __future__ import annotations

import ast
import re
from typing import Callable

from ..core import REPO_ROOT, Row, digest, slug
from ..inventory import corpus, doc_corpus, tokens

CHECKS: dict[str, Callable[[], tuple[list[Row], dict]]] = {}


def register(name: str):
    def wrap(fn):
        CHECKS[name] = fn
        return fn
    return wrap


#  Blocks whose header says what the code inside is for. Two identical statements
#  in one scope are told apart by the chain of these they sit in.
BLOCK_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
               ast.Try, ast.ExceptHandler, ast.Match, ast.match_case)
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _block_fingerprint(node: ast.AST) -> str:
    """What a block is, as text: formatting, blank lines and comments do not count."""
    if isinstance(node, (ast.If, ast.While)):
        return f"{type(node).__name__} {ast.unparse(node.test)}"
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return f"{type(node).__name__} {ast.unparse(node.target)} in {ast.unparse(node.iter)}"
    if isinstance(node, (ast.With, ast.AsyncWith)):
        items = ", ".join(ast.unparse(item.context_expr) for item in node.items)
        return f"{type(node).__name__} {items}"
    if isinstance(node, ast.ExceptHandler):
        return "Except " + (ast.unparse(node.type) if node.type else "*")
    if isinstance(node, ast.Match):
        return f"Match {ast.unparse(node.subject)}"
    if isinstance(node, ast.match_case):
        guard = f" if {ast.unparse(node.guard)}" if node.guard else ""
        return f"Case {ast.unparse(node.pattern)}{guard}"
    return type(node).__name__


class SourceIndex:
    """One module's structure by line: enclosing scope, block chain, statements.

    A site's identity is what the code says, so this reads the parse tree rather
    than scanning text. A statement is therefore the whole statement as Python
    understands it - however it is wrapped, however long its message - and the
    `def`/`class`/`if` chain around it ignores blank lines, comments and
    formatting.

    A file that does not parse yields no structure. Its sites still get keys, from
    the text on their own line, so the audit keeps running on a tree that is
    momentarily broken.
    """

    def __init__(self, text: str) -> None:
        self.scopes: dict[int, str] = {}
        self.blocks: dict[int, str] = {}
        self.raises: dict[int, str] = {}
        self.contexts: dict[int, str] = {}
        self.root_context = digest(text)
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return
        self.root_context = digest(ast.dump(tree, include_attributes=False))
        self._walk(tree, "", ())
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                #  Canonical source: reflowing or re-quoting the message is not a
                #  change of identity, a changed message is.
                self.raises.setdefault(node.lineno, ast.unparse(node))

    def scope(self, line: int) -> str:
        """`Class.method` chain containing `line`, outermost first."""
        return self.scopes.get(line, "")

    def block(self, line: int) -> str:
        """The block chain inside that scope, e.g. `If self.flag`."""
        return self.blocks.get(line, "")

    def context(self, line: int) -> str:
        """Semantic enclosing scope, for otherwise indistinguishable duplicates."""
        return self.contexts.get(line, self.root_context)

    def statement(self, line: int) -> str:
        """The `raise` statement starting on `line`, or "" when there is none."""
        return self.raises.get(line, "")

    def _walk(self, node: ast.AST, scope: str, blocks: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, SCOPE_NODES):
                name = f"{scope}.{child.name}" if scope else child.name
                #  From the line after the `def`/`class`, so the header belongs to the
                #  scope around it and comments above the first statement still do not.
                self._fill(self.scopes, child, name, child.lineno + 1)
                self._fill(self.blocks, child, "", child.lineno + 1)   # its own chain
                self._fill(self.contexts, child, digest(ast.dump(child, include_attributes=False)),
                           child.lineno + 1)
                self._walk(child, name, ())
            elif isinstance(child, BLOCK_NODES):
                chain = blocks + (_block_fingerprint(child),)
                self._fill(self.blocks, child, ".".join(chain))
                self._walk(child, scope, chain)
            else:
                self._walk(child, scope, blocks)

    @staticmethod
    def _fill(mapping: dict[int, str], node: ast.AST, value: str,
              start: int | None = None) -> None:
        if isinstance(node, ast.match_case):
            first, last = node.pattern.lineno, node.body[-1].end_lineno
        else:
            first, last = node.lineno, node.end_lineno or node.lineno
        first = first if start is None else start
        for line in range(first, last + 1):
            mapping[line] = value


def site_keys(path: str, index: SourceIndex, sites: list[tuple], *,
              drop: str = "") -> list[str]:
    """One stable key per site in one file: `(line, text)` or `(line, text, label)`.

    The key is the file, the enclosing `Class.method`, and the statement's own
    text. A statement that appears once in its scope is keyed by nothing else, so
    it survives any edit above it and any reformatting of the statement itself.
    A `label` is for two findings that share one line, such as two flags in one
    expression; it keeps them apart without either depending on the other.

    Two identical statements in one scope cannot be told apart by text, so their
    keys also carry the blocks they sit in *and* the group they form. Adding,
    removing or re-contexting a member changes every member's key, which reopens
    the group for review rather than sliding a decision onto a neighbouring site.
    """
    anchors = [(index.scope(site[0]), index.block(site[0]), site[1],
                site[2] if len(site) > 2 else "") for site in sites]
    groups: dict[tuple[str, str, str], list[int]] = {}
    for position, (scope, _block, statement, label) in enumerate(anchors):
        groups.setdefault((scope, statement, label), []).append(position)
    keys = [""] * len(anchors)
    for (scope, statement, label), members in groups.items():
        body = statement.replace(drop, "", 1) if drop else statement
        where = f"{slug(scope, 40)}-" if scope else ""
        base = (f"{path}:{where}{slug(body, 48)}-{digest(scope, statement, label)[:6]}"
                + (f":{label}" if label else ""))
        if len(members) == 1:
            keys[members[0]] = base
            continue
        blocks = [anchors[position][1] for position in members]
        group = digest(*sorted(blocks))[:6]
        if len(set(blocks)) < len(blocks):
            # Source order alone cannot distinguish these sites. Any semantic
            # change to their scope reopens the group rather than moving a call.
            group = digest(group, index.context(sites[members[0]][0]))[:6]
        counted: dict[str, int] = {}
        for position in members:
            key = f"{base}@{group}@{digest(anchors[position][1])[:6]}"
            counted[key] = counted.get(key, 0) + 1
            keys[position] = key if counted[key] == 1 else f"{key}~{counted[key]}"
    return keys


def finding(kind: str, key: str, title: str, location: str, *,
            fix: str, why: str, evidence: dict | None = None) -> Row:
    """A finding always carries the edit that closes it."""
    return Row(
        kind=kind, key=key, title=title, location=location,
        evidence={"fix": fix, "why": why, **(evidence or {})},
        signals=[why],
        hints={"anchor": location.split(":")[0], "refs": [key], "tokens": tokens(title)},
    )


from . import invariants, skeletons, drift  # noqa: E402,F401
