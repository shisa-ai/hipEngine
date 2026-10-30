"""Bounded source-literal reader for backend kernel package declarations.

Cold GGUF admission and the quant-route audit read backend capability
declarations out of kernel package source. Neither may import the backend
package (registration needs the HIP toolchain) and neither may evaluate
arbitrary code. This pins the reader contract the RF-8 policy split relies on:
literal and container-wrapper values, explicit same-package import aliases and
re-exports, ordered local literal aliases, and fail-closed rejection of
import cycles, external imports, path escapes, missing modules, and arbitrary
calls.

The gfx1151 package declares its policy owners in ``POLICY_MODULE_NAMES``;
the root read merges those owners' literal declarations, so a name the
initializer no longer re-binds still reads as its owner's value. A literal-only
reader that ignored the owner declaration would report an empty
``GGUF_CONSUMER_LAYERS`` and make public GGUF construction refuse the backend.
The live-declaration tests below are the regression guard; the exact certified
layer set stays owned by the consumer-surface parity test.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from hipengine.loading.backend_source_literals import (
    LITERAL_STATUS,
    NONLITERAL_STATUS,
    read_backend_assignments,
)
from hipengine.loading.qwen35_gguf_consumer_surface import (
    GGUF_CONSUMER_LAYER_DECLARATION_NAME,
    backend_gguf_consumer_layers,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _backend_package(
    root: pathlib.Path, backend: str, files: dict[str, str]
) -> pathlib.Path:
    """Write a synthetic backend package; ``""`` is its ``__init__.py``."""

    package = root / "hipengine" / "kernels" / backend
    package.mkdir(parents=True)
    for name, source in files.items():
        path = package / "__init__.py" if name == "" else package / f"{name}.py"
        path.write_text(source, encoding="utf-8")
    return package


def test_reader_resolves_literals_annotations_and_container_wrappers(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "": (
                "FLAG = True\n"
                "COUNT: int = 7\n"
                "NAMES = (\"linear\", \"rmsnorm\")\n"
                "LAYERS: frozenset[str] = frozenset({\"linear\", \"rmsnorm\"})\n"
                "EMPTY_LAYERS: frozenset[str] = frozenset()\n"
                "NOT_A_LITERAL = True if FLAG else False\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    assert (assignments["FLAG"].value, assignments["FLAG"].status) == (
        True,
        LITERAL_STATUS,
    )
    assert (assignments["COUNT"].value, assignments["COUNT"].status) == (
        7,
        LITERAL_STATUS,
    )
    assert assignments["NAMES"].value == ("linear", "rmsnorm")
    assert assignments["LAYERS"].value == frozenset({"linear", "rmsnorm"})
    assert assignments["EMPTY_LAYERS"].value == frozenset()
    assert assignments["EMPTY_LAYERS"].status == LITERAL_STATUS
    # An expression is unreadable, never guessed.
    assert assignments["NOT_A_LITERAL"].status == NONLITERAL_STATUS
    assert assignments["NOT_A_LITERAL"].value is None


def test_reader_follows_explicit_same_package_import_aliases(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "policy_data": (
                "CONSUMER_LAYERS: frozenset[str] = "
                "frozenset({\"linear\", \"embedding\"})\n"
                "SHAPES = ((64, 32),)\n"
            ),
            "": (
                "from hipengine.kernels.hip_demo import policy_data as _policy_data\n"
                "from .policy_data import SHAPES as _SHAPES\n"
                "import hipengine.kernels.hip_demo.policy_data as _absolute_data\n"
                "CONSUMER_LAYERS = _policy_data.CONSUMER_LAYERS\n"
                "SHAPES = _SHAPES\n"
                "ABSOLUTE_LAYERS = _absolute_data.CONSUMER_LAYERS\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    for name in ("CONSUMER_LAYERS", "ABSOLUTE_LAYERS"):
        assert assignments[name].value == frozenset({"linear", "embedding"})
        assert assignments[name].status == LITERAL_STATUS
    assert assignments["SHAPES"].value == ((64, 32),)
    assert assignments["SHAPES"].status == LITERAL_STATUS


def test_reader_resolves_ordered_local_literal_aliases(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "": (
                "BASE_SHAPES = ((64, 32),)\n"
                "ALIAS_SHAPES = BASE_SHAPES\n"
                "ANNOTATED_ALIAS: tuple = ALIAS_SHAPES\n"
                "FORWARD_ALIAS = LATER\n"
                "LATER = 1\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    assert assignments["ALIAS_SHAPES"].value == ((64, 32),)
    assert assignments["ANNOTATED_ALIAS"].value == ((64, 32),)
    assert assignments["ANNOTATED_ALIAS"].status == LITERAL_STATUS
    # Ordered: a forward local reference cannot execute at module import time.
    assert assignments["FORWARD_ALIAS"].status == NONLITERAL_STATUS


def test_reader_treats_a_rebound_name_as_a_local_value(tmp_path):
    # Sequential module execution: a name rebound after its import is no
    # longer the imported module or constant.
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "policy_data": "LAYERS = frozenset({\"linear\"})\n",
            "": (
                "from . import policy_data as _data\n"
                "_data = 1\n"
                "REBOUND_MODULE = _data.anything\n"
                "from .policy_data import LAYERS\n"
                "LAYERS = (\"local\",)\n"
                "AFTER_REBIND = LAYERS\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    assert assignments["REBOUND_MODULE"].status == NONLITERAL_STATUS
    assert assignments["AFTER_REBIND"].value == ("local",)
    assert assignments["AFTER_REBIND"].status == LITERAL_STATUS


def test_reader_reports_direct_reexports_without_a_local_alias(tmp_path):
    # ``from .policy import VALUE as VALUE`` is a re-export: VALUE is a
    # module-level binding of this module, not just an import table entry.
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "policy": (
                "CONSUMER_LAYERS = frozenset({\"linear\"})\n"
                "SHAPES = ((64, 32),)\n"
            ),
            "": (
                "from .policy import CONSUMER_LAYERS as CONSUMER_LAYERS\n"
                "from .policy import SHAPES\n"
                "from . import policy as _policy\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    assert assignments["CONSUMER_LAYERS"].value == frozenset({"linear"})
    assert assignments["CONSUMER_LAYERS"].status == LITERAL_STATUS
    assert assignments["SHAPES"].value == ((64, 32),)
    assert assignments["SHAPES"].status == LITERAL_STATUS
    # A module alias is a binding, but a module object is not a literal.
    assert assignments["_policy"].status == NONLITERAL_STATUS


def test_imports_invalidate_earlier_literal_bindings(tmp_path):
    # Source order decides: a later import rebinds the name, so an earlier
    # literal (or an earlier same-package module alias) is stale. This holds
    # for rejected/external imports too, which never certify the old value.
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "policy_data": "CONSUMER_LAYERS = frozenset({\"linear\"})\n",
            "": (
                "VALUE = 1\n"
                "from os import sep as VALUE\n"
                "COPY = VALUE\n"
                "LAYERS = frozenset({\"linear\"})\n"
                "import os as LAYERS\n"
                "COPY_LAYERS = LAYERS\n"
                "from . import policy_data as _data\n"
                "MODULE_COPY = _data.CONSUMER_LAYERS\n"
                "import os as _data\n"
                "AFTER_REBIND = _data.CONSUMER_LAYERS\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    for name in ("VALUE", "COPY", "LAYERS", "COPY_LAYERS"):
        assert assignments[name].status == NONLITERAL_STATUS, name
        assert assignments[name].value is None, name
    # The accepted module alias was usable before it was rebound externally.
    assert assignments["MODULE_COPY"].value == frozenset({"linear"})
    assert assignments["AFTER_REBIND"].status == NONLITERAL_STATUS
    assert assignments["AFTER_REBIND"].value is None


def test_reader_rejects_malformed_and_shadowed_container_wrappers(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "": (
                "BAD_ITERABLE = frozenset(1)\n"
                "BAD_MEMBER = set({[]})\n"
                "BAD_TUPLE = tuple(1)\n"
                "frozenset = build_wrapper()\n"
                "SHADOWED = frozenset({\"linear\"})\n"
                "GOOD_AFTER_SHADOW = set({\"linear\"})\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    for name in ("BAD_ITERABLE", "BAD_MEMBER", "BAD_TUPLE", "SHADOWED"):
        assert assignments[name].status == NONLITERAL_STATUS, name
        assert assignments[name].value is None, name
    # Shadowing is per name and ordered: ``set`` is still the builtin here.
    assert assignments["GOOD_AFTER_SHADOW"].value == {"linear"}
    assert assignments["GOOD_AFTER_SHADOW"].status == LITERAL_STATUS


def test_reader_invalidates_containers_mutated_after_assignment(tmp_path):
    # A module-level mutation makes the earlier literal stale; the name must
    # not keep certifying the pre-mutation value.
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "": (
                "NAMES = []\n"
                "NAMES.append(\"linear\")\n"
                "MUTATED = NAMES\n"
                "BARE_METHOD = NAMES.append\n"
                "LAYERS = frozenset({\"linear\"})\n"
                "LAYERS.add(\"rmsnorm\")\n"
                "MUTATED_LAYERS = LAYERS\n"
                "SHAPES = [[64, 32]]\n"
                "SHAPES[0][0] = 1\n"
                "MUTATED_SHAPES = SHAPES\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    for name in (
        "NAMES",
        "MUTATED",
        "BARE_METHOD",
        "LAYERS",
        "MUTATED_LAYERS",
        "SHAPES",
        "MUTATED_SHAPES",
    ):
        assert assignments[name].status == NONLITERAL_STATUS, name


def test_reader_rejects_import_and_local_alias_cycles(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "a": "from . import b as _b\nVALUE = _b.VALUE\n",
            "b": "from . import a as _a\nVALUE = _a.VALUE\n",
            "": (
                "from . import a as _a\n"
                "VALUE = _a.VALUE\n"
                "SELF = SELF\n"
            ),
        },
    )

    # Terminates, and every cycle-dependent declaration is unreadable.
    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    assert assignments["VALUE"].status == NONLITERAL_STATUS
    assert assignments["VALUE"].value is None
    assert assignments["SELF"].status == NONLITERAL_STATUS


def test_reader_rejects_external_imports_calls_and_missing_modules(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "": (
                "import os as _os\n"
                "from hipengine.kernels import backends as _backends\n"
                "from hipengine.kernels.hip_peer import policy as _peer\n"
                "from . import missing_module as _missing\n"
                "EXTERNAL = _os.sep\n"
                "EXTERNAL_FROM = _backends.SOMETHING\n"
                "PEER = _peer.VALUE\n"
                "MISSING = _missing.VALUE\n"
                "CALL = sorted({3, 1})\n"
                "BINOP = frozenset({\"a\"} | {\"b\"})\n"
                "COMPREHENSION = tuple(name for name in ())\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    for name in (
        "EXTERNAL",
        "EXTERNAL_FROM",
        "PEER",
        "MISSING",
        "CALL",
        "BINOP",
        "COMPREHENSION",
    ):
        assert assignments[name].status == NONLITERAL_STATUS, name
        assert assignments[name].value is None, name
    assert read_backend_assignments("hip_absent", root=tmp_path) == {}


def test_reader_reports_unreadable_sources_without_raising(tmp_path):
    _backend_package(tmp_path, "hip_demo", {"": "BROKEN = (\n"})
    assert read_backend_assignments("hip_demo", root=tmp_path) == {}


@pytest.mark.parametrize(
    "backend", ["../hip_peer", "hip_demo/../hip_peer", "..", "", "hip.demo", "hip demo"]
)
def test_reader_never_escapes_the_backend_package(tmp_path, backend):
    _backend_package(tmp_path, "hip_demo", {"": "VALUE = 1\n"})
    assert read_backend_assignments(backend, root=tmp_path) == {}


def test_reader_matches_the_live_backend_consumer_declarations():
    # The gfx1151 declaration moved into policy_capability and is re-exported
    # from the package initializer as an import alias. gfx1100 still declares
    # it locally. Both must read as literals; the exact certified set is bound
    # by the consumer-surface parity test, so this only pins the role names.
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        entry = read_backend_assignments(backend, root=ROOT)[
            GGUF_CONSUMER_LAYER_DECLARATION_NAME
        ]
        assert entry.status == LITERAL_STATUS, backend
        assert isinstance(entry.value, frozenset), backend
        assert {"linear", "dense_gemv", "embedding", "rmsnorm"} <= entry.value


def test_consumer_surface_reader_follows_policy_module_aliases(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "policy_capability": (
                "GGUF_CONSUMER_LAYERS: frozenset[str] = "
                "frozenset({\"linear\", \"rmsnorm\"})\n"
            ),
            "": (
                "from hipengine.kernels.hip_demo import policy_capability "
                "as _policy_capability\n"
                "GGUF_CONSUMER_LAYERS = _policy_capability.GGUF_CONSUMER_LAYERS\n"
            ),
        },
    )
    assert backend_gguf_consumer_layers("hip_demo", root=tmp_path) == frozenset(
        {"linear", "rmsnorm"}
    )


def test_consumer_surface_reader_fails_closed_on_unreadable_declarations(tmp_path):
    cases = {
        "cycle": {
            "a": (
                "from . import b as _b\n"
                "GGUF_CONSUMER_LAYERS = _b.GGUF_CONSUMER_LAYERS\n"
            ),
            "b": (
                "from . import a as _a\n"
                "GGUF_CONSUMER_LAYERS = _a.GGUF_CONSUMER_LAYERS\n"
            ),
            "": (
                "from . import a as _a\n"
                "GGUF_CONSUMER_LAYERS = _a.GGUF_CONSUMER_LAYERS\n"
            ),
        },
        "call": {"": "GGUF_CONSUMER_LAYERS = compute_layers()\n"},
        "wrong_type": {"": "GGUF_CONSUMER_LAYERS = (\"linear\",)\n"},
        "non_string_items": {"": "GGUF_CONSUMER_LAYERS = frozenset({1, 2})\n"},
        "absent": {"": "OTHER = frozenset({\"linear\"})\n"},
    }
    for case, files in cases.items():
        backend = f"hip_{case}"
        _backend_package(tmp_path, backend, files)
        assert backend_gguf_consumer_layers(backend, root=tmp_path) == frozenset(), case


_COLD_DRIVER = """\
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
sys.path.insert(0, str(repo_root))

blocked = ("hipengine.kernels.hip_", "hipengine.kernels.cuda_", "torch")


class BackendImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(blocked):
            raise ImportError("blocked backend import: " + fullname)
        return None


sys.meta_path.insert(0, BackendImportGuard())

from hipengine.loading.backend_source_literals import read_backend_assignments
from hipengine.loading.qwen35_gguf_consumer_surface import (
    backend_gguf_consumer_layers,
)

entry = read_backend_assignments("hip_gfx1151")["GGUF_CONSUMER_LAYERS"]
print(json.dumps({
    "status": entry.status,
    "layers": sorted(entry.value or ()),
    "consumer_layers": sorted(backend_gguf_consumer_layers("hip_gfx1151")),
    "loaded_blocked": sorted(
        name for name in sys.modules if name.startswith(blocked)
    ),
}))
"""


def test_reader_merges_constants_owned_by_the_declared_policy_modules(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "policy_capability": (
                "GGUF_CONSUMER_LAYERS: frozenset[str] = "
                "frozenset({\"linear\", \"rmsnorm\"})\n"
                "_PRIVATE_FLAG = True\n"
                "helper = 1\n"
            ),
            "policy_gguf": (
                "SHAPES = ((64, 32),)\n"
                "GGUF_CONSUMER_LAYERS = frozenset({\"embedding\"})\n"
            ),
            "": (
                "POLICY_MODULE_NAMES = (\"policy_capability\", \"policy_gguf\")\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    assert assignments["GGUF_CONSUMER_LAYERS"].value == frozenset(
        {"linear", "rmsnorm"}
    )
    assert assignments["GGUF_CONSUMER_LAYERS"].status == LITERAL_STATUS
    assert assignments["SHAPES"].value == ((64, 32),)
    assert assignments["_PRIVATE_FLAG"].value is True
    # Owners contribute their uppercase declarations only; a lower-case helper
    # is module-internal and never becomes a package capability.
    assert "helper" not in assignments
    # Ownership is ordered: the first declared owner wins a shared name.
    assert backend_gguf_consumer_layers("hip_demo", root=tmp_path) == frozenset(
        {"linear", "rmsnorm"}
    )


def test_reader_package_binding_wins_over_the_declared_owner(tmp_path):
    _backend_package(
        tmp_path,
        "hip_demo",
        {
            "policy_capability": (
                "GGUF_CONSUMER_LAYERS = frozenset({\"linear\", \"rmsnorm\"})\n"
                "SHAPES = ((64, 32),)\n"
            ),
            "": (
                "POLICY_MODULE_NAMES = (\"policy_capability\",)\n"
                "GGUF_CONSUMER_LAYERS = compute_layers()\n"
                "SHAPES = ((8, 8),)\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_demo", root=tmp_path)

    # The package's own binding wins even when it is nonliteral: the owner
    # literal is never promoted past a package declaration the reader cannot
    # certify. Admission then fails closed exactly as for an absent name.
    assert assignments["GGUF_CONSUMER_LAYERS"].status == NONLITERAL_STATUS
    assert backend_gguf_consumer_layers("hip_demo", root=tmp_path) == frozenset()
    # A literal package binding likewise beats a different owner literal.
    assert assignments["SHAPES"].value == ((8, 8),)


def test_reader_fails_closed_on_malformed_owner_declarations(tmp_path):
    owner = "GGUF_CONSUMER_LAYERS = frozenset({\"linear\"})\n"
    cases = {
        "absent": "OTHER = 1\n",
        "list": "POLICY_MODULE_NAMES = [\"policy_data\"]\n",
        "string": "POLICY_MODULE_NAMES = \"policy_data\"\n",
        "non_string_item": "POLICY_MODULE_NAMES = (\"policy_data\", 3)\n",
        "path_escape": "POLICY_MODULE_NAMES = (\"../policy_data\",)\n",
        "nonliteral": "POLICY_MODULE_NAMES = owners()\n",
        "missing_module": "POLICY_MODULE_NAMES = (\"policy_missing\",)\n",
    }
    for case, initializer in cases.items():
        backend = f"hip_{case}"
        _backend_package(
            tmp_path, backend, {"policy_data": owner, "": initializer}
        )
        assignments = read_backend_assignments(backend, root=tmp_path)
        assert backend_gguf_consumer_layers(backend, root=tmp_path) == frozenset(), case
        if case == "absent":
            assert "POLICY_MODULE_NAMES" not in assignments, case
            continue
        # A well-formed declaration is still reported as the literal it is;
        # only the merge it would authorize is withheld.
        assert "POLICY_MODULE_NAMES" in assignments, case
        if case == "missing_module":
            assert assignments["POLICY_MODULE_NAMES"].is_literal, case


def test_reader_stays_bounded_on_cyclic_owner_declarations(tmp_path):
    _backend_package(
        tmp_path,
        "hip_cycle",
        {
            "policy_a": (
                "from . import policy_b as _b\n"
                "GGUF_CONSUMER_LAYERS = _b.GGUF_CONSUMER_LAYERS\n"
            ),
            "policy_b": (
                "from . import policy_a as _a\n"
                "GGUF_CONSUMER_LAYERS = _a.GGUF_CONSUMER_LAYERS\n"
            ),
            "": "POLICY_MODULE_NAMES = (\"policy_a\", \"policy_b\")\n",
        },
    )
    assert backend_gguf_consumer_layers("hip_cycle", root=tmp_path) == frozenset()

    # An owner that imports the package declaration is the same cycle seen from
    # the other side: bounded, fail-closed for the cycle, and the owner's own
    # literal declarations still merge.
    _backend_package(
        tmp_path,
        "hip_self",
        {
            "policy_a": (
                "from hipengine.kernels.hip_self import POLICY_MODULE_NAMES as _NAMES\n"
                "SHAPES = ((64, 32),)\n"
            ),
            "": "POLICY_MODULE_NAMES = (\"policy_a\",)\n",
        },
    )
    assignments = read_backend_assignments("hip_self", root=tmp_path)
    assert assignments["SHAPES"].value == ((64, 32),)
    assert assignments["SHAPES"].status == LITERAL_STATUS
    assert assignments["POLICY_MODULE_NAMES"].value == ("policy_a",)


def test_reader_keeps_duplicate_ordered_owner_names_bounded(tmp_path):
    _backend_package(
        tmp_path,
        "hip_dup",
        {
            "policy_data": "GGUF_CONSUMER_LAYERS = frozenset({\"linear\"})\n",
            "": "POLICY_MODULE_NAMES = (\"policy_data\", \"policy_data\")\n",
        },
    )

    assignments = read_backend_assignments("hip_dup", root=tmp_path)

    # A repeated owner is read once per declaration position and the first
    # merge wins, so a duplicate cannot flip a name to a later value.
    assert assignments["POLICY_MODULE_NAMES"].value == (
        "policy_data",
        "policy_data",
    )
    assert assignments["GGUF_CONSUMER_LAYERS"].value == frozenset({"linear"})
    assert assignments["GGUF_CONSUMER_LAYERS"].status == LITERAL_STATUS
    assert backend_gguf_consumer_layers("hip_dup", root=tmp_path) == frozenset(
        {"linear"}
    )


def test_reader_skips_an_unreadable_declared_owner(tmp_path):
    _backend_package(
        tmp_path,
        "hip_broken",
        {
            "policy_broken": "def broken(:\n",
            "policy_ok": "SHAPES = ((64, 32),)\n",
            "": (
                "POLICY_MODULE_NAMES = "
                "(\"policy_broken\", \"policy_missing\", \"policy_ok\")\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_broken", root=tmp_path)

    # An unparseable or absent owner contributes nothing and does not stop the
    # remaining owners from merging.
    assert assignments["SHAPES"].value == ((64, 32),)
    assert assignments["SHAPES"].status == LITERAL_STATUS
    assert backend_gguf_consumer_layers("hip_broken", root=tmp_path) == frozenset()


def test_reader_package_none_and_false_bindings_win_over_owner_literals(tmp_path):
    _backend_package(
        tmp_path,
        "hip_falsy",
        {
            "policy_data": "FLAG = True\nDISABLED = True\n",
            "": (
                "POLICY_MODULE_NAMES = (\"policy_data\",)\n"
                "FLAG = None\n"
                "DISABLED = False\n"
            ),
        },
    )

    assignments = read_backend_assignments("hip_falsy", root=tmp_path)

    # An explicit None/False package binding is a real declaration, not an
    # absent one: the owner's truthy literal must not replace it.
    assert assignments["FLAG"].value is None
    assert assignments["FLAG"].status == LITERAL_STATUS
    assert assignments["DISABLED"].value is False
    assert assignments["DISABLED"].status == LITERAL_STATUS


def test_cold_reader_adds_no_backend_package_import():
    # Startup isolation: a fresh interpreter that forbids every GPU backend
    # package (and torch) before any import must still read the declaration.
    completed = subprocess.run(
        [sys.executable, "-c", _COLD_DRIVER, str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    payload = json.loads(completed.stdout)
    assert payload["status"] == LITERAL_STATUS
    assert payload["layers"] == payload["consumer_layers"]
    assert payload["layers"], "gfx1151 must declare its GGUF consumer layers"
    assert payload["loaded_blocked"] == []
