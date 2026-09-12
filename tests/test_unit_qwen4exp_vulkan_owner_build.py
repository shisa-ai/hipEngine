from pathlib import Path

import pytest

from scripts.qwen4exp_vulkan_owner_build import (
    compile_command,
    instrument_model,
    link_command,
    replace_once,
)


def test_compile_and_link_never_write_reference_outputs():
    record = {
        "file": "/reference/model.cpp",
        "command": "c++ -I/reference -o CMakeFiles/model.o -c /reference/model.cpp",
    }
    cmd = compile_command(record, Path("/owned/model.cpp"), Path("/owned/model.o"))
    assert cmd[cmd.index("-o") + 1] == "/owned/model.o"
    assert cmd[cmd.index("-c") + 1] == "/owned/model.cpp"
    linked = link_command(
        "c++ -shared -Wl,--dependency-file=reference.d -o original.so original.o shaders.o",
        "original.o",
        Path("/owned/model.o"),
        Path("/owned/library.so"),
    )
    assert linked == ["c++", "-shared", "-o", "/owned/library.so", "/owned/model.o", "shaders.o"]


def test_anchors_fail_closed():
    with pytest.raises(ValueError, match="once"):
        replace_once("x x", "x", "y")
    with pytest.raises(ValueError, match="anchor"):
        instrument_model("unrecognized source")
    with pytest.raises(ValueError, match="dependency"):
        compile_command(
            {"file": "a.cpp", "command": "c++ -MMD -o a.o -c a.cpp"}, Path("b.cpp"), Path("b.o")
        )
