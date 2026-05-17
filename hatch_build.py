"""Custom Hatch build hooks for hipEngine release artifacts."""

from __future__ import annotations

import platform
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Mark release wheels as Linux x86-64 platform wheels.

    hipEngine ships an x86-64 AOTriton shared-library runtime in the package.
    The Python modules are pure, but the artifact is not portable to arbitrary
    platforms, so the wheel must not be tagged ``py3-none-any``. The current
    vendored runtime audits to a glibc 2.39 floor; ROCm libraries remain
    external system dependencies, not bundled wheel payloads.
    """

    def initialize(self, _version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return

        build_data["pure_python"] = False
        build_data["tag"] = _wheel_tag()


def _wheel_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux" or machine not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "hipEngine wheels currently bundle an x86-64 Linux AOTriton runtime; "
            "build the sdist only or add a matching wheel tag/artifact for this platform."
        )
    return "py3-none-manylinux_2_39_x86_64"
