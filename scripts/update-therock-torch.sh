#!/usr/bin/env bash
# Update the active Python environment to a pinned, compatible TheRock ROCm +
# PyTorch nightly stack.  Defaults target gfx1100/W7900 and the multi-arch
# wheel index documented in https://github.com/ROCm/TheRock/blob/main/RELEASES.md.

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

exec "${PYTHON_BIN}" - "$@" <<'PY'
from __future__ import annotations

import argparse
import html
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable

try:
    from packaging.version import Version
except Exception:  # pragma: no cover - packaging is normally present with pip.
    try:
        from pip._vendor.packaging.version import Version  # type: ignore
    except Exception:  # pragma: no cover
        Version = None  # type: ignore

DEFAULT_INDEX_URL = "https://rocm.nightlies.amd.com/whl-multi-arch/"
ROCM_VERSION_RE = re.compile(r"\+rocm(?P<rocm>\d+\.\d+\.\d+a(?P<date>\d{8}))")
ROCM_SDK_VERSION_RE = re.compile(r"^(?P<rocm>\d+\.\d+\.\d+a(?P<date>\d{8}))$")


@dataclass(frozen=True)
class Wheel:
    filename: str
    version: str
    py_tag: str
    abi_tag: str
    platform_tag: str


@dataclass(frozen=True)
class Plan:
    date: str
    rocm_version: str
    torch_version: str
    torchvision_version: str
    torchaudio_version: str | None
    device: str
    python_tag: str
    platform_tag: str
    specs: list[str]


class ResolverError(RuntimeError):
    pass


class SimpleIndex:
    def __init__(self, index_url: str) -> None:
        self.index_url = index_url.rstrip("/") + "/"
        self._cache: dict[str, list[str]] = {}

    def files(self, package: str) -> list[str]:
        normalized = package.lower().replace("_", "-")
        if normalized in self._cache:
            return self._cache[normalized]
        url = urllib.parse.urljoin(self.index_url, normalized + "/")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise ResolverError(f"failed to read {url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ResolverError(f"failed to read {url}: {exc}") from exc
        hrefs = re.findall(r"href=[\"']([^\"']+)[\"']", text, flags=re.IGNORECASE)
        files = []
        for href in hrefs:
            name = urllib.parse.unquote(href.rsplit("/", 1)[-1])
            name = html.unescape(name)
            if name:
                files.append(name)
        self._cache[normalized] = files
        return files

    def wheels(self, package: str) -> list[Wheel]:
        prefix = package.lower().replace("-", "_") + "-"
        out: list[Wheel] = []
        for name in self.files(package):
            if not name.endswith(".whl") or not name.lower().startswith(prefix):
                continue
            rest = name[len(prefix) : -4]
            parts = rest.rsplit("-", 3)
            if len(parts) != 4:
                continue
            version, py_tag, abi_tag, platform_tag = parts
            out.append(Wheel(name, version, py_tag, abi_tag, platform_tag))
        return out

    def wheel_versions(
        self,
        package: str,
        *,
        py_tag: str | None = None,
        abi_tag: str | None = None,
        platform_tag: str | None = None,
    ) -> set[str]:
        versions = set()
        for wheel in self.wheels(package):
            if py_tag is not None and wheel.py_tag != py_tag:
                continue
            if abi_tag is not None and wheel.abi_tag != abi_tag:
                continue
            if platform_tag is not None and wheel.platform_tag != platform_tag:
                continue
            versions.add(wheel.version)
        return versions

    def sdist_versions(self, package: str) -> set[str]:
        prefix = package.lower().replace("-", "_") + "-"
        versions = set()
        for name in self.files(package):
            lower = name.lower()
            if lower.startswith(prefix) and lower.endswith(".tar.gz"):
                versions.add(name[len(prefix) : -len(".tar.gz")])
        return versions


@dataclass(frozen=True)
class TorchCandidate:
    date: str
    rocm_version: str
    torch_version: str
    torch_base: str
    torchvision_version: str
    torchaudio_version: str | None


def version_key(version: str):
    base = version.split("+", 1)[0]
    if Version is not None:
        try:
            return Version(base)
        except Exception:
            pass
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"([0-9]+)", base))


def normalize_device(device: str) -> str:
    device = device.strip().lower()
    if device.startswith("device-"):
        device = device[len("device-") :]
    if not re.fullmatch(r"gfx[0-9a-z]+", device):
        raise ResolverError(f"unsupported device {device!r}; expected e.g. gfx1100")
    return device


def python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def host_platform_tag() -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        return "linux_x86_64"
    raise ResolverError(
        f"this helper currently expects Linux x86_64 TheRock wheels; got sys.platform={sys.platform!r}, machine={machine!r}"
    )


def parse_rocm_from_local_version(version: str) -> tuple[str, str] | None:
    match = ROCM_VERSION_RE.search(version)
    if not match:
        return None
    return match.group("rocm"), match.group("date")


def parse_rocm_sdk_version(version: str) -> tuple[str, str] | None:
    match = ROCM_SDK_VERSION_RE.match(version)
    if not match:
        return None
    return match.group("rocm"), match.group("date")


def torch_parts(torch_base: str) -> tuple[int, int]:
    pieces = torch_base.split(".")
    if len(pieces) < 2:
        raise ResolverError(f"cannot infer torchvision/torchaudio version from torch {torch_base!r}")
    return int(pieces[0]), int(pieces[1])


def expected_torchvision_base(torch_base: str) -> str:
    major, minor = torch_parts(torch_base)
    if major != 2:
        raise ResolverError(f"cannot infer torchvision version for torch {torch_base!r}")
    return f"0.{minor + 15}.0"


def expected_torchaudio_base(torch_base: str) -> str:
    major, minor = torch_parts(torch_base)
    return f"{major}.{minor}.0"


def matching_torchaudio_version(
    versions: Iterable[str], torch_base: str, rocm_version: str
) -> str | None:
    audio_base = expected_torchaudio_base(torch_base)
    suffix = f"+rocm{rocm_version}"
    candidates = [v for v in versions if v.endswith(suffix) and v.split("+", 1)[0].startswith(audio_base)]
    if not candidates:
        return None
    exact = f"{audio_base}{suffix}"
    if exact in candidates:
        return exact
    return sorted(candidates, key=version_key, reverse=True)[0]


def family_device_package(device: str) -> str | None:
    if re.fullmatch(r"gfx11(00|01|02|03|50|51|52|53)", device):
        return "amd-torch-device-gfx11"
    if device in {"gfx1200", "gfx1201"}:
        return "amd-torch-device-gfx12-0"
    return None


def collect_rocm_versions(index: SimpleIndex, device: str, platform_tag: str) -> dict[str, str]:
    rocm_versions = index.sdist_versions("rocm")
    required_wheel_packages = [
        "rocm-sdk-core",
        "rocm-sdk-libraries",
        "rocm-sdk-devel",
        f"rocm-sdk-device-{device}",
    ]
    package_versions = {
        pkg: index.wheel_versions(pkg, py_tag="py3", abi_tag="none", platform_tag=platform_tag)
        for pkg in required_wheel_packages
    }
    by_date: dict[str, str] = {}
    for version in rocm_versions:
        parsed = parse_rocm_sdk_version(version)
        if not parsed:
            continue
        rocm_version, date = parsed
        if all(rocm_version in versions for versions in package_versions.values()):
            by_date[date] = rocm_version
    return by_date


def resolve_plan(args: argparse.Namespace) -> Plan:
    device = normalize_device(args.device)
    py_tag = python_tag()
    plat_tag = host_platform_tag()
    index = SimpleIndex(args.index_url)

    rocm_by_date = collect_rocm_versions(index, device, plat_tag)
    if not rocm_by_date:
        raise ResolverError(f"no complete ROCm SDK stack found for {device} at {args.index_url}")

    torch_versions = index.wheel_versions("torch", py_tag=py_tag, abi_tag=py_tag, platform_tag=plat_tag)
    vision_versions = index.wheel_versions("torchvision", py_tag=py_tag, abi_tag=py_tag, platform_tag=plat_tag)
    audio_versions = set()
    if args.torchaudio:
        audio_versions = index.wheel_versions("torchaudio", py_tag=py_tag, abi_tag=py_tag, platform_tag=plat_tag)
    amd_torch_versions = index.wheel_versions(
        f"amd-torch-device-{device}", py_tag=py_tag, abi_tag=py_tag, platform_tag=plat_tag
    )
    family_pkg = family_device_package(device)
    family_versions = set()
    if family_pkg is not None:
        family_versions = index.wheel_versions(family_pkg, py_tag=py_tag, abi_tag=py_tag, platform_tag=plat_tag)
    amd_vision_versions = index.wheel_versions(
        f"amd-torchvision-device-{device}", py_tag=py_tag, abi_tag=py_tag, platform_tag=plat_tag
    )

    candidates: list[TorchCandidate] = []
    for torch_version in torch_versions:
        parsed = parse_rocm_from_local_version(torch_version)
        if not parsed:
            continue
        rocm_version, date = parsed
        if args.date is not None and date != args.date:
            continue
        if date not in rocm_by_date or rocm_by_date[date] != rocm_version:
            continue
        torch_base = torch_version.split("+", 1)[0]
        if args.torch_version != "auto":
            wanted = args.torch_version
            if torch_base != wanted and not torch_base.startswith(wanted + "."):
                continue
        if torch_version not in amd_torch_versions:
            continue
        if family_pkg is not None and torch_version not in family_versions:
            continue
        vision_base = expected_torchvision_base(torch_base)
        vision_version = f"{vision_base}+rocm{rocm_version}"
        if vision_version not in vision_versions or vision_version not in amd_vision_versions:
            continue
        audio_version = None
        if args.torchaudio:
            audio_version = matching_torchaudio_version(audio_versions, torch_base, rocm_version)
            if audio_version is None:
                continue
        candidates.append(
            TorchCandidate(
                date=date,
                rocm_version=rocm_version,
                torch_version=torch_version,
                torch_base=torch_base,
                torchvision_version=vision_version,
                torchaudio_version=audio_version,
            )
        )

    if not candidates:
        hint = ""
        if args.torchaudio:
            hint = " Try --no-torchaudio if you want the newest torch/torchvision pair even when torchaudio lags."
        if args.date is not None:
            hint += " Try omitting --date to let the script choose the latest complete nightly."
        raise ResolverError(f"no compatible TheRock torch stack found for {device}, {py_tag}, {plat_tag}.{hint}")

    candidates.sort(key=lambda c: (int(c.date), version_key(c.torch_base)), reverse=True)
    chosen = candidates[0]
    rocm_version = chosen.rocm_version
    specs = [
        f"rocm=={rocm_version}",
        f"rocm-sdk-core=={rocm_version}",
        f"rocm-sdk-libraries=={rocm_version}",
        f"rocm-sdk-devel=={rocm_version}",
        f"rocm-sdk-device-{device}=={rocm_version}",
        f"torch[device-{device}]=={chosen.torch_version}",
        f"torchvision[device-{device}]=={chosen.torchvision_version}",
    ]
    if chosen.torchaudio_version is not None:
        specs.append(f"torchaudio=={chosen.torchaudio_version}")
    return Plan(
        date=chosen.date,
        rocm_version=rocm_version,
        torch_version=chosen.torch_version,
        torchvision_version=chosen.torchvision_version,
        torchaudio_version=chosen.torchaudio_version,
        device=device,
        python_tag=py_tag,
        platform_tag=plat_tag,
        specs=specs,
    )


def print_plan(args: argparse.Namespace, plan: Plan, pip_cmd: list[str]) -> None:
    summary = {
        "index_url": args.index_url,
        "python": sys.executable,
        "python_tag": plan.python_tag,
        "platform_tag": plan.platform_tag,
        "device": plan.device,
        "nightly_date": plan.date,
        "rocm": plan.rocm_version,
        "torch": plan.torch_version,
        "torchvision": plan.torchvision_version,
        "torchaudio": plan.torchaudio_version,
        "pip_command": pip_cmd,
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return
    print("TheRock ROCm/PyTorch update plan")
    print(f"  index:        {args.index_url}")
    print(f"  python:       {sys.executable} ({plan.python_tag})")
    print(f"  platform:     {plan.platform_tag}")
    print(f"  device:       {plan.device}")
    print(f"  nightly date: {plan.date}")
    print(f"  ROCm SDK:     {plan.rocm_version}")
    print(f"  torch:        {plan.torch_version}")
    print(f"  torchvision:  {plan.torchvision_version}")
    print(f"  torchaudio:   {plan.torchaudio_version or 'disabled'}")
    print("\nPinned package specs:")
    for spec in plan.specs:
        print(f"  {spec}")
    print("\nPip command:")
    print("  " + " ".join(sh_quote(x) for x in pip_cmd))


def sh_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+@%-]+", value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(sh_quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/update-therock-torch.sh",
        description=(
            "Update the active Python environment to a pinned TheRock multi-arch "
            "ROCm SDK + PyTorch nightly stack. By default the latest complete "
            "nightly date is discovered from the package index."
        )
    )
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL, help=f"TheRock index URL (default: {DEFAULT_INDEX_URL})")
    parser.add_argument("--device", default="gfx1100", help="GPU target, e.g. gfx1100 or device-gfx1100 (default: gfx1100)")
    parser.add_argument("--date", help="Require a specific nightly date YYYYMMDD instead of auto-discovering latest")
    parser.add_argument(
        "--torch-version",
        default="auto",
        help="Torch base version to require, e.g. 2.11.0 or 2.12.0 (default: auto/latest compatible)",
    )
    parser.add_argument("--no-torchaudio", dest="torchaudio", action="store_false", help="Do not require/install torchaudio")
    parser.set_defaults(torchaudio=True)
    parser.add_argument("--print-plan", action="store_true", help="Only resolve and print the plan; do not invoke pip")
    parser.add_argument("--json", action="store_true", help="Print the resolved plan as JSON")
    parser.add_argument("--pip-dry-run", action="store_true", help="Run pip install --dry-run after resolving pins")
    parser.add_argument("--use-pip-cache", action="store_true", help="Allow pip to cache huge wheels (default: pass --no-cache-dir)")
    parser.add_argument("--skip-init", action="store_true", help="Skip rocm-sdk init after a successful install")
    parser.add_argument("--test", action="store_true", help="Run rocm-sdk test after init")
    parser.add_argument("--verify-torch", action="store_true", help="Import torch and print ROCm device availability after install")
    parser.add_argument(
        "--extra-pip-arg",
        action="append",
        default=[],
        help="Append an extra argument to pip install (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.date is not None and not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must be YYYYMMDD")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        plan = resolve_plan(args)
    except ResolverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pip_cmd = [sys.executable, "-m", "pip", "install", "--index-url", args.index_url, "-U"]
    if args.pip_dry_run:
        pip_cmd.append("--dry-run")
    if not args.use_pip_cache:
        pip_cmd.append("--no-cache-dir")
    pip_cmd.extend(args.extra_pip_arg)
    pip_cmd.extend(plan.specs)

    print_plan(args, plan, pip_cmd)
    if args.print_plan:
        return 0

    run(pip_cmd)
    if args.pip_dry_run:
        return 0

    if not args.skip_init:
        rocm_sdk = shutil.which("rocm-sdk")
        if rocm_sdk is not None:
            run([rocm_sdk, "init"])
            if args.test:
                run([rocm_sdk, "test"])
        else:
            run([sys.executable, "-m", "rocm_sdk", "init"])
            if args.test:
                run([sys.executable, "-m", "rocm_sdk", "test"])

    if args.verify_torch:
        code = """
import torch
print('torch', torch.__version__)
print('torch.version.hip', torch.version.hip)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device 0', torch.cuda.get_device_name(0))
""".strip()
        run([sys.executable, "-c", code])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
PY
