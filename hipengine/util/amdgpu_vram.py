"""Poll amdgpu VRAM usage from sysfs.

The amdgpu kernel driver exposes byte-accurate VRAM accounting under
``/sys/class/drm/card*/device/mem_info_vram_*``. Every userspace backend that
allocates VRAM through the kernel driver (HIP, Vulkan, OpenCL) shows up here
identically, so this single source of truth lets us compare hipEngine,
llama.cpp HIP, llama.cpp Vulkan, etc. on the same scale without needing a HIP
context or a per-backend hook.

The reads are microsecond-cheap (no syscall fanout, just one short file
read), so :class:`VramSampler` can poll down to ~1 ms intervals without
measurable perturbation. We intentionally do not try to attribute usage to a
specific process — the value is whole-card committed VRAM, which is what we
care about for "peak GiB" benchmark rows.

Example::

    from hipengine.util.amdgpu_vram import VramSampler, select_card

    card = select_card()  # auto-pick the only amdgpu card
    with VramSampler(card, interval_ms=10) as sampler:
        run_my_benchmark()
    result = sampler.result()
    print(f"peak {result.peak_gib:.3f} GiB (delta {result.peak_delta_gib:.3f})")
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DRM_ROOT = Path("/sys/class/drm")
_GIB = 1 << 30


@dataclass(frozen=True)
class AmdgpuCard:
    """One amdgpu DRM card that exposes VRAM accounting."""

    card_name: str  # e.g. "card1"
    pci_id: str  # e.g. "0000:c3:00.0"
    sysfs_path: Path
    vram_total_bytes: int

    @property
    def vram_used_path(self) -> Path:
        return self.sysfs_path / "mem_info_vram_used"

    @property
    def vram_total_path(self) -> Path:
        return self.sysfs_path / "mem_info_vram_total"

    @property
    def vram_total_gib(self) -> float:
        return self.vram_total_bytes / _GIB

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_name": self.card_name,
            "pci_id": self.pci_id,
            "sysfs_path": str(self.sysfs_path),
            "vram_total_bytes": self.vram_total_bytes,
            "vram_total_gib": self.vram_total_gib,
        }


def list_amdgpu_cards(root: Path | str = _DRM_ROOT) -> list[AmdgpuCard]:
    """Enumerate amdgpu cards that expose ``mem_info_vram_total`` in sysfs.

    Cards without VRAM accounting (e.g. EPYC iGPU stubs, display-only DRM
    nodes) are filtered out. The result is sorted by card name so calls are
    deterministic.
    """

    cards: list[AmdgpuCard] = []
    root_path = Path(root)
    if not root_path.is_dir():
        return cards
    for entry in sorted(root_path.glob("card*")):
        # Skip connector / output nodes like "card1-DP-2".
        if "-" in entry.name:
            continue
        device = entry / "device"
        total_path = device / "mem_info_vram_total"
        if not total_path.exists():
            continue
        try:
            total = int(total_path.read_text().strip())
        except (OSError, ValueError):
            continue
        try:
            pci_id = os.path.basename(os.readlink(device))
        except OSError:
            pci_id = ""
        cards.append(
            AmdgpuCard(
                card_name=entry.name,
                pci_id=pci_id,
                sysfs_path=device,
                vram_total_bytes=total,
            )
        )
    return cards


def select_card(
    *,
    card_name: str | None = None,
    pci_id: str | None = None,
    index: int | None = None,
) -> AmdgpuCard:
    """Pick one amdgpu card by name, PCI id, or zero-based enumeration index.

    With no filter, returns the only enumerated card; raises if there are
    multiple cards present and no disambiguator was supplied.
    """

    cards = list_amdgpu_cards()
    if not cards:
        raise RuntimeError(f"no amdgpu cards with VRAM accounting under {_DRM_ROOT}")
    if card_name is not None:
        for card in cards:
            if card.card_name == card_name:
                return card
        names = [c.card_name for c in cards]
        raise KeyError(f"no amdgpu card named {card_name!r}; have {names}")
    if pci_id is not None:
        for card in cards:
            if card.pci_id == pci_id:
                return card
        ids = [c.pci_id for c in cards]
        raise KeyError(f"no amdgpu card with pci_id {pci_id!r}; have {ids}")
    if index is not None:
        if index < 0 or index >= len(cards):
            raise IndexError(f"card index {index} out of range; {len(cards)} cards present")
        return cards[index]
    if len(cards) != 1:
        names = [c.card_name for c in cards]
        raise RuntimeError(
            f"multiple amdgpu cards present {names}; pass card_name/pci_id/index"
        )
    return cards[0]


def read_vram_used(card: AmdgpuCard) -> int:
    """Read current VRAM-used bytes for one card."""

    return int(card.vram_used_path.read_text().strip())


@dataclass
class VramSamples:
    """Result snapshot from a :class:`VramSampler` run."""

    card: AmdgpuCard
    baseline_bytes: int
    peak_bytes: int
    final_bytes: int
    samples_count: int
    interval_seconds: float
    elapsed_seconds: float
    samples: list[tuple[float, int]] = field(default_factory=list)

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / _GIB

    @property
    def baseline_gib(self) -> float:
        return self.baseline_bytes / _GIB

    @property
    def final_gib(self) -> float:
        return self.final_bytes / _GIB

    @property
    def peak_delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)

    @property
    def peak_delta_gib(self) -> float:
        return self.peak_delta_bytes / _GIB

    def to_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "card": self.card.to_dict(),
            "baseline_bytes": self.baseline_bytes,
            "baseline_gib": self.baseline_gib,
            "peak_bytes": self.peak_bytes,
            "peak_gib": self.peak_gib,
            "peak_delta_bytes": self.peak_delta_bytes,
            "peak_delta_gib": self.peak_delta_gib,
            "final_bytes": self.final_bytes,
            "final_gib": self.final_gib,
            "samples_count": self.samples_count,
            "interval_seconds": self.interval_seconds,
            "elapsed_seconds": self.elapsed_seconds,
        }
        if include_samples and self.samples:
            payload["samples"] = [
                {"t_seconds": t, "used_bytes": v} for t, v in self.samples
            ]
        return payload


class VramSampler:
    """Background thread that polls one amdgpu card's VRAM-used at fixed interval.

    Sysfs reads take a few microseconds, so ``interval_ms`` can be set
    aggressively (1-10 ms) when we only care about the peak. The sampler
    keeps a running max, and optionally a full trace (``keep_samples=True``)
    for plotting.

    Use as a context manager::

        with VramSampler(card, interval_ms=10) as sampler:
            run_work()
        result = sampler.result()

    Or manually::

        sampler = VramSampler(card, interval_ms=10)
        sampler.start()
        try:
            run_work()
        finally:
            sampler.stop()
        result = sampler.result()
    """

    def __init__(
        self,
        card: AmdgpuCard | None = None,
        interval_ms: float = 50.0,
        *,
        keep_samples: bool = False,
    ) -> None:
        if card is None:
            card = select_card()
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be > 0, got {interval_ms!r}")
        self.card = card
        self.interval_ms = float(interval_ms)
        self.interval_seconds = self.interval_ms / 1000.0
        self.keep_samples = bool(keep_samples)
        self._vram_used_path = str(card.vram_used_path)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._baseline: int = 0
        self._peak: int = 0
        self._final: int = 0
        self._count: int = 0
        self._start_time: float = 0.0
        self._elapsed: float = 0.0
        self._samples: list[tuple[float, int]] = []

    def __enter__(self) -> "VramSampler":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _read(self) -> int:
        # open/read/close on every call is fine: sysfs files are pseudofiles
        # and mmap/lseek caching does not give us the latest value.
        with open(self._vram_used_path, "rb") as fp:
            return int(fp.read().strip())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("VramSampler already started")
        self._stop.clear()
        self._samples.clear()
        self._baseline = self._read()
        self._peak = self._baseline
        self._final = self._baseline
        self._count = 1
        self._start_time = time.perf_counter()
        if self.keep_samples:
            self._samples.append((0.0, self._baseline))
        self._thread = threading.Thread(
            target=self._run, name="amdgpu-vram-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        # Final read so peak reflects the moment of completion even if the
        # poll thread was mid-sleep when the work finished.
        try:
            final = self._read()
        except OSError:
            pass
        else:
            self._final = final
            if final > self._peak:
                self._peak = final
            self._count += 1
            if self.keep_samples:
                self._samples.append((time.perf_counter() - self._start_time, final))
        self._elapsed = time.perf_counter() - self._start_time

    def _run(self) -> None:
        interval = self.interval_seconds
        start = self._start_time
        keep = self.keep_samples
        next_t = start + interval
        while not self._stop.is_set():
            try:
                val = self._read()
            except OSError:
                val = None
            if val is not None:
                if val > self._peak:
                    self._peak = val
                self._count += 1
                if keep:
                    self._samples.append((time.perf_counter() - start, val))
            now = time.perf_counter()
            wait = next_t - now
            if wait > 0:
                # Event.wait returns early if stop() is called.
                if self._stop.wait(wait):
                    break
            else:
                # We are behind schedule; resync rather than catch up in a
                # burst that would skew sample density.
                next_t = now
            next_t += interval

    def peek(self) -> int:
        """Return the current running-peak (bytes) without stopping."""

        return self._peak

    def result(self) -> VramSamples:
        return VramSamples(
            card=self.card,
            baseline_bytes=self._baseline,
            peak_bytes=self._peak,
            final_bytes=self._final,
            samples_count=self._count,
            interval_seconds=self.interval_seconds,
            elapsed_seconds=self._elapsed,
            samples=list(self._samples) if self.keep_samples else [],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poll amdgpu VRAM usage from sysfs and print the peak. "
            "Run with --duration to sample for a fixed time, or --list to "
            "enumerate cards."
        )
    )
    parser.add_argument("--list", action="store_true", help="List amdgpu cards and exit")
    parser.add_argument("--card-name", help="Match by /sys/class/drm card name, e.g. card1")
    parser.add_argument("--pci-id", help="Match by PCI id, e.g. 0000:c3:00.0")
    parser.add_argument("--index", type=int, help="Match by zero-based enumeration index")
    parser.add_argument(
        "--poll",
        type=float,
        default=50.0,
        help="Polling interval in milliseconds (default: 50)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Sampling duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--keep-samples",
        action="store_true",
        help="Retain the full sample trace in the JSON output",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    if args.list:
        cards = list_amdgpu_cards()
        if args.json:
            print(json.dumps([c.to_dict() for c in cards], indent=2))
        else:
            for card in cards:
                print(
                    f"{card.card_name}\tpci={card.pci_id}\t"
                    f"total={card.vram_total_gib:.3f} GiB"
                )
        return

    card = select_card(card_name=args.card_name, pci_id=args.pci_id, index=args.index)
    sampler = VramSampler(
        card=card, interval_ms=args.poll, keep_samples=args.keep_samples
    )
    sampler.start()
    try:
        time.sleep(max(0.0, args.duration))
    finally:
        sampler.stop()
    result = sampler.result()

    if args.json:
        print(json.dumps(result.to_dict(include_samples=args.keep_samples), indent=2))
    else:
        print(
            f"card={card.card_name} pci={card.pci_id} "
            f"baseline={result.baseline_gib:.3f} GiB "
            f"peak={result.peak_gib:.3f} GiB "
            f"delta={result.peak_delta_gib:.3f} GiB "
            f"samples={result.samples_count} "
            f"interval={result.interval_seconds*1000:.1f} ms "
            f"elapsed={result.elapsed_seconds:.3f} s"
        )


if __name__ == "__main__":
    main()
