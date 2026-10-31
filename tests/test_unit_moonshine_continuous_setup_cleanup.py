"""Exercise benchmark topology setup ownership without CUDA or checkpoints."""

import json
import sys
from types import SimpleNamespace

import pytest

from hipengine.core import cuda
from hipengine.loading import moonshine as loading
from hipengine.runtime import moonshine_cuda_batch as batch_module
from hipengine.runtime import moonshine_cuda_continuous as scheduler_module
from scripts import benchmark_moonshine_cuda_continuous as bench


@pytest.mark.parametrize("topology", ["uniform-t256", "exact-two-region"])
def test_continuous_benchmark_cli_keeps_explicit_topology(monkeypatch, topology):
    monkeypatch.setattr(sys, "argv", ["benchmark_moonshine_cuda_continuous.py",
        "--snapshot-dir", "synthetic", "--fixture-dir", "synthetic",
        "--output", "synthetic.json", "--topology", topology])
    assert bench.parse_args().topology == topology


def test_continuous_benchmark_cli_rejects_unknown_topology(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_moonshine_cuda_continuous.py",
        "--snapshot-dir", "synthetic", "--fixture-dir", "synthetic",
        "--output", "synthetic.json", "--topology", "mixed"])
    with pytest.raises(SystemExit) as error:
        bench.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("topology,failure", [
    ("uniform-t256", "prepare1"), ("uniform-t256", "scheduler"),
    ("uniform-t256", "body"), ("uniform-t256", "prepare2"),
    ("exact-two-region", "construct2"), ("exact-two-region", "prepare1"),
    ("exact-two-region", "prepare2"), ("exact-two-region", "scheduler"),
    ("exact-two-region", "body"), ("exact-two-region", "prepare3"),
    ("exact-two-region", "scheduler_close"), ("exact-two-region", "close2"),
    ("uniform-t256", None), ("exact-two-region", None),
])
def test_continuous_benchmark_releases_every_constructed_decoder(monkeypatch, tmp_path, topology, failure):
    created, closed, weight_frees, scheduler_calls = [], [], [], []
    args = SimpleNamespace(requests=2, warmup=0, iterations=1, arrival_interval_steps=1,
        batches=(2,), allow_dirty=True, exclusive_gpu=False, gpu_index=0,
        topology=topology, fixture_dir=tmp_path, snapshot_dir=tmp_path,
        max_graphs=1, output=tmp_path/"result.json")
    monkeypatch.setattr(bench, "parse_args", lambda: args)
    monkeypatch.setattr(bench, "git_state", lambda root: {"dirty": False})
    monkeypatch.setattr(bench, "_gpu_processes", lambda index: [])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(get_device_name=lambda index: "synthetic",
            get_device_capability=lambda index: (12, 0)),
        __version__="synthetic", version=SimpleNamespace(cuda="synthetic")))
    monkeypatch.setattr(bench, "implementation_sha256", lambda *a, **kw: {})
    monkeypatch.setattr(bench, "dependency_adjusted_bytes", lambda *a: {})
    runtime = SimpleNamespace(set_device=lambda index: None)
    monkeypatch.setattr(cuda, "get_cuda_runtime", lambda: runtime)
    monkeypatch.setattr(bench, "load_fixtures", lambda *a: ([object()], 1))
    monkeypatch.setattr(bench, "workload_fixtures", lambda *a: [])
    loaded = SimpleNamespace(weights=SimpleNamespace(free=lambda **kw: weight_frees.append(kw)))
    monkeypatch.setattr(loading, "load_moonshine_model", lambda *a, **kw: loaded)

    class Decoder:
        def __init__(self, **kwargs):
            self.index = len(created)+1
            if failure == f"construct{self.index}":
                raise RuntimeError("synthetic topology failure")
            created.append(self)
            self.closed = False
            self.teardown_returned_to_baseline = False
        def prepare_decoder_kernels(self):
            if failure == f"prepare{self.index}":
                raise RuntimeError("synthetic topology failure")
        def close(self):
            assert not self.closed
            self.closed = True
            self.teardown_returned_to_baseline = True
            closed.append(self.index)
            if failure == f"close{self.index}":
                raise RuntimeError("synthetic topology failure")
        def batch_token_graph_contract(self):
            return {}
        def allocation_contract(self):
            return {"workspace_nbytes": 1}

    class Scheduler:
        topology = "synthetic"
        def __init__(self, *decoders, **kwargs):
            scheduler_calls.append((self.topology, decoders, kwargs))
            if failure == "scheduler":
                raise RuntimeError("synthetic topology failure")
            self.decoders, self.kwargs = decoders, kwargs
        def close(self):
            if failure == "scheduler_close":
                raise RuntimeError("synthetic topology failure")
            if self.kwargs.get("owns_decoder") or self.kwargs.get("owns_decoders"):
                for decoder in self.decoders:
                    decoder.close()
        def graph_cache_contract(self):
            return {"topology": self.topology}
        def scheduler_contract(self):
            return {}

    class UniformScheduler(Scheduler):
        topology = "uniform_t256_rederived"

    class ExactScheduler(Scheduler):
        topology = "exact_two_region"

    monkeypatch.setattr(batch_module, "MoonshineCudaBatchRuntime", Decoder)
    monkeypatch.setattr(scheduler_module, "MoonshineCudaContinuousBatchRuntime", UniformScheduler)
    monkeypatch.setattr(scheduler_module, "MoonshineCudaExactContinuousBatchRuntime", ExactScheduler)
    def sample(*a, **kw):
        if failure == "body":
            raise RuntimeError("synthetic topology failure")
        return [bench.WorkloadResult(1.0, (1.0, 1.0), {})]
    monkeypatch.setattr(bench, "benchmark_route", sample)
    if failure is None:
        assert bench.main() == 0
    else:
        with pytest.raises(RuntimeError, match="synthetic topology failure"):
            bench.main()
    assert sorted(closed) == list(range(1, len(created)+1))
    assert all(decoder.closed for decoder in created)
    assert weight_frees == [{"runtime": runtime}]
    expected_topology = (
        "exact_two_region" if topology == "exact-two-region"
        else "uniform_t256_rederived"
    )
    expected_contract = (
        "bit-exact t32/t256 regions with one D2D state transfer"
        if topology == "exact-two-region"
        else "full-corpus-qualified uniform t256 reassociation"
    )
    if scheduler_calls:
        selected_topology, decoders, kwargs = scheduler_calls[0]
        assert selected_topology == expected_topology
        assert len(decoders) == (2 if topology == "exact-two-region" else 1)
        assert kwargs["max_pending"] == args.requests
        assert kwargs["max_graphs"] == args.max_graphs
    assert args.output.exists() is (failure is None)
    if failure is None:
        report = json.loads(args.output.read_text())
        assert report["scope"]["continuous_topology"] == topology
        assert report["correctness"]["continuous_numerical_contract"] == expected_contract
        assert report["results"]["b2"]["continuous_graph_cache"]["topology"] == expected_topology
