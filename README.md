# hipENGINE

hipENGINE is a ROCm-native local LLM inference engine for AMD RDNA3 / W7900-class GPUs.

- Architecture and roadmap: [`docs/PLAN.md`](docs/PLAN.md)
- Benchmark procedures and baselines: [`docs/BENCHMARK.md`](docs/BENCHMARK.md)
- Kernel port playbook: [`docs/KERNELS.md`](docs/KERNELS.md)
- RDNA3 / W7900 performance model: [`docs/ROOFLINE.md`](docs/ROOFLINE.md)

Status: early scaffold. The runtime hot path is intentionally torch-free; `torch` is reserved for the optional `[torch]` extra at user-boundary interop points.

## License

HIPENGINE source code is licensed under **AGPL-3.0-or-later**. Model weights, checkpoints, and external datasets remain under their own licenses.
