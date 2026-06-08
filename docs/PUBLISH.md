# Publishing Checklist

Use this checklist to cut a new hipEngine release and publish it to GitHub and
PyPI. The canonical repo/wordmark is `hipEngine`; the PyPI project and Python
import package are normalized lowercase `hipengine`.

## Versioning

Use semver-style bumps while the public API is still alpha:

- Patch (`0.1.1`): packaging fixes, docs fixes, non-breaking bug fixes.
- Minor (`0.2.0`): new public API/server features, new supported model paths,
  new backend/quant surfaces, or meaningful runtime behavior additions.
- Major (`1.0.0`): intentional breaking changes to Python APIs, server API,
  config/env names, plugin registry semantics, or artifact/cache layout.

## Release Punch List

- [ ] Start from a clean release scope: `git status -sb`.
- [ ] Sync the release base: `git fetch --tags origin` and `git pull --ff-only`.
- [ ] Confirm large/binary assets are present and LFS-backed where expected:
      `git lfs ls-files`.
- [ ] Pick the next version number.
- [ ] Update package metadata in `pyproject.toml`.
- [ ] Update user-facing release notes in `CHANGELOG.md`.
- [ ] Re-read `README.md`, `docs/API.md`, and this file; update examples if
      install, backend selection, server flags, package names, or publish steps changed.
- [ ] Re-read `docs/BENCHMARK.md` and `benchmarks/README.md` if the release notes
      mention performance; every performance claim needs the project evidence policy.
- [ ] Run release validation:
      `python3 -m compileall -q hipengine scripts tests`.
- [ ] Run release validation:
      `uv run --extra dev python -m pytest -q`.
- [ ] Run release validation against the minimum supported Python when available:
      `uv run --python 3.10 --extra dev python -m pytest -q`.
- [ ] Run CLI smokes:
      `uv run --extra dev hipengine --help` and
      `uv run --extra dev hipengine serve --help`.
- [ ] Run the relevant ROCm smoke/correctness gate for the release scope. For
      kernel/runtime releases, follow `docs/TESTING.md` and `docs/KERNELS.md`
      including cached-build profiler guidance.
- [ ] Remove stale build artifacts before rebuilding: `rm -rf dist/`.
- [ ] Build fresh artifacts: `python3 -m build`.
- [ ] Confirm the wheel is not pure-any. For the current bundled AOTriton runtime,
      expect `hipengine-X.Y.Z-py3-none-manylinux_2_39_x86_64.whl` and
      `Root-Is-Purelib: false`.
- [ ] Verify package metadata/rendering:
      `uvx --from twine twine check dist/*`.
- [ ] Smoke-test the built wheel before upload from outside the repo so `uv` does
      not use the local checkout:
      `WHEEL=$(pwd)/dist/hipengine-X.Y.Z-py3-none-manylinux_2_39_x86_64.whl; (cd /tmp && uv run --isolated --with "${WHEEL}" hipengine serve --help)`.
- [ ] Stage only release files explicitly and review them:
      `git add ...`, `git diff --staged --name-only`, `git diff --staged`.
- [ ] Commit release metadata:
      `git commit -m "chore: prepare vX.Y.Z release"`.
- [ ] Push `main` only after validation and review: `git push origin main`.
- [ ] Create an annotated tag:
      `git tag -a vX.Y.Z -m "vX.Y.Z"`.
- [ ] Push the tag: `git push origin vX.Y.Z`.
- [ ] Create or draft the GitHub release from the annotated tag and the matching
      `CHANGELOG.md` entry. Do not leave the release notes empty: copy the
      changelog section to a temp file and run
      `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file /tmp/notes.md`.
- [ ] Verify the GitHub release notes rendered correctly:
      `gh release view vX.Y.Z --json url,name,body`.
- [ ] Publish to PyPI with trusted publishing if configured. If publishing
      manually, prefer exact artifacts over a broad glob:
      `uv publish dist/hipengine-X.Y.Z.tar.gz dist/hipengine-X.Y.Z-py3-none-manylinux_2_39_x86_64.whl`.
- [ ] If `uv publish` is unavailable, use the fallback upload path:
      `uvx --from twine twine upload dist/hipengine-X.Y.Z.tar.gz dist/hipengine-X.Y.Z-py3-none-manylinux_2_39_x86_64.whl`.
- [ ] Verify the published install path from PyPI:
      `uvx --refresh --from "hipengine==X.Y.Z" hipengine serve --help`.
- [ ] Verify the PyPI project page, GitHub release, and Git tag all show the new
      version correctly.
- [ ] Confirm the tree is clean again: `git status -sb`.

## Notes

- Do not publish from a dirty tree.
- Do not reuse old `dist/` artifacts; rebuild for every release.
- Do not add long-lived PyPI tokens to the repo. Prefer trusted publishing or a
  local/user-scoped token configured outside the repository.
- Current wheels are Linux x86-64 only because they bundle an x86-64 AOTriton
  shared-library runtime. The vendored runtime currently audits to a
  `manylinux_2_39_x86_64` floor; ROCm libraries are external system dependencies,
  not bundled wheel payloads. Do not retag as `py3-none-any` or to an older
  `manylinux` floor without a fresh native-artifact audit.
- If backend selection, environment variables, server flags, or cache/artifact
  paths change, update `README.md`, `docs/API.md`, and `CHANGELOG.md` in the same
  release commit.
- The benchmark changelog is not a release changelog. Keep package-level release
  notes in `CHANGELOG.md`; keep measurement history in `benchmarks/CHANGELOG.md`.
