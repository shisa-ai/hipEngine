"""Reconcile traced scratch sizes with explicitly supplied cached code objects.

Library paths must be produced from the intended source/compiler/build settings.
A match is metadata consistency, not proof that a historical trace loaded that
binary: retain loaded-library provenance separately when making that claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


RUNTIME_KERNELS = {'__amd_rocclr_copyBuffer', '__amd_rocclr_fillBufferUnAligned'}


def parse_notes(text):
    return [dict(mangled=name, private_segment_fixed_size=int(size))
            for name, size in re.findall(
                r'\.name:\s+(\S+)\s+\.private_segment_fixed_size:\s+(\d+)', text)]


def reconcile(traces, metadata):
    rows = []
    for name in sorted({r['Kernel_Name'] for r in traces}):
        observed = sorted({int(r['Scratch_Size']) for r in traces if r['Kernel_Name'] == name})
        matches = [r for r in metadata if r['symbol'] == name]
        runtime = name in RUNTIME_KERNELS
        status = ('runtime-unresolved' if runtime else 'missing' if not matches
                  else 'ambiguous' if len(matches) != 1
                  else 'match' if observed == [matches[0]['private_segment_fixed_size']]
                  else 'mismatch')
        rows.append({'symbol': name, 'observed_scratch_bytes': observed,
                     'metadata': matches, 'status': status})
    project = [r for r in rows if r['status'] != 'runtime-unresolved']
    return {'rows': rows, 'project_metadata_complete': bool(project) and
            all(r['status'] == 'match' for r in project),
            'all_metadata_complete': bool(rows) and all(r['status'] == 'match' for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libraries-json', type=Path, required=True)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--llvm-bin', type=Path, required=True)
    parser.add_argument('--arch', default='gfx1151')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    libraries = json.loads(args.libraries_json.read_text())
    metadata, provenance, traces = [], [], []
    for library in libraries:
        path = Path(library['path'])
        with tempfile.TemporaryDirectory(prefix='hipengine-scratch-') as tmp:
            copied = Path(tmp) / path.name
            shutil.copyfile(path, copied)
            subprocess.run([str(args.llvm_bin / 'llvm-objdump'), '--offloading', str(copied)],
                           check=True, capture_output=True, text=True)
            objects = list(Path(tmp).glob(f'*.hipv4-amdgcn-amd-amdhsa--{args.arch}'))
            if len(objects) != 1:
                raise ValueError(f'{path}: expected one {args.arch} code object, found {objects}')
            notes = subprocess.check_output([str(args.llvm_bin / 'llvm-readobj'), '--notes',
                                             str(objects[0])], text=True)
            entries = parse_notes(notes)
            if not entries:
                raise ValueError(f'{path}: no kernel metadata')
            names = subprocess.check_output(['c++filt'],
                input='\n'.join(e['mangled'] for e in entries) + '\n', text=True).splitlines()
            if len(names) != len(entries):
                raise ValueError('demangler count mismatch')
            for entry, name in zip(entries, names):
                metadata.append({**entry, 'symbol': name, 'library': str(path)})
            provenance.append({**library, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'code_object_sha256': hashlib.sha256(objects[0].read_bytes()).hexdigest(),
                               'notes_sha256': hashlib.sha256(notes.encode()).hexdigest()})
    for path in args.trace:
        with path.open() as f:
            traces.extend(csv.DictReader(f))
    report = reconcile(traces, metadata)
    report.update(schema=1, libraries=provenance,
                  traces={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.trace},
                  loaded_library_provenance_verified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('project_metadata_complete', 'all_metadata_complete')}))
    if not report['project_metadata_complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
