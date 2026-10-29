"""Check original Host scope and corrected-baseline device object identity."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
baseline = root.parent / 'baseline'
original = Path('/home/hxb/flash-attention-npu')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((root / 'host_only_manifest.json').read_text())
for name in ['csrc/ascend910/flash_attn_npu/flash_api.cpp',
             'flash_attn_npu/flash_attn_npu_interface.py']:
    assert sha(root / name) == manifest['copied_original_files'][name]
    assert sha(original / name) == manifest['copied_original_files'][name]
host = 'csrc/ascend910/flash_attn_npu/fag_general_host.cpp'
assert sha(original / host) == manifest['copied_original_files'][host]
diff = ''.join(difflib.unified_diff((original / host).read_text().splitlines(True),
                                  (root / host).read_text().splitlines(True),
                                  fromfile='original/' + host, tofile='host_original/' + host))
(root / 'excluded_device_wiring.patch').write_text(diff)
files = subprocess.check_output(['git', 'ls-files'], cwd=root, text=True).splitlines()
changed = [p for p in files if (root/p).is_file() and (baseline/p).is_file()
           and sha(root/p) != sha(baseline/p)]
production = [p for p in changed if not p.startswith('tests/')]
assert sorted(production) == sorted([host, 'csrc/ascend910/flash_attn_npu/flash_api.cpp',
                                     'flash_attn_npu/flash_attn_npu_interface.py']), production
objects = Path('build/lib.linux-aarch64-cpython-312/flash_attn_npu/flash_attn_npu.flash_attn_npu.objs')
old = {p.name:sha(p) for p in (baseline/objects).glob('*.o')}
new = {p.name:sha(p) for p in (root/objects).glob('*.o')}
assert old.keys() == new.keys() and len(new) == 94
different = [p for p in old if old[p] != new[p]]
assert sorted(different) == ['fag_general_host.o', 'flash_api.o'], different
binary = next((root/'flash_attn_npu').glob('*.so'))
assert sha(binary) == sha(root/'build/lib.linux-aarch64-cpython-312/flash_attn_npu'/binary.name)
manifest.update(binary=str(binary), binary_sha256=sha(binary),
                production_diff_vs_corrected_baseline=production,
                object_evidence=dict(same_count=92, different=different, added=[], hashes=new),
                actual_host_source_sha256=sha(root/host))
(root/'host_only_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('PASS: original Host/Python restored; only device-specialization wiring removed.')
print('PASS: 92 unchanged objects, 2 Host objects rebuilt, no new device objects.')
print('Binary SHA256:', manifest['binary_sha256'])
