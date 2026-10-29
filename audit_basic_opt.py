"""Record the imported binary and compare it with the committed Host baseline."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
baseline = root.parent/'host_original_before_basic_opt'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = json.loads((baseline/'host_only_manifest.json').read_text())
port = json.loads((root/'basic_opt_port.json').read_text())
for name in ('flash_attn_npu/flash_attn_npu_interface.py',
             'csrc/ascend910/flash_attn_npu/fag_general_host.cpp', 'setup.py'):
    assert sha(root/name) == sha(baseline/name), name
objects = root/'build/lib.linux-aarch64-cpython-312/flash_attn_npu/flash_attn_npu.flash_attn_npu.objs'
new = {p.name:sha(p) for p in objects.glob('*.o')}
old = before['object_evidence']['hashes']
assert old.keys() == new.keys()
changed = [name for name in old if old[name] != new[name]]
assert not any(name.startswith('fwd_') for name in changed), changed
binary = next((root/'flash_attn_npu').glob('*.so'))
assert sha(binary) != before['binary_sha256']
port.update(binary=str(binary), binary_sha256=sha(binary),
            baseline_binary_sha256=before['binary_sha256'],
            objects=dict(unchanged_count=len(old)-len(changed),changed=changed,hashes=new),
            source_sha256={str(p.relative_to(root)):sha(p) for p in (root/'csrc/ascend910/flash_attn_npu').rglob('*')
                           if p.is_file() and p.suffix in ('.cpp','.hpp','.h')})
(root/'basic_opt_port.json').write_text(json.dumps(port,indent=2)+'\n')
original = json.loads((root.parent/'validation/basic_opt_original_snapshot.json').read_text())
manifest = dict(variant='fag_v2_basic_opt_port', binary=str(binary), binary_sha256=sha(binary),
                port=port, host_baseline_manifest=before,
                original_status=original['original_status'],original_hashes=original['original_hashes'])
(root/'host_only_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('New binary:',port['binary_sha256'])
print('Changed objects:',changed)
print('Forward objects unchanged; General Host, Python interface and build flags preserved.')
