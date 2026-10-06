"""Assemble the reference-fit deliverable from an accepted candidate (no game files are touched).

python package.py <candidate_dir> <package_name>
"""
import hashlib, json, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / sys.argv[1]
NAME = sys.argv[2]
BASE = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930'
            r'\chunli_physique_20261004\deliverable\HUF_REVAMP511_ChunLiInspired_20261005'
            r'\HUF_REVAMP511_ChunLiInspired_20261005')
OUT = HERE / 'deliverable' / NAME
assert not OUT.exists(), OUT
SOURCE_PIN = '26e6bdb1f4db45cd20c7e51a6457c0b60e7c01e8d405933a4b4b1d6a1df61048'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(src, rel):
    dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); return dst


regression = json.loads((CAND / 'fresh_blender_regression.json').read_text())
assert regression.get('new_strict_crossings_total') == 0 and regression.get('source_files_unchanged'), 'candidate not accepted'
assert sha(BASE / 'editable/HUF_REVAMP511_ChunLiInspired.blend') == SOURCE_PIN

# editable source + native payload
put(CAND / 'editable/HUF_REVAMP511_ChunLiInspired.blend', f'editable/{NAME.rsplit("_", 1)[0]}.blend')
for f in sorted((CAND / 'data/chara/HUF').iterdir()):
    put(f, f'data/chara/HUF/{f.name}')
# authoring: JSON sources, writer/reader, exporter that carries positions + normals, templates = approved Chun-Li EMDs
for n in ('asset_io.py', 'emd_writer.py'):
    put(BASE / 'authoring' / n, f'authoring/{n}')
put(HERE / 'export_native.py', 'authoring/export_native.py')
for lod in range(4):
    put(CAND / f'authoring/lod{lod}_geometry.json', f'authoring/lod{lod}_geometry.json')
emds = sorted(p.name for p in (CAND / 'data/chara/HUF').glob('*.emd'))
for n in emds:
    put(BASE / 'data/chara/HUF' / n, f'authoring/native_templates/{n}')
base_contract = json.loads((BASE / 'authoring/source_contract.json').read_text())
required = {}
for rel in (['authoring/asset_io.py', 'authoring/emd_writer.py', 'authoring/export_native.py']
            + [f'authoring/lod{l}_geometry.json' for l in range(4)]
            + [f'authoring/native_templates/{n}' for n in emds]
            + [f'editable/{NAME.rsplit("_", 1)[0]}.blend']):
    required[rel] = sha(OUT / rel)
contract = {'authority': 'Reference-fit derivative of the user-selected Chun-Li source (sha256 ' + SOURCE_PIN + ') against the supplied turnaround reference sheet.',
            'runtime_status': 'Unverified - Runtime Validation Required',
            'source_space': base_contract.get('source_space'),
            'required_inputs': required,
            'output_sha256': {n: sha(OUT / 'data/chara/HUF' / n) for n in emds}}
(OUT / 'authoring/source_contract.json').write_text(json.dumps(contract, indent=2))
# reproduce all 16 EMDs from source with the shipped exporter
with tempfile.TemporaryDirectory() as t:
    r = subprocess.run([sys.executable, str(OUT / 'authoring/export_native.py'), '--output', t], capture_output=True, text=True, cwd=OUT / 'authoring')
    assert r.returncode == 0, r.stderr
    reexport = json.loads(r.stdout)
    assert reexport['all_exact'] and reexport['reconstructed_native_files'] == 16
    for n in emds:
        assert sha(Path(t) / n) == sha(OUT / 'data/chara/HUF' / n), n
# QA evidence
qa = {'localized_changes.json': CAND, 'fields_used.json': CAND, 'field_reference_functions.json': CAND,
      'blender_change_manifest.json': CAND, 'fresh_blender_regression.json': CAND, 'whole_mesh_audit.json': CAND,
      'normal_consistency.json': CAND, 'reference_metrics.json': CAND, 'rig_regression.log': CAND}
for n, d in qa.items():
    if (d / n).exists():
        put(d / n, f'qa/{n}')
(OUT / 'qa/native_reexport_verification.json').write_text(json.dumps(reexport, indent=2))
unchanged = [p.name for p in (BASE / 'data/chara/HUF').iterdir() if sha(p) == sha(OUT / 'data/chara/HUF' / p.name)]
changed = [p.name for p in (BASE / 'data/chara/HUF').iterdir() if p.name not in unchanged]
(OUT / 'qa/payload_comparison.json').write_text(json.dumps({'changed': sorted(changed), 'byte_identical_to_source_package': sorted(unchanged)}, indent=2))
for rel in ('reference/supplied_reference_sheet.webp', 'reference/landmarks_manual.json', 'reference/mask_debug.png'):
    put(HERE / rel, f'qa/reference/{Path(rel).name}')
for p in (HERE / 'qa_views').glob('*.png'):
    put(p, f'qa/views/{p.name}')
for n in ('fields.json', 'build_candidate.py', 'build_blend.py', 'validate_candidate.py', 'whole_mesh_audit.py', 'ref_segment.py',
          'compare_silhouettes.py', 'model_views.py', 'render_shaded.py', 'closeups.py', 'final_metrics.py', 'check_normals.py',
          'attribute_crossings.py', 'package.py'):
    put(HERE / n, f'workflow/{n}')
print(json.dumps({'package': str(OUT), 'changed': changed, 'unchanged_count': len(unchanged)}, indent=1))
