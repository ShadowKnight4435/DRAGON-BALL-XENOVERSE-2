"""Assemble the nude-base refinement deliverable on the v13 production baseline (no game files are touched).

python package_v13.py <candidate_dir> <package_name>
Assembles evidence; it does not decide approval (see QA_REPORT.md for the gate classification).
"""
import hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / sys.argv[1]
NAME = sys.argv[2]
V13 = HERE.parent / 'reference_fit_20261005/candidate_v13'
TOOLS = HERE.parent / 'chunli_physique_20261004/deliverable/HUF_REVAMP511_ChunLiInspired_20261005/HUF_REVAMP511_ChunLiInspired_20261005'
REF = HERE.parent / 'reference_fit_20261005/reference'
OUT = HERE / 'deliverable' / NAME
assert not OUT.exists(), OUT
V13_PIN = '48a898567622e9bdd4452e8a5035d81fb163899d738a022519145a2f6f873524'
BLEND = 'HUF_REVAMP511_NudeBaseRefine.blend'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(src, rel):
    dst = OUT / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); return dst


regression = json.loads((CAND / 'fresh_blender_regression.json').read_text())
assert regression.get('source_files_unchanged'), 'regression did not complete'
assert sha(V13 / 'editable/HUF_REVAMP511_ChunLiInspired.blend') == V13_PIN
manifest = json.loads((CAND / 'blender_change_manifest.json').read_text())
assert manifest['source_sha256'] == V13_PIN and manifest['output_sha256'] == sha(CAND / 'editable' / BLEND)

put(CAND / 'editable' / BLEND, f'editable/{BLEND}')
for f in sorted((CAND / 'data/chara/HUF').iterdir()):
    put(f, f'data/chara/HUF/{f.name}')
for n in ('asset_io.py', 'emd_writer.py'):
    put(TOOLS / 'authoring' / n, f'authoring/{n}')
put(HERE / 'export_native.py', 'authoring/export_native.py')
for lod in range(4):
    put(CAND / f'authoring/lod{lod}_geometry.json', f'authoring/lod{lod}_geometry.json')
emds = sorted(p.name for p in (CAND / 'data/chara/HUF').glob('*.emd'))
for n in emds:   # templates = the v13 production EMDs (structural authority)
    put(V13 / 'data/chara/HUF' / n, f'authoring/native_templates/{n}')
base_contract = json.loads((TOOLS / 'authoring/source_contract.json').read_text())
required = {rel: sha(OUT / rel) for rel in
            ['authoring/asset_io.py', 'authoring/emd_writer.py', 'authoring/export_native.py']
            + [f'authoring/lod{l}_geometry.json' for l in range(4)]
            + [f'authoring/native_templates/{n}' for n in emds] + [f'editable/{BLEND}']}
contract = {'authority': 'Localized refinement of the v13 production baseline (blend sha256 ' + V13_PIN + '); native templates are the v13 EMDs.',
            'runtime_status': 'Unverified - Runtime Validation Required',
            'source_space': base_contract.get('source_space'),
            'required_inputs': required,
            'output_sha256': {n: sha(OUT / 'data/chara/HUF' / n) for n in emds}}
(OUT / 'authoring/source_contract.json').write_text(json.dumps(contract, indent=2))
with tempfile.TemporaryDirectory() as t:
    r = subprocess.run([sys.executable, str(OUT / 'authoring/export_native.py'), '--output', t], capture_output=True, text=True, cwd=OUT / 'authoring')
    assert r.returncode == 0, r.stderr
    reexport = json.loads(r.stdout)
    assert reexport['all_exact'] and reexport['reconstructed_native_files'] == 16
    for n in emds:
        assert sha(Path(t) / n) == sha(OUT / 'data/chara/HUF' / n), n
(OUT / 'qa').mkdir(exist_ok=True)
(OUT / 'qa/native_reexport_verification.json').write_text(json.dumps(reexport, indent=2))
for n in ('localized_changes.json', 'fields_used.json', 'field_reference_functions.json', 'blender_change_manifest.json',
          'fresh_blender_regression.json', 'whole_mesh_audit.json', 'quality_metrics.json', 'deform_vs_v13.json', 'deform_vs_src.json',
          'deform_all_anims_lod0.json', 'format_check.json', 'component_audit.json', 'dev_final_v13.json', 'dev_final_x2m.json',
          'x2m_usage_audit.json', 'locality_vs_v13.json'):
    if (CAND / n).exists():
        put(CAND / n, f'qa/{n}')
for p in (CAND / 'qa_images').glob('*.png'):
    put(p, f'qa/images/{p.name}')
for p in (CAND / 'qa_extra').glob('*'):
    put(p, f'qa/{p.name}')
unchanged = sorted(p.name for p in (V13 / 'data/chara/HUF').iterdir() if sha(p) == sha(OUT / 'data/chara/HUF' / p.name))
changed = sorted(p.name for p in (V13 / 'data/chara/HUF').iterdir() if p.name not in unchanged)
(OUT / 'qa/payload_comparison_vs_v13.json').write_text(json.dumps({'changed': changed, 'byte_identical_to_v13': unchanged}, indent=2))
for rel in ('supplied_reference_sheet.webp', 'landmarks_manual.json'):
    if (REF / rel).exists():
        put(REF / rel, f'qa/reference/{rel}')
for n in ('build_candidate.py', 'build_blend_v13.py', 'validate_candidate_v13.py', 'whole_mesh_audit_v13.py', 'quality_metrics.py',
          'ean_pose.py', 'posekit.py', 'region_compare.py', 'surface_deviation.py', 'section_profiles.py', 'hip_profile.py',
          'crease_diff.py', 'render_regions.py', 'audit_components.py', 'attribute_ean.py', 'package_v13.py', 'poses_extended.json',
          'fields_final.json'):
    put(HERE / n, f'workflow/{n}')
print(json.dumps({'package': str(OUT), 'changed_vs_v13': changed, 'unchanged_count': len(unchanged)}, indent=1))
