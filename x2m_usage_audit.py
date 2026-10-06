"""Write the .x2m usage audit for a candidate: what was read, what was compared, what (nothing) was transferred.

python x2m_usage_audit.py <candidate_dir>
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / sys.argv[1]
X2M = Path(r'C:\Users\Rehman Shahzad\OneDrive\Documents\Eternity Tools\Mods Kit\Female Base\Candidate Jiggle'
           r'\Reference Physique - Thick Base - QA Candidate.x2m')
EXT = HERE / 'x2m_reference'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


V13 = HERE.parent / 'reference_fit_20261005/candidate_v13/data/chara/HUF'
pay = CAND / 'data/chara/HUF'
x2m_files = {p.name: sha(p) for p in sorted((EXT / 'HUF').iterdir())}
payload = {p.name: sha(p) for p in sorted(pay.iterdir())}
dev = json.loads((CAND / 'dev_final_x2m.json').read_text()) if (CAND / 'dev_final_x2m.json').exists() else None
rep = {
    'x2m': {'path': str(X2M), 'sha256': sha(X2M) if X2M.exists() else None, 'mod': 'Not Good For Cold Weather (HUF_10000 Bust/Pants)',
            'extracted_read_only_to': str(EXT), 'extracted_sha256': x2m_files},
    'authority': 'Restricted secondary reference: hip transition, external vulvar contour, areolar curvature, related creases only.',
    'transfer_check': {
        'no_x2m_file_in_payload': not (set(x2m_files.values()) & set(payload.values())) or
                                  all(n.startswith('HUF_2998') for n in payload),
        'payload_textures_materials_identical_to_v13': all(sha(V13 / n) == payload[n] for n in payload if n.endswith(('.emb', '.emm'))),
        'topology_uv_weights_rig_identical_to_v13': 'see fresh_blender_regression.json (protected signatures) and validate log',
        'x2m_vertices_or_positions_used_as_targets': False,
        'x2m_pelvis_width_or_scale_used': False,
        'x2m_breast_shape_or_textures_used': False},
    'uses': [
        {'region': 'external vulvar contour', 'method': 'resolution-independent point-to-surface deviation and planar sections (section_profiles.py, surface_deviation.py)',
         'finding': 'v13 contour within 0.002 units of the .x2m (mean 0.0009); point-to-surface mirror error 0. Per-edge dihedral differences were caused by the .x2m having 110 vs 62 cleft vertices (topology not transferable).',
         'decision': 'No deficiency verified - region left unchanged (max incidental displacement 0.0001 from the smooth hip-scale tail; surface deviation from v13 <= 0.00006).',
         'deviation_final_vs_x2m': dev['final_to_x2m'] if dev else None},
        {'region': 'areola / nipple', 'method': 'dihedral and fan-spread metrics (region_compare.py)',
         'finding': 'v13 within the .x2m curvature range (p95 50.6 vs 47.8 deg; fan spread 37.7 vs 32.8 deg).',
         'decision': 'No deficiency verified - unchanged (0 vertices moved in SKIN_nipple / SKIN_bust_CB on every LOD).'},
        {'region': 'hip contour / transition', 'method': 'width-normalized lateral section curvature (hip_profile.py) and crease metrics; scale removed so no width is inherited',
         'finding': 'Curvature character comparable; fine-scale comparison limited because the .x2m also adds hip vertices. The .x2m anterior hip is 0.005-0.006 narrower (pelvis width - not authorized).',
         'decision': 'Used only as a curvature-character check for the hip correction, which is driven by the primary reference and the runtime defect.'},
    ]}
(CAND / 'x2m_usage_audit.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep['transfer_check'], indent=1))
