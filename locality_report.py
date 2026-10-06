"""Locality of a candidate vs the v13 production baseline: changed vertices per LOD/part/region, protected regions.

python locality_report.py <candidate_dir> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np

H = 1.0761; SOLE = -0.697
V13 = Path(__file__).resolve().parent.parent / 'reference_fit_20261005/candidate_v13'
CAND = Path(sys.argv[1]); OUT = Path(sys.argv[2])


def regions(p, mats_v):
    x, y, z = p[:, 0], p[:, 1], p[:, 2]; h = (y - SOLE) / H; ax = np.abs(x)
    return {
        'forearm_|x|>0.27': ax > 0.27,
        'waist_flank_h0.73-0.83': (h > 0.73) & (h < 0.83) & (ax <= 0.27),
        'hip_pelvis_h0.52-0.73': (h >= 0.52) & (h <= 0.73),
        'vulva_core_|x|<0.03_h0.53-0.64_front': (ax < 0.03) & (h > 0.53) & (h < 0.64) & (z < 0),
        'areola_nipple_materials': np.isin(mats_v, ['SKIN_nipple', 'SKIN_bust_CB']),
        'pubic_hair_component': mats_v == 'HAIR_pubic',
    }


rep = {'baseline': str(V13), 'candidate': str(CAND), 'lods': []}
for lod in range(4):
    vd = {d['part']: d for d in json.loads((V13 / f'authoring/lod{lod}_geometry.json').read_text())}
    row = {'lod': lod, 'parts': []}
    for d in json.loads((CAND / f'authoring/lod{lod}_geometry.json').read_text()):
        pv = np.asarray(vd[d['part']]['positions']); pc = np.asarray(d['positions']); f = np.asarray(d['faces'])
        disp = np.linalg.norm(pc - pv, axis=1); moved = disp > 0
        nchg = np.any(np.abs(np.asarray(d['normals']) - np.asarray(vd[d['part']]['normals'])) > 0, axis=1)
        mats_v = np.array([''] * len(pc), dtype=object)
        for j, t in enumerate(f):
            mats_v[t] = d['materials'][d['material_indices'][j]]['name']
        R = regions(pv, mats_v)
        locked = np.asarray(d['locked'], int)
        part = {'part': d['part'], 'vertices': len(pc), 'moved_vs_v13': int(moved.sum()), 'normals_changed_vs_v13': int(nchg.sum()),
                'max_displacement': float(disp.max()), 'locked_seam_vertices_moved': int(moved[locked].sum()) if len(locked) else 0,
                'regions': {}}
        accounted = np.zeros(len(pc), bool)
        for k, m in R.items():
            part['regions'][k] = {'vertices': int(m.sum()), 'moved': int((moved & m).sum()), 'max_displacement': float(disp[m].max()) if m.any() else 0.0}
            accounted |= m
        part['moved_outside_named_regions'] = int((moved & ~accounted).sum())
        part['max_displacement_outside_named_regions'] = float(disp[~accounted].max()) if (~accounted).any() else 0.0
        row['parts'].append(part)
    rep['lods'].append(row)
OUT.write_text(json.dumps(rep, indent=1))
for row in rep['lods']:
    for p in row['parts']:
        if p['moved_vs_v13'] or p['normals_changed_vs_v13']:
            print(f"LOD{row['lod']} {p['part']:5s} moved {p['moved_vs_v13']:4d}/{p['vertices']}  max {p['max_displacement']:.4f}  seam {p['locked_seam_vertices_moved']}  outside {p['moved_outside_named_regions']} ({p['max_displacement_outside_named_regions']:.4f})  "
                  + '  '.join(f"{k.split('_')[0]}:{v['moved']}/{v['max_displacement']:.4f}" for k, v in p['regions'].items()))
        else:
            print(f"LOD{row['lod']} {p['part']:5s} unchanged")
