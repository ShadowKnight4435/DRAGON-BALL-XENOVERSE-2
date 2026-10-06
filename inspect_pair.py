"""Print per-vertex field displacement of the triangles in a new crossing pair.

python inspect_pair.py <candidate_dir> <pose> <lod> [pair_index]
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
cand, pose, lod = sys.argv[1], sys.argv[2], int(sys.argv[3])
k = int(sys.argv[4]) if len(sys.argv) > 4 else 0
src = (HERE / 'build_candidate.py').read_text().split('changes = []; summary = []')[0]
for s in ("assert not CAND.exists(), CAND", "(CAND / 'authoring').mkdir(parents=True)", "(CAND / 'data/chara/HUF').mkdir(parents=True)"):
    src = src.replace(s, '')
sys.argv = ['x', '_q_', str(HERE / cand / 'fields_used.json')]
ns = {'__file__': str(HERE / 'build_candidate.py'), '__name__': 'q'}
exec(compile(src, 'bc', 'exec'), ns)
BASE, field = ns['BASE'], ns['field']
r = json.loads((HERE / cand / 'fresh_blender_regression.json').read_text())
pair = [s['new_crossings'] for p in r['sources']['refined']['poses'] if p['pose'] == pose
        for s in p['intersections'] if s['lod'] == lod][0][k]
rows = []
for d in json.loads((BASE / f'authoring/lod{lod}_geometry.json').read_text()):
    f = np.asarray(d['faces']); p = np.asarray(d['positions'])
    for j in range(len(f)):
        if d['materials'][d['material_indices'][j]]['name'] != 'HAIR_pubic':
            rows.append((d['part'], f[j], p))
for t in pair:
    part, vs, p = rows[t]
    D, parts = field(part, p)
    print('triangle', t, part)
    for v in vs:
        print(f'   v{v:5d} pos {p[v].round(4)}  disp {D[v].round(5)}  ' +
              ' '.join(f'{n}={m[v]:.5f}' for n, m in parts.items()))
