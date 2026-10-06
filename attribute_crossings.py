"""Attribute new strict crossings (from fresh_blender_regression.json) to parts and edit fields.

python attribute_crossings.py <candidate_dir>
"""
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
CAND = HERE / sys.argv[1]
rep = json.loads((CAND / 'fresh_blender_regression.json').read_text())
src = (HERE / 'build_candidate.py').read_text().split("changes = []; summary = []")[0]
src = src.replace("assert not CAND.exists(), CAND", "").replace("(CAND / 'authoring').mkdir(parents=True)", "").replace("(CAND / 'data/chara/HUF').mkdir(parents=True)", "")
sys.argv = ['x', '_attr_', str(CAND / 'fields_used.json')]
ns = {'__file__': str(HERE / 'build_candidate.py'), '__name__': 'attr'}
exec(compile(src, 'bc', 'exec'), ns)
BASE = ns['BASE']; field = ns['field']
tri = {}
for lod in range(4):
    ds = json.loads((BASE / f'authoring/lod{lod}_geometry.json').read_text())
    rows = []
    for d in ds:
        f = np.asarray(d['faces']); p = np.asarray(d['positions'])
        kept = [j for j in range(len(f)) if d['materials'][d['material_indices'][j]]['name'] != 'HAIR_pubic']
        D, parts = field(d['part'], p) if d['part'] in ('Bust', 'Pants') else (np.zeros_like(p), {})
        for j in kept:
            vs = f[j]
            mags = {k: float(v[vs].max()) for k, v in parts.items()}
            rows.append((d['part'], p[vs].mean(0), mags))
    tri[lod] = rows
for prow in rep['sources']['refined']['poses']:
    for s in prow['intersections']:
        if not s['new_crossings']:
            continue
        lod = s['lod']; c = Counter(); locs = []
        for a, b in s['new_crossings']:
            for t in (a, b):
                part, cen, mags = tri[lod][t]
                dom = max(mags, key=mags.get) if mags and max(mags.values()) > 1e-6 else 'unedited'
                c[(part, dom)] += 1
            locs.append((tri[lod][a][0], tri[lod][a][1].round(3).tolist(), tri[lod][b][0], tri[lod][b][1].round(3).tolist()))
        print(f"{prow['pose']:14s} LOD{lod} new {len(s['new_crossings']):3d}  triangles by (part, dominant edit): {dict(c)}")
        for l in locs[:3]:
            print('        e.g.', l)
