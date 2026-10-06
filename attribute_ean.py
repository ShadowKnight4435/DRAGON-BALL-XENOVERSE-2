"""Attribute ean_pose 'cross' failures (new-region pairs / length increases) to parts and edits.

python attribute_ean.py <deform.json> <candidate_dir>
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rep = json.loads(Path(sys.argv[1]).read_text()); CAND = Path(sys.argv[2])
base = Path(rep['base'])
PARTS = ('Bust', 'Pants', 'Rist', 'Boots')


def tris(pkg, lod):
    rows = []
    for d in json.loads((Path(pkg) / f'authoring/lod{lod}_geometry.json').read_text()):
        pass
    ds = {d['part']: d for d in json.loads((Path(pkg) / f'authoring/lod{lod}_geometry.json').read_text())}
    for pn in PARTS:
        d = ds[pn]; p = np.asarray(d['positions']); f = np.asarray(d['faces'])
        for j in range(len(f)):
            if d['materials'][d['material_indices'][j]]['name'] != 'HAIR_pubic':
                rows.append((pn, f[j], p))
    return rows


cache = {}
for r in rep['rows']:
    if r['severity_gate_pass']:
        continue
    lod = r['lod']
    if lod not in cache:
        cache[lod] = (tris(base, lod), tris(CAND, lod))
    B, C = cache[lod]
    print(f"LOD{lod} {r['anim']}:{r['frame']}  len {r['intersection_length_base']:.4f}->{r['intersection_length_cand']:.4f}  new-region {len(r['new_region_pairs'])}")
    for a, b in r['new_region_pairs'][:6]:
        desc = []
        for t in (a, b):
            pn, vs, pb = B[t]; pc = C[t][2]
            disp = float(np.linalg.norm(pc[vs] - pb[vs], axis=1).max())
            c = pb[vs].mean(0); h = (c[1] + 0.697) / 1.0761
            desc.append(f"{pn} h{h:.3f} ({c[0]:+.3f},{c[2]:+.3f}) moved {disp:.4f}")
        print('     ', ' <> '.join(desc))
