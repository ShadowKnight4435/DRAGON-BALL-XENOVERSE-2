"""Locate a crossing pair (triangle indices as numbered by ean_pose.scan) and show its vertices in several packages.

python pair_info.py <lod> <a> <b> tag=pkg [tag=pkg ...]
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posekit as K

lod, a, b = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
specs = [s.split('=', 1) for s in sys.argv[4:]]


def tri_table(parts):
    rows = []
    for pname in K.PARTS:
        part = parts[pname]
        for i, m in enumerate(part['mats']):
            if m != 'HAIR_pubic':
                rows.append((pname, i, part['f'][i], m))
    return rows


pk = {t: K.load(p, lod) for t, p in specs}
rows = tri_table(pk[specs[0][0]])
for t in (a, b):
    pname, fi, vs, mat = rows[t]
    print(f'tri {t}: {pname} face {fi} mat {mat} verts {list(map(int, vs))}')
    for v in vs:
        line = f'   v{int(v):5d}'
        p0 = pk[specs[0][0]][pname]['p'][v]
        line += f'  h {K.hh(p0[1]):.4f} x {p0[0]:+.4f} z {p0[2]:+.4f}'
        for tag, _ in specs[1:]:
            d = pk[tag][pname]['p'][v] - p0
            line += f' | {tag} d=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f})'
        bones = pk[specs[0][0]][pname]['bones']; W = pk[specs[0][0]][pname]['W'][v]
        top = np.argsort(-W)[:3]
        line += '  w ' + ', '.join(f'{bones[j]}:{W[j]:.2f}' for j in top if W[j] > 0)
        print(line)
