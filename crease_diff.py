"""List rest-pose concave creases present in package A but weaker in package B (same topology), LOD00.

python crease_diff.py <A_pkg> <B_pkg> <part> <hmin> <hmax> [thr_deg] [min_abs_x]
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import region_compare as RC

A, B, part = sys.argv[1], sys.argv[2], sys.argv[3]
hmin, hmax = float(sys.argv[4]), float(sys.argv[5])
thr = float(sys.argv[6]) if len(sys.argv) > 6 else 20.0
minx = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0


def rows(pkg):
    d = [x for x in json.loads((Path(pkg) / 'authoring/lod0_geometry.json').read_text()) if x['part'] == part][0]
    p = np.asarray(d['positions']); f = np.asarray(d['faces'])
    mats = [d['materials'][m]['name'] for m in d['material_indices']]
    skin = np.array([m != 'HAIR_pubic' for m in mats])
    key, inv = np.unique(np.round(p, 6), axis=0, return_inverse=True)
    return RC.dihedrals(key, inv[f[skin]])


ra, rb = rows(A), rows(B)
assert len(ra) == len(rb)
out = []
for (ma, aa, _), (mb, ab, _) in zip(ra, rb):
    hh = RC.h(ma)
    if hmin < hh < hmax and abs(ma[0]) >= minx and aa < -thr and aa < ab - 3:
        out.append((aa, ab, hh, ma))
for aa, ab, hh, m in sorted(out, key=lambda t: t[0]):
    print(f'  A {aa:6.1f}  B {ab:6.1f}  h {hh:.3f}  x {m[0]:+.4f}  z {m[2]:+.4f}')
print(len(out), 'edges')
