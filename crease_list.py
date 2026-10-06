"""List significant edges (|dihedral| > thr) in a window of LOD00 Pants skin, right side, with left-mirror angle.

python crease_list.py <pkg_dir> [thr] [hmin hmax xmin xmax zmax]
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crease_map import load, edge_table, H, SOLE

pkg = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 15
hmin, hmax, xmin, xmax, zmax = (map(float, sys.argv[3:8]) if len(sys.argv) > 7 else (0.55, 0.70, -0.001, 0.13, 0.0))
p, f = load(pkg)
rows = edge_table(p, f)
hh = (p[:, 1] - SOLE) / H
lookup = {}
for a, b, ang in rows:
    lookup[(round(p[a, 0], 4), round(p[a, 1], 4), round(p[a, 2], 4), round(p[b, 0], 4), round(p[b, 1], 4), round(p[b, 2], 4))] = ang
    lookup[(round(p[b, 0], 4), round(p[b, 1], 4), round(p[b, 2], 4), round(p[a, 0], 4), round(p[a, 1], 4), round(p[a, 2], 4))] = ang
out = []
for a, b, ang in rows:
    m = (p[a] + p[b]) / 2
    if abs(ang) < thr or not (hmin < (m[1] - SOLE) / H < hmax and xmin <= m[0] <= xmax and m[2] < zmax):
        continue
    mir = lookup.get((round(-p[a, 0], 4), round(p[a, 1], 4), round(p[a, 2], 4), round(-p[b, 0], 4), round(p[b, 1], 4), round(p[b, 2], 4)))
    out.append((m[0], a, b, ang, mir, np.linalg.norm(p[a] - p[b])))
for mx, a, b, ang, mir, L in sorted(out, key=lambda t: ((p[t[1], 1] + p[t[2], 1]) / 2)):
    print(f'{a:4d}-{b:<4d}  {ang:+6.1f}  mirror {mir if mir is None else round(mir, 1)!s:>6}  len {L:.4f}   '
          f'a(x {p[a,0]:+.4f} h {hh[a]:.4f} z {p[a,2]:+.4f})  b(x {p[b,0]:+.4f} h {hh[b]:.4f} z {p[b,2]:+.4f})')
print(len(out), 'edges')
