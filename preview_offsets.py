"""Preview package (LOD00 only) = candidate + LOD00 Pants offsets, stored normals rotated with the geometric normal.

python preview_offsets.py <candidate_dir> <offsets.json> <out_dir>
"""
import json, sys
from pathlib import Path
import numpy as np

CAND, OFF, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ds = json.loads((CAND / 'authoring/lod0_geometry.json').read_text())
off = json.loads(OFF.read_text())['displacement']


def rotate_normals(p, q, f, n0, moved):
    def geo(pp):
        fn = np.cross(pp[f[:, 1]] - pp[f[:, 0]], pp[f[:, 2]] - pp[f[:, 0]]); g = np.zeros_like(pp)
        for k in range(3):
            np.add.at(g, f[:, k], fn)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
    g0, g1 = geo(p), geo(q)
    axis = np.cross(g0, g1); s = np.linalg.norm(axis, axis=1); c = (g0 * g1).sum(1); k = axis / np.maximum(s, 1e-30)[:, None]
    nt = n0 * c[:, None] + np.cross(k, n0) * s[:, None] + k * (k * n0).sum(1, keepdims=True) * (1 - c)[:, None]
    fan = np.zeros(len(p), bool); fan[np.unique(f[np.any(moved[f], axis=1)])] = True
    n1 = n0.copy(); sel = fan & (s > 1e-12); n1[sel] = nt[sel]
    return n1


for d in ds:
    if d['part'] != 'Pants':
        continue
    p = np.asarray(d['positions']); f = np.asarray(d['faces']); n0 = np.asarray(d['normals'], float)
    D = np.zeros_like(p)
    for k, v in off.items():
        D[int(k)] = v
    q = p + D; moved = np.linalg.norm(D, axis=1) > 0
    d['normals'] = rotate_normals(p, q, f, n0, moved).tolist(); d['positions'] = q.tolist()
(OUT / 'authoring').mkdir(parents=True, exist_ok=True)
(OUT / 'authoring/lod0_geometry.json').write_text(json.dumps(ds))
print('preview written', OUT)
