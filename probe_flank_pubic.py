"""Locate concave creases in the waist flank and characterise pubic-hair cards (rest pose, LOD00). Read-only.

blender -b -P probe_flank_pubic.py -- <tag>=<pkg> [...]
"""
from pathlib import Path
import sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posekit as K

argv = sys.argv[sys.argv.index('--') + 1:]
for spec in argv:
    tag, pkg = spec.split('=', 1)
    parts = K.load(pkg, 0)
    print(f'===== {tag}')
    for pn in ('Bust', 'Pants'):
        part = parts[pn]; p = part['p']
        keep = np.array([m != 'HAIR_pubic' for m in part['mats']]); f = part['f'][keep]
        n = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True)
        c = p[f].mean(1); E = {}
        for fi, t in enumerate(f):
            for k in range(3):
                E.setdefault(tuple(sorted((int(t[k]), int(t[(k + 1) % 3])))), []).append(fi)
        rows = []
        for (a, b), fs in E.items():
            if len(fs) != 2: continue
            mid = (p[a] + p[b]) / 2; h = K.hh(mid[1])
            if not (0.69 < h < 0.83 and abs(mid[0]) > 0.045 and abs(mid[0]) < 0.2): continue
            ang = np.degrees(np.arccos(np.clip(np.dot(n[fs[0]], n[fs[1]]), -1, 1)))
            if np.dot(n[fs[0]], c[fs[1]] - c[fs[0]]) > 0 and ang > 20:
                rows.append((ang, h, mid.round(3).tolist(), a, b))
        rows.sort(reverse=True)
        print(f'  {pn}: concave >20deg in flank band: {len(rows)}')
        for r in rows[:12]:
            print(f'     {r[0]:5.1f} deg  h {r[1]:.3f}  mid {r[2]}  edge {r[3]}-{r[4]}')
    # pubic cards: connected components of HAIR_pubic faces, signed offset to skin
    pt = parts['Pants']; p = pt['p']; hm = np.array([m == 'HAIR_pubic' for m in pt['mats']])
    hf = pt['f'][hm]; sf = pt['f'][~hm]
    tree = BVHTree.FromPolygons([Vector(v) for v in p], sf.tolist(), all_triangles=True)
    parent = {}
    def root(i):
        while parent.setdefault(i, i) != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for t in hf:
        for k in (1, 2):
            ra, rb = root(int(t[0])), root(int(t[k]))
            if ra != rb: parent[rb] = ra
    cards = {}
    for v in np.unique(hf): cards.setdefault(root(int(v)), []).append(int(v))
    stats = []
    for vs in cards.values():
        sd = []
        for v in vs:
            loc, nrm, idx, dist = tree.find_nearest(Vector(p[v]))
            sd.append(dist if np.dot(p[v] - np.asarray(loc), np.asarray(nrm)) >= 0 else -dist)
        sd = np.asarray(sd); stats.append((len(vs), sd.min(), sd.max(), (sd > 0).mean()))
    stats = np.asarray(stats)
    print(f'  pubic cards: {len(stats)}  verts/card {np.median(stats[:,0]):.0f}  '
          f'fully below skin: {int((stats[:,2] < 0).sum())}  fully above: {int((stats[:,1] > 0).sum())}  '
          f'rooted (spans surface): {int(((stats[:,1] < 0) & (stats[:,2] > 0)).sum())}  '
          f'max height above skin: {stats[:,2].max():.4f}  deepest root: {stats[:,1].min():.4f}  '
          f'card max-above median: {np.median(stats[:,2]):.4f}')
