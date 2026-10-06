"""Crease map of the pelvis / groin skin (LOD00 Pants): edges coloured by signed dihedral, optional vertex ids.

python crease_map.py <out.png> <pkg_dir> [--view front|q34|side|low] [--ids] [--hmin 0.50 --hmax 0.74]
Concave (crease) = blue, convex (ridge) = red; colour saturates at 45 deg. Read-only.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

H = 1.0761; SOLE = -0.697


def load(pkg, part='Pants', lod=0):
    d = [x for x in json.loads((Path(pkg) / f'authoring/lod{lod}_geometry.json').read_text()) if x['part'] == part][0]
    mats = [d['materials'][m]['name'] for m in d['material_indices']]
    keep = np.array([m != 'HAIR_pubic' for m in mats])
    return np.asarray(d['positions']), np.asarray(d['faces'])[keep]


def edge_table(p, f):
    """Welded edges -> (v_a, v_b, signed dihedral deg). Vertex ids are original ids (first of each welded group)."""
    key, first, inv = np.unique(np.round(p, 6), axis=0, return_index=True, return_inverse=True)
    wf = inv[f]
    n = np.cross(key[wf[:, 1]] - key[wf[:, 0]], key[wf[:, 2]] - key[wf[:, 0]]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    c = key[wf].mean(1); E = {}
    for fi, t in enumerate(wf):
        for k in range(3):
            E.setdefault(tuple(sorted((int(t[k]), int(t[(k + 1) % 3])))), []).append(fi)
    rows = []
    for (a, b), fs in E.items():
        if len(fs) != 2:
            continue
        ang = float(np.degrees(np.arccos(np.clip(np.dot(n[fs[0]], n[fs[1]]), -1, 1))))
        sgn = -1 if np.dot(n[fs[0]], c[fs[1]] - c[fs[0]]) > 0 else 1
        rows.append((first[a], first[b], sgn * ang))
    return rows


def project(P, view):
    x, y, z = P[..., 0], (P[..., 1] - SOLE) / H, P[..., 2]
    if view == 'front':
        return np.stack([x, y], -1)
    if view == 'side':
        return np.stack([-z, y], -1)
    if view == 'q34':   # camera front-left, 35 deg
        a = np.radians(35); return np.stack([x * np.cos(a) + z * np.sin(a), y], -1)
    if view == 'low':   # looking up from below-front, 40 deg
        a = np.radians(40); return np.stack([x, y * np.cos(a) * H + z * np.sin(a)], -1)
    raise ValueError(view)


if __name__ == '__main__':
    argv = sys.argv[1:]
    OUT, PKG = Path(argv[0]), Path(argv[1])
    opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
    VIEW = opt('--view', 'front'); HMIN = float(opt('--hmin', 0.50)); HMAX = float(opt('--hmax', 0.74))
    XMAX = float(opt('--xmax', 0.20))
    p, f = load(PKG)
    rows = edge_table(p, f)
    hh = (p[:, 1] - SOLE) / H
    sel = lambda v: HMIN < hh[v] < HMAX and abs(p[v, 0]) < XMAX and p[v, 2] < (0.03 if VIEW != 'side' else 0.2)
    segs, cols, lw = [], [], []
    for a, b, ang in rows:
        if not (sel(a) and sel(b)):
            continue
        segs.append(project(np.stack([p[a], p[b]]), VIEW))
        t = min(abs(ang) / 45.0, 1.0)
        cols.append((0.75 * (1 - t), 0.75 * (1 - t), 0.75 * (1 - t) + 0.25 * t if ang < 0 else 0.75 * (1 - t)) if ang < 0 else (0.75 * (1 - t) + t, 0.75 * (1 - t), 0.75 * (1 - t)))
        lw.append(0.6 + 3.0 * t)
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw))
    if '--ids' in argv:
        for v in sorted({a for a, b, _ in rows} | {b for a, b, _ in rows}):
            if sel(v) and p[v, 0] >= -0.002:
                q = project(p[v], VIEW); ax.text(q[0], q[1], str(v), fontsize=6, color='k')
    ax.autoscale(); ax.set_aspect('equal' if VIEW != 'low' else 'auto'); ax.grid(alpha=.2)
    ax.set_title(f'{PKG.name} LOD00 Pants {VIEW}: blue=concave, red=convex (sat. 45 deg)')
    fig.tight_layout(); fig.savefig(OUT, dpi=90); print('wrote', OUT)
