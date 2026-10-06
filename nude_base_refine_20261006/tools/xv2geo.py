"""Shared read-only geometry helpers (numpy/scipy; mathutils optional). Game axes: x lateral (+ = model left), y up, z back (+).

Normalized height h = (y - SOLE) / H with SOLE/H taken from the production body (prior convention kept for comparability).
"""
import json
from pathlib import Path
import numpy as np

H = 1.0761; SOLE = -0.697
PARTS = ('Bust', 'Pants', 'Rist', 'Boots')


def hh(y):
    return (np.asarray(y) - SOLE) / H


def yy(h):
    return np.asarray(h) * H + SOLE


class Mesh:
    def __init__(self, name, p, f, mats, mat_names, W=None, groups=None, uv=None, cn=None, cn_raw=None, edges=None, props=None):
        self.name = name; self.p = np.asarray(p, float); self.f = np.asarray(f, int); self.mi = np.asarray(mats, int)
        self.mat_names = list(mat_names); self.W = W; self.groups = groups; self.uv = uv; self.cn = cn; self.cn_raw = cn_raw
        self.edges = edges; self.props = props or {}

    @property
    def fmat(self):
        return np.array([self.mat_names[i] for i in self.mi])

    def skin_faces(self):
        return np.array(['HAIR' not in n for n in self.fmat])


def load_npz(path):
    """-> {object_name: Mesh} in game axes, plus meta dict."""
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    out = {}
    for n, info in meta['objects'].items():
        co = d[n + '/co'].astype(float)
        out[n] = Mesh(n, co[:, [0, 2, 1]], d[n + '/tri'], d[n + '/mat'], info['materials'], W=d[n + '/W'], groups=info['groups'],
                      uv=d[n + '/uv'], cn=d[n + '/cn'][:, [0, 2, 1]], cn_raw=d[n + '/cn_raw'], edges=d[n + '/edges'], props=info['props'])
    return out, meta


def load_x2m(emd_path):
    """x2m EMD -> single welded-by-index Mesh per file (submeshes concatenated), game axes as stored."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from asset_io import emd
    P = []; F = []; M = []; names = []; off = 0; N = []; UV = []
    for k, s in enumerate(emd(emd_path)):
        p = np.asarray(s['positions'], float); P.append(p); N.append(np.asarray(s['normals'], float)); UV.append(np.asarray(s['uvs'], float))
        for g in s['groups']:
            t = np.asarray(g['indices']).reshape(-1, 3) + off; F.append(t); M.append(np.full(len(t), k))
        names.append(s['name']); off += len(p)
    m = Mesh(Path(emd_path).stem, np.concatenate(P), np.concatenate(F), np.concatenate(M), names)
    m.vn = np.concatenate(N); m.vuv = np.concatenate(UV)
    return m


def weld(p, decimals=6):
    key, first, inv = np.unique(np.round(p, decimals), axis=0, return_index=True, return_inverse=True)
    return inv.ravel(), p[first]


def face_normals(p, f, unit=True):
    n = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    if unit:
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return n


def vertex_normals(p, f):
    fn = face_normals(p, f, unit=False); g = np.zeros_like(p)
    for k in range(3):
        np.add.at(g, f[:, k], fn)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)


def edge_faces(f):
    """Manifold interior edges -> (edges[E,2], faces[E,2]); boundary and non-manifold edges returned separately."""
    from collections import defaultdict
    E = defaultdict(list)
    for fi, t in enumerate(f):
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3])
            E[(a, b) if a < b else (b, a)].append(fi)
    inter = [(e, fs) for e, fs in E.items() if len(fs) == 2]
    boundary = [e for e, fs in E.items() if len(fs) == 1]
    nonman = [e for e, fs in E.items() if len(fs) > 2]
    return (np.array([e for e, _ in inter], int).reshape(-1, 2), np.array([fs for _, fs in inter], int).reshape(-1, 2),
            np.array(boundary, int).reshape(-1, 2), np.array(nonman, int).reshape(-1, 2))


def signed_dihedral(p, f, edges, efaces):
    """Degrees between adjacent face normals; negative = concave (crease/groove), positive = convex (ridge)."""
    n = face_normals(p, f); c = p[f].mean(1)
    n1 = n[efaces[:, 0]]; n2 = n[efaces[:, 1]]
    ang = np.degrees(np.arccos(np.clip((n1 * n2).sum(1), -1, 1)))
    sgn = np.where(((n1 * (c[efaces[:, 1]] - c[efaces[:, 0]])).sum(1)) > 0, -1.0, 1.0)
    return sgn * ang


def plane_section(p, f, axis, value):
    """Intersection segments of the triangle mesh with plane coord[axis] == value -> (S,2,3)."""
    t = p[f]; d = t[:, :, axis] - value
    segs = []
    for tri, dd in zip(t, d):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if dd[a] * dd[b] < 0:
                s = dd[a] / (dd[a] - dd[b]); pts.append(tri[a] + (tri[b] - tri[a]) * s)
        if len(pts) == 2:
            segs.append(pts)
    return np.asarray(segs).reshape(-1, 2, 3)


def plane_points(p, f, axis, value):
    """Vectorized: all edge/plane intersection points."""
    t = p[f]; out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da = t[:, a, axis] - value; db = t[:, b, axis] - value
        k = da * db < 0
        s = da[k] / (da[k] - db[k]); out.append(t[k, a] + (t[k, b] - t[k, a]) * s[:, None])
    return np.concatenate(out) if out else np.zeros((0, 3))


def combine(meshes, skin_only=True):
    """Concatenate meshes (e.g. one LOD) -> p, f, owner array, per-face material names."""
    P = []; F = []; O = []; FM = []; off = 0
    for k, m in enumerate(meshes):
        keep = m.skin_faces() if skin_only else np.ones(len(m.f), bool)
        P.append(m.p); F.append(m.f[keep] + off); O.append(np.full(len(m.p), k)); FM.extend(m.fmat[keep]); off += len(m.p)
    return np.concatenate(P), np.concatenate(F), np.concatenate(O), np.array(FM)
