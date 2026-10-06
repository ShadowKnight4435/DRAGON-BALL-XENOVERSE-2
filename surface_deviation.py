"""Exact point-to-surface deviation between two LOD00 Pants skins (groin / pelvis window), both directions,
plus mirror (left/right) error of each. Read-only.

python surface_deviation.py <out.json> A=<src> B=<src>      (src as in section_profiles.load)
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from section_profiles import load, H, SOLE


def closest_on_tris(P, A, B, C):
    """Closest points on triangles (A,B,C) to points P (all (n,3)), Ericson 5.1.5, vectorized."""
    ab = B - A; ac = C - A; ap = P - A
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = P - B; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = P - C; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    den = np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = vb / den; w = vc / den
    out = A + ab * v[:, None] + ac * w[:, None]
    # edge / vertex regions
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); t = d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3); out[m] = (A + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); t = d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6); out[m] = (A + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6)); out[m] = (B + (C - B) * t[:, None])[m]
    m = (d1 <= 0) & (d2 <= 0); out[m] = A[m]
    m = (d3 >= 0) & (d4 <= d3); out[m] = B[m]
    m = (d6 >= 0) & (d5 <= d6); out[m] = C[m]
    return out


def point_surface(P, p, f, k=16):
    """Unsigned distance and signed (by face normal) offset from points P to mesh (p,f)."""
    cen = p[f].mean(1); tree = cKDTree(cen)
    _, idx = tree.query(P, k=k)
    best = np.full(len(P), np.inf); sgn = np.zeros(len(P))
    fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]); fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-20)
    for j in range(k):
        tri = f[idx[:, j]]
        q = closest_on_tris(P, p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]])
        d = np.linalg.norm(P - q, axis=1); m = d < best
        best[m] = d[m]; sgn[m] = ((P - q) * fn[idx[:, j]]).sum(1)[m]
    return best, sgn


def window(p):
    hh = (p[:, 1] - SOLE) / H
    return (hh > 0.50) & (hh < 0.70) & (np.abs(p[:, 0]) < 0.12) & (p[:, 2] < 0.03)


def regions(p):
    hh = (p[:, 1] - SOLE) / H; ax = np.abs(p[:, 0]); front = p[:, 2] < -0.0
    return {'vulva_core': (ax < 0.03) & (hh > 0.53) & (hh < 0.64) & front,
            'inguinal': (ax >= 0.03) & (ax < 0.09) & (hh > 0.55) & (hh < 0.67) & front,
            'anterior_hip': (ax >= 0.09) & (hh > 0.55) & (hh < 0.70) & front,
            'mons': (ax < 0.05) & (hh >= 0.64) & (hh < 0.70) & front,
            'perineum_inner_thigh': (hh <= 0.58) & (ax < 0.07)}


def mirror_error(p, f, sel):
    q = p[sel] * np.array([-1, 1, 1])
    d, _ = point_surface(q, p, f)
    return {'p95': float(np.percentile(d, 95)), 'max': float(d.max())}


if __name__ == '__main__':
    out = Path(sys.argv[1]); (ta, sa), (tb, sb) = [a.split('=', 1) for a in sys.argv[2:4]]
    pa, fa = load(sa); pb, fb = load(sb)
    rep = {}
    for (t1, p1, f1), (t2, p2, f2) in (((ta, pa, fa), (tb, pb, fb)), ((tb, pb, fb), (ta, pa, fa))):
        used = np.unique(f1); sel = np.zeros(len(p1), bool); sel[used] = True; sel &= window(p1)
        d, s = point_surface(p1[sel], p2, f2)
        R = regions(p1[sel]); r = {}
        for k, m in R.items():
            if m.any():
                r[k] = {'n': int(m.sum()), 'mean': float(d[m].mean()), 'p95': float(np.percentile(d[m], 95)), 'max': float(d[m].max()),
                        'signed_mean': float(s[m].mean())}
        top = np.argsort(-d)[:12]
        r['top'] = [{'v': int(np.where(sel)[0][i]), 'h': float((p1[sel][i, 1] - SOLE) / H), 'x': float(p1[sel][i, 0]),
                     'z': float(p1[sel][i, 2]), 'd': float(d[i]), 'signed': float(s[i])} for i in top]
        r['mirror'] = {k: mirror_error(p1, f1, np.where(sel)[0][m]) for k, m in R.items() if m.any()}
        rep[f'{t1}_to_{t2}'] = r
    out.write_text(json.dumps(rep, indent=1))
    for k, r in rep.items():
        print(k)
        for g in ('vulva_core', 'inguinal', 'anterior_hip', 'mons', 'perineum_inner_thigh'):
            if g in r:
                v = r[g]; mm = r['mirror'][g]
                print(f'  {g:22s} n {v["n"]:4d} mean {v["mean"]*1000:6.2f}e-3 p95 {v["p95"]*1000:6.2f}e-3 max {v["max"]*1000:6.2f}e-3 signed {v["signed_mean"]*1000:+6.2f}e-3 | mirror p95 {mm["p95"]*1000:5.2f}e-3 max {mm["max"]*1000:5.2f}e-3')
        for t in r['top'][:6]:
            print('   top', {a: round(b, 4) if isinstance(b, float) else b for a, b in t.items()})
