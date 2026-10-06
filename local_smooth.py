"""Localized, feature-preserving (Taubin) smoothing of the external vulvar / inguinal region on LOD00,
propagated to lower LODs and to the pubic-hair cards. Pure numpy; game axes.
"""
import numpy as np

H = 1.0761; SOLE = -0.697


def sstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10)


def band(x, lo0, lo1, hi0, hi1):
    return sstep(lo0, lo1, x) * (1 - sstep(hi0, hi1, x))


def region_weight(p, P):
    """Smooth mask: vulva core (weight 1) plus inguinal folds (weight P['inguinal_strength'])."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]; h = (y - SOLE) / H; ax = np.abs(x)
    front = 1 - sstep(P['front_z_full'], P['front_z_zero'], z)            # z must be in front (negative)
    core = (1 - sstep(P['core_x_full'], P['core_x_zero'], ax)) * band(h, *P['core_h']) * front
    ing = band(ax, *P['ing_x']) * band(h, *P['ing_h']) * front * P['inguinal_strength']
    return np.maximum(core, ing)


def taubin(p, f, w, iters, lam=0.5, mu=-0.53):
    nbr = [set() for _ in range(len(p))]
    for t in f:
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3]); nbr[a].add(b); nbr[b].add(a)
    idx = [np.fromiter(s, int) for s in nbr]
    act = np.where(w > 1e-6)[0]
    q = p.copy()
    for _ in range(iters):
        for coef in (lam, mu):
            L = np.zeros((len(act), 3))
            for k, v in enumerate(act):
                if len(idx[v]): L[k] = q[idx[v]].mean(0) - q[v]
            q[act] += coef * w[act, None] * L
    return q - p


def lod0_delta(pants, P, locked):
    """pants: dict(p=current positions, f=faces, mats=[...]) for LOD00. Returns per-vertex displacement."""
    p = pants['p']; mats = pants['mats']; f = pants['f']
    skin = np.array([m != 'HAIR_pubic' for m in mats])
    w = region_weight(p, P)
    hair_v = np.unique(f[~skin]); w[hair_v] = 0; w[list(locked)] = 0
    d = taubin(p, f[skin], w, P['iterations'], P.get('lambda', 0.5), P.get('mu', -0.53))
    return d * P.get('strength', 1.0)


def propagate(src_rest, src_delta, dst_rest, sigma=0.006, coincide=1e-6):
    """Displacement for dst vertices: exact at coincident rest positions, Gaussian-RBF elsewhere."""
    act = np.where(np.linalg.norm(src_delta, axis=1) > 1e-9)[0]
    out = np.zeros_like(dst_rest)
    if not len(act): return out
    S = src_rest[act]; D = src_delta[act]
    for i, v in enumerate(dst_rest):
        dd = np.linalg.norm(S - v, axis=1)
        j = int(dd.argmin())
        if dd[j] < coincide:
            out[i] = D[j]; continue
        if dd[j] > 4 * sigma: continue
        k = np.exp(-(dd / sigma) ** 2);
        out[i] = (k[:, None] * D).sum(0) / max(k.sum(), 1e-12) * min(1.0, k.max() / np.exp(-1))
    return out
