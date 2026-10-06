"""Width-normalized lateral hip contour (curvature character only, scale removed) at several heights.
Detects flattened planes (long low-curvature runs) and shelves (curvature spikes) on the lateral pelvis.

python hip_profile.py <out_prefix> <tag>=<src> [...]      (src as in section_profiles.load)
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from section_profiles import load, slice_mesh, H, SOLE

HS = [0.580, 0.600, 0.620, 0.640, 0.660, 0.680]
THETA = np.radians(np.arange(-85, 85.1, 0.5))   # 0 = lateral (+x), -90 = front (-z), +90 = back (+z)


def lateral_contour(p, f, hh, side=1.0):
    y = SOLE + hh * H
    S = slice_mesh(p, f, 1, y)
    S = S[(S[:, :, 0] * side).min(1) > 0.0]
    if not len(S):
        return None
    a = S[:, 0][:, [0, 2]] * [side, 1]; b = S[:, 1][:, [0, 2]] * [side, 1]
    allz = np.concatenate([a[:, 1], b[:, 1]]); zc = (allz.min() + allz.max()) / 2
    c = np.array([0.0, zc]); pts = []
    for t in THETA:
        d = np.array([np.cos(t), np.sin(t)])
        e = b - a; w = a - c
        den = d[0] * e[:, 1] - d[1] * e[:, 0]
        ok = np.abs(den) > 1e-12
        s = np.where(ok, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1), -1)   # distance along ray
        u = np.where(ok, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1), -1)        # segment parameter
        m = ok & (s > 0) & (u >= 0) & (u <= 1)
        pts.append(c + d * s[m].max() if m.any() else [np.nan, np.nan])
    P = np.asarray(pts); hw = np.nanmax(P[:, 0])
    return P, hw, zc


def curvature(P, hw, win=0.015, step=0.001):
    good = ~np.isnan(P[:, 0]); P = P[good]; th = THETA[good]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.concatenate([[0], np.cumsum(seg)])
    ss = np.arange(0, s[-1], step); Q = np.stack([np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])], 1)
    tq = np.interp(ss, s, th)
    k = int(round(win / step)); kap = np.full(len(Q), np.nan)
    for i in range(k, len(Q) - k):
        d1 = Q[i] - Q[i - k]; d2 = Q[i + k] - Q[i]
        ang = np.arctan2(d1[0] * d2[1] - d1[1] * d2[0], (d1 * d2).sum())
        kap[i] = ang / (2 * win) * hw           # + = convex (outward bulge; contour runs front->lateral->back, CCW in x-z), dimensionless
    return np.degrees(tq), kap, step


def metrics(tq, kap, step, hw):
    lat = (tq > -60) & (tq < 60) & ~np.isnan(kap)
    k = kap[lat]; tl = tq[lat]
    flat = k < 0.35; run = best = 0; end = 0
    for i, v in enumerate(flat):
        run = run + 1 if v else 0
        if run > best:
            best = run; end = i
    return {'kappa_min': float(k.min()), 'kappa_max': float(k.max()), 'kappa_p50': float(np.median(k)),
            'flat_run_over_hw': float(best * step / hw), 'flat_run_deg': [float(tl[end - best + 1]), float(tl[end])] if best else [],
            'concave_samples': int((k < 0).sum())}


if __name__ == '__main__':
    out = sys.argv[1]; specs = [a.split('=', 1) for a in sys.argv[2:]]
    meshes = {t: load(s) for t, s in specs}
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(2, len(HS), figsize=(4 * len(HS), 8))
    rep = {}
    for j, hh in enumerate(HS):
        for i, (t, (p, f)) in enumerate(meshes.items()):
            for side, ls in ((1.0, '-'), (-1.0, ':')):
                r = lateral_contour(p, f, hh, side)
                if r is None:
                    continue
                P, hw, zc = r; tq, kap, step = curvature(P, hw)
                rep.setdefault(t, {}).setdefault(f'{hh:.3f}', {})['R' if side > 0 else 'L'] = dict(metrics(tq, kap, step, hw), halfwidth=float(hw))
                axs[0, j].plot(P[:, 0] / hw, (P[:, 1] - zc) / hw, ls, color=cols[i], lw=1.2, label=t if side > 0 else None)
                axs[1, j].plot(tq, kap, ls, color=cols[i], lw=1.0, label=t if side > 0 else None)
        axs[0, j].set_title(f'h {hh:.3f}: lateral section / half-width'); axs[0, j].set_aspect('equal'); axs[0, j].grid(alpha=.3)
        axs[0, j].set_xlabel('x/hw (lateral)'); axs[0, j].set_ylabel('(z-zc)/hw (back +)'); axs[0, j].legend(fontsize=7)
        axs[1, j].set_title('normalized curvature vs angle (0 = lateral)'); axs[1, j].axhline(0, color='k', lw=.6)
        axs[1, j].axhline(0.35, color='gray', lw=.5, ls='--'); axs[1, j].set_ylim(-1.5, 6); axs[1, j].grid(alpha=.3); axs[1, j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out + '.png', dpi=72)
    Path(out + '.json').write_text(json.dumps(rep, indent=1))
    for t, r in rep.items():
        print(t)
        for hh, v in r.items():
            print(f'  h {hh}', '  '.join(f"{s}: kmin {m['kappa_min']:+.2f} kmax {m['kappa_max']:.2f} flat {m['flat_run_over_hw']:.2f} conc {m['concave_samples']:3d}" for s, m in v.items()))
