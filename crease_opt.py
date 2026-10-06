"""Localized crease / contour refinement of the hip fold and external vulvar contours on LOD00 Pants.

python crease_opt.py <candidate_dir> <params.json> <out_offsets.json>

Each active vertex may only move along its own (rest, area-weighted) normal; offsets are mirror-symmetric; the
vulvar slit (|x| < slit_x) and everything outside the window stay fixed. Least-squares objective:
  * hip-fold chain dihedrals follow a smooth deep-to-faint profile (targets in params),
  * edges sharper than their cap are softened toward the cap,
  * no edge may become sharper than max(original, cap) and no crease/ridge may flip sign,
  * small, smooth offsets (regularization + neighbour smoothness).
Writes per-vertex displacement vectors (game axes) keyed by original LOD00 vertex id. Read-only on inputs.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

H = 1.0761; SOLE = -0.697
CAND = Path(sys.argv[1]); P = json.loads(Path(sys.argv[2]).read_text()); OUT = Path(sys.argv[3])
d = [x for x in json.loads((CAND / 'authoring/lod0_geometry.json').read_text()) if x['part'] == 'Pants'][0]
mats = [d['materials'][m]['name'] for m in d['material_indices']]
skin = np.array([m != 'HAIR_pubic' for m in mats])
p0 = np.asarray(d['positions']); f0 = np.asarray(d['faces'])[skin]
locked = set(map(int, d['locked']))
key, first, inv = np.unique(np.round(p0, 6), axis=0, return_index=True, return_inverse=True)
X = key.copy(); F = inv[f0]; nV = len(X)
hh = (X[:, 1] - SOLE) / H


def face_normals(Y):
    n = np.cross(Y[F[:, 1]] - Y[F[:, 0]], Y[F[:, 2]] - Y[F[:, 0]]); return n


fn = face_normals(X); VN = np.zeros_like(X)
for k in range(3):
    np.add.at(VN, F[:, k], fn)
VN /= np.maximum(np.linalg.norm(VN, axis=1, keepdims=True), 1e-30)

# edges -> two faces
E = {}
for fi, t in enumerate(F):
    for k in range(3):
        E.setdefault(tuple(sorted((int(t[k]), int(t[(k + 1) % 3])))), []).append(fi)
edges = np.array([e for e, fs in E.items() if len(fs) == 2]); efaces = np.array([E[tuple(e)] for e in edges])
nbr = [set() for _ in range(nV)]
for a, b in edges:
    nbr[a].add(b); nbr[b].add(a)


def dihedral(Y):
    n = face_normals(Y); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    c = Y[F].mean(1); n1 = n[efaces[:, 0]]; n2 = n[efaces[:, 1]]
    ang = np.degrees(np.arccos(np.clip((n1 * n2).sum(1), -1, 1)))
    sgn = np.where(((n1 * (c[efaces[:, 1]] - c[efaces[:, 0]])).sum(1)) > 0, -1.0, 1.0)
    return sgn * ang


def inside(v, w):
    return (w['hmin'] < hh[v] < w['hmax']) and abs(X[v, 0]) <= w['xmax'] and X[v, 2] <= w['zmax']


W_eval, W_act = P['eval_window'], P['active_window']
lockedw = {int(inv[i]) for i in locked}
active = [v for v in range(nV) if inside(v, W_act) and abs(X[v, 0]) >= P['slit_x'] and v not in lockedw]
# mirror pairing: variables for x >= 0 side only (midline vertices are excluded by slit_x)
pos = {tuple(np.round(X[v] * [1, 1, 1], 5)): v for v in range(nV)}
var_of = {}; var_verts = []
for v in active:
    if X[v, 0] > 0:
        var_of[v] = len(var_verts); var_verts.append(v)
for v in active:
    if X[v, 0] < 0:
        u = pos.get(tuple(np.round(X[v] * [-1, 1, 1], 5)))
        if u is not None and u in var_of:
            var_of[v] = var_of[u]
nvar = len(var_verts)
act_idx = np.array(list(var_of)); act_var = np.array([var_of[v] for v in act_idx])

# edges evaluated: both endpoints in eval window, not both in the slit, right half (mirror counted once)
mid = (X[edges[:, 0]] + X[edges[:, 1]]) / 2
ev = np.array([inside(a, W_eval) and inside(b, W_eval) and not (abs(X[a, 0]) < P['slit_x'] and abs(X[b, 0]) < P['slit_x'])
               and mid[i, 0] >= -1e-6 for i, (a, b) in enumerate(edges)])
th0 = dihedral(X)


def wid(orig):
    return int(inv[orig])


chain = {}
for a, b, tgt in P['chain_targets']:
    e = tuple(sorted((wid(a), wid(b))))
    idx = np.where((edges[:, 0] == e[0]) & (edges[:, 1] == e[1]))[0]
    assert len(idx) == 1, (a, b)
    chain[int(idx[0])] = tgt
soft = {}
for a, b, tgt in P.get('soften_targets', []):   # one-sided: penalize only the part above the target
    e = tuple(sorted((wid(a), wid(b))))
    idx = np.where((edges[:, 0] == e[0]) & (edges[:, 1] == e[1]))[0]
    assert len(idx) == 1, (a, b)
    soft[int(idx[0])] = tgt
vz = P['vulva_zone']
in_vulva = np.array([abs(m[0]) < vz['xmax'] and vz['hmin'] < (m[1] - SOLE) / H < vz['hmax'] for m in mid])
# fold band: edges within band_r of the hip-fold chain polyline (curvature may be redistributed there)
chain_pts = np.array([X[wid(v)] for v in P.get('band_polyline', [])]) if P.get('band_polyline') else np.zeros((0, 3))


def dist_polyline(q):
    best = np.inf
    for a_, b_ in zip(chain_pts[:-1], chain_pts[1:]):
        ab = b_ - a_; u = np.clip(np.dot(q - a_, ab) / max(np.dot(ab, ab), 1e-30), 0, 1)
        best = min(best, np.linalg.norm(q - (a_ + u * ab)))
    return best


in_band = np.array([dist_polyline(m) < P.get('band_r', 0.0) for m in mid]) if len(chain_pts) else np.zeros(len(mid), bool)
in_capzone = in_vulva | in_band
cap = np.where(th0 >= 0, np.where(in_vulva, P['cap_convex_vulva'], P['cap_convex']), np.where(in_vulva, P['cap_concave_vulva'], P['cap_concave']))
cap = np.where(in_capzone, cap, np.inf)          # caps only act in the vulva zone and fold band
ev_idx = np.where(ev)[0]
chain_idx = np.array(sorted(chain)); chain_tgt = np.array([chain[i] for i in chain_idx])
nochain = np.array([i for i in ev_idx if i not in chain])
soft_idx = np.array(sorted(soft), int); soft_tgt = np.array([soft[i] for i in soft_idx])
# no edge may sharpen: over-cap edges not at all, others by at most 'slack' degrees (band: 'slack_band')
slack = np.where(in_band & ~in_vulva, P.get('slack_band', P.get('slack', 5.0)), P.get('slack', 5.0))   # never inside the vulva zone
allow = np.where(np.abs(th0) > cap, np.abs(th0), np.abs(th0) + slack)
allow = np.where(np.isinf(cap), np.abs(th0) + slack, allow)
bound = np.array([P['tmax_rim'] if abs(X[v, 0]) < P['rim_x'] else (P['tmax'] if abs(X[v, 0]) < P.get('thigh_x', 1.0) and hh[v] < P.get('vulva_zone')['hmax'] + 0.01 else P.get('tmax_thigh', P['tmax'])) for v in var_verts])
act_edges = np.array([(var_of[a], var_of[b]) for a, b in edges if a in var_of and b in var_of and var_of[a] != var_of[b]])


def displaced(t):
    Y = X.copy(); Y[act_idx] += t[act_var][:, None] * VN[act_idx]; return Y


fair_zone = in_capzone[nochain]


def residuals(t):
    th = dihedral(displaced(t))
    r = [P['w_chain'] * (th[chain_idx] - chain_tgt) / 8.0]
    if P.get('mode') == 'fair' and len(soft_idx):
        r.append(P.get('w_target_soften', 1.0) * np.maximum(0, np.abs(th[soft_idx]) - soft_tgt) / 6.0)
    if P.get('mode') == 'fair' and 'slack_concave' in P:
        # no new / deepened concavity beyond slack_concave on edges that were not clearly concave (no grooves)
        a_ = th[nochain]; mild = th0[nochain] > -15
        r.append(P['w_nosharpen'] * np.where(mild, np.maximum(0, (th0[nochain] - P['slack_concave']) - a_), 0) / 2.0)
    if P.get('mode') == 'fair':
        # fairing: squared crease angles inside the vulva zone / fold band (spreads curvature, rounds lips);
        # edges outside the zone keep their angle; no sign flips; small smooth offsets.
        a = th[nochain]
        r.append(P['w_fair'] * np.where(fair_zone, a, 0) / 20.0)
        r.append(P['w_keep'] * np.where(fair_zone, 0, a - th0[nochain]) / 4.0)
        flip = np.abs(th0[nochain]) >= 6
        r.append(P['w_noflip'] * np.where(flip, np.maximum(0, -a * np.sign(th0[nochain])), 0) / 3.0)
        r.append(P['w_nosharpen'] * np.maximum(0, np.abs(a) - (np.abs(th0[nochain]) + P.get('slack', 5.0))) / 2.0)
        r.append(P['w_reg'] * t / 0.0012)
        if len(act_edges):
            r.append(P['w_smooth'] * (t[act_edges[:, 0]] - t[act_edges[:, 1]]) / 0.0015)
        return np.concatenate(r)
    if len(soft_idx):
        r.append(P.get('w_target_soften', 1.0) * np.maximum(0, np.abs(th[soft_idx]) - soft_tgt) / 6.0)
    a = np.abs(th[nochain])
    r.append(P['w_soften'] * np.where(np.isinf(cap[nochain]), 0, np.maximum(0, a - np.where(np.isinf(cap[nochain]), 0, cap[nochain]))) / 10.0)
    r.append(P['w_nosharpen'] * np.maximum(0, a - allow[nochain]) / 2.0)
    flip = np.abs(th0[nochain]) >= 6
    r.append(P['w_noflip'] * np.where(flip, np.maximum(0, -th[nochain] * np.sign(th0[nochain])), 0) / 3.0)
    r.append(P['w_reg'] * t / 0.0012)
    if len(act_edges):
        r.append(P['w_smooth'] * (t[act_edges[:, 0]] - t[act_edges[:, 1]]) / 0.0015)
    return np.concatenate(r)


res = least_squares(residuals, np.zeros(nvar), bounds=(-bound, bound), x_scale=0.001, diff_step=1e-3, max_nfev=P.get('max_nfev', 400))
t = res.x; Y = displaced(t); th = dihedral(Y)
print(f'variables {nvar} (active welded vertices {len(act_idx)}), evaluated edges {len(ev_idx)}, cost {res.cost:.3f}, status {res.status}')
print('chain (target / before -> after):')
for i, tg in zip(chain_idx, chain_tgt):
    a, b = edges[i]; print(f'   {first[a]:4d}-{first[b]:<4d} {tg:+6.1f} / {th0[i]:+7.1f} -> {th[i]:+7.1f}')
ch = [(abs(th[i] - th0[i]), i) for i in ev_idx if abs(th[i] - th0[i]) > 2.5 and i not in chain]
print('other edges changed > 2.5 deg:')
for _, i in sorted(ch, reverse=True)[:40]:
    a, b = edges[i]
    print(f'   {first[a]:4d}-{first[b]:<4d} {th0[i]:+7.1f} -> {th[i]:+7.1f}   h {(mid[i,1]-SOLE)/H:.3f} x {mid[i,0]:+.4f} z {mid[i,2]:+.4f}')
mag = np.abs(t)
print(f'offsets: max {mag.max():.4f}  p95 {np.percentile(mag, 95):.4f}  moved(>1e-5) {int((mag > 1e-5).sum())}/{nvar} variables')
over_before = int((np.abs(th0[nochain]) > cap[nochain]).sum()); over_after = int((np.abs(th[nochain]) > cap[nochain]).sum())
print('soften targets (target / before -> after):')
for i, tg in zip(soft_idx, soft_tgt):
    a, b = edges[i]; print(f'   {first[a]:4d}-{first[b]:<4d} <= {tg:5.1f} / {th0[i]:+7.1f} -> {th[i]:+7.1f}')
sharper = int((np.abs(th[nochain]) > allow[nochain] + 0.5).sum())
print(f'edges above cap: {over_before} -> {over_after};  edges sharpened beyond max(orig,cap)+0.5: {sharper}')
disp = {}
for v_w, var in var_of.items():
    if abs(t[var]) < P.get('snap', 1e-6):
        continue
    vec = (t[var] * VN[v_w]).tolist()
    for o in np.where(inv == v_w)[0]:
        disp[int(o)] = vec
OUT.write_text(json.dumps({'candidate': str(CAND), 'params': P, 'lod0_rest_positions': {k: p0[k].tolist() for k in disp},
                           'displacement': disp, 'max_offset': float(mag.max()),
                           'chain': [[int(first[edges[i][0]]), int(first[edges[i][1]]), float(tg), float(th0[i]), float(th[i])] for i, tg in zip(chain_idx, chain_tgt)]}, indent=1))
print('still above cap:')
for i in nochain:
    if abs(th[i]) > cap[i]:
        a, b = edges[i]; print(f'   {first[a]:4d}-{first[b]:<4d} cap {cap[i]:4.0f}  {th0[i]:+7.1f} -> {th[i]:+7.1f}   h {(mid[i,1]-SOLE)/H:.3f} x {mid[i,0]:+.4f} z {mid[i,2]:+.4f}')
print('wrote', OUT, len(disp), 'vertex displacements (incl. duplicates)')
