"""Quantify crease sharpness / fade-out / symmetry in authorised .x2m regions (rest pose). Read-only.

python region_compare.py <out.json> <tag>=<pkg_dir | emd:<dir>:<prefix>> [...]
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_io import emd

H = 1.0761; SOLE = -0.697


def load(src):
    """Welded skin surfaces per part: positions, faces, material per face (pubic hair excluded)."""
    parts = {}
    if src.startswith('emd:'):
        folder, prefix = src[4:].rsplit(':', 1)
        for part in ('Bust', 'Pants'):
            P = []; Fs = []; M = []; off = 0
            for s in emd(Path(folder) / f'{prefix}_{part}.emd'):
                if s['name'] == 'HAIR_pubic': continue
                p = np.asarray(s['positions']); f = np.concatenate([np.asarray(g['indices']).reshape(-1, 3) for g in s['groups']])
                P.append(p); Fs.append(f + off); M += [s['name']] * len(f); off += len(p)
            parts[part] = (np.concatenate(P), np.concatenate(Fs), M)
    else:
        for d in json.loads((Path(src) / 'authoring/lod0_geometry.json').read_text()):
            if d['part'] not in ('Bust', 'Pants'): continue
            mats = [d['materials'][m]['name'] for m in d['material_indices']]
            keep = [i for i, m in enumerate(mats) if m != 'HAIR_pubic']
            parts[d['part']] = (np.asarray(d['positions']), np.asarray(d['faces'])[keep], [mats[i] for i in keep])
    out = {}
    for k, (p, f, m) in parts.items():
        key, inv = np.unique(np.round(p, 5), axis=0, return_inverse=True)
        out[k] = (key, inv[f], m)
    return out


def dihedrals(p, f):
    n = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]); a = np.linalg.norm(n, axis=1); n = n / np.maximum(a, 1e-30)[:, None]
    c = p[f].mean(1); E = {}
    for fi, t in enumerate(f):
        for k in range(3):
            E.setdefault(tuple(sorted((int(t[k]), int(t[(k + 1) % 3])))), []).append(fi)
    rows = []
    for (a_, b_), fs in E.items():
        if len(fs) != 2 or a_ == b_: continue
        ang = float(np.degrees(np.arccos(np.clip(np.dot(n[fs[0]], n[fs[1]]), -1, 1))))
        sgn = -1 if np.dot(n[fs[0]], c[fs[1]] - c[fs[0]]) > 0 else 1
        rows.append(((p[a_] + p[b_]) / 2, sgn * ang, np.linalg.norm(p[a_] - p[b_])))
    return rows


def stats(rows, sel):
    r = [x for x in rows if sel(x[0])]
    if not r: return None
    a = np.array([x[1] for x in r])
    return {'edges': len(r), 'abs_p95': float(np.percentile(np.abs(a), 95)), 'abs_max': float(np.abs(a).max()),
            'concave_max': float(max(0, -a.min())), 'concave_over_40': int((a < -40).sum()), 'convex_over_40': int((a > 40).sum())}


def mirror_error(p, sel):
    q = p[[sel(v) for v in p]]
    if len(q) < 3: return None
    m = q * np.array([-1, 1, 1])
    d = np.array([np.min(np.linalg.norm(q - v, axis=1)) for v in m])
    return {'p95': float(np.percentile(d, 95)), 'max': float(d.max())}


h = lambda v: (v[1] - SOLE) / H
REG = {
    'vulva_central': lambda v: abs(v[0]) < 0.035 and 0.535 < h(v) < 0.625 and v[2] < -0.02,
    'vulva_crease_line': lambda v: abs(v[0]) < 0.006 and 0.535 < h(v) < 0.625 and v[2] < -0.02,
    'inguinal_folds': lambda v: 0.035 < abs(v[0]) < 0.09 and 0.55 < h(v) < 0.66 and v[2] < -0.03,
    'pubic_mound': lambda v: abs(v[0]) < 0.05 and 0.60 < h(v) < 0.68 and v[2] < -0.04,
    'nipple_areola': lambda v: abs(abs(v[0]) - 0.075) < 0.03 and v[2] < -0.11 and 0.80 < h(v) < 0.90,
    'lateral_hip': lambda v: abs(v[0]) > 0.12 and 0.55 < h(v) < 0.70,
}
if __name__ == '__main__':
    OUT = Path(sys.argv[1]); specs = [a.split('=', 1) for a in sys.argv[2:]]
    rep = {}
    for tag, src in specs:
        S = load(src); r = {}
        rowsP = dihedrals(*S['Pants'][:2]); rowsB = dihedrals(*S['Bust'][:2])
        for name, sel in REG.items():
            rows = rowsB if name == 'nipple_areola' else rowsP
            r[name] = stats(rows, sel)
        # crease profile along the central line: max concave dihedral per height bin (fade-out behaviour)
        prof = {}
        for mid, ang, _ in rowsP:
            if REG['vulva_crease_line'](mid):
                b = round(h(mid) / 0.01) * 0.01
                prof[f'{b:.2f}'] = max(prof.get(f'{b:.2f}', 0.0), max(0.0, -ang))
        r['crease_profile_concave_deg_by_h'] = dict(sorted(prof.items()))
        r['vulva_mirror'] = mirror_error(S['Pants'][0], REG['vulva_central'])
        # nipple apex: most anterior point per side and the spread of face normals around it
        pB, fB, mB = S['Bust']
        for side in (-1, 1):
            cand = np.where((pB[:, 0] * side > 0.04) & (pB[:, 0] * side < 0.12) & (pB[:, 2] < -0.1))[0]
            apex = cand[np.argmin(pB[cand, 2])]
            ring = np.unique(fB[np.any(fB == apex, axis=1)])
            nf = np.cross(pB[fB[:, 1]] - pB[fB[:, 0]], pB[fB[:, 2]] - pB[fB[:, 0]]); nf /= np.linalg.norm(nf, axis=1, keepdims=True)
            fan = nf[np.any(fB == apex, axis=1)]
            r[f'nipple_{"L" if side < 0 else "R"}'] = {'apex': pB[apex].round(4).tolist(), 'fan_faces': int(len(fan)),
                'fan_normal_spread_deg': float(np.degrees(np.arccos(np.clip(fan @ fan.mean(0) / np.linalg.norm(fan.mean(0)), -1, 1))).max())}
        rep[tag] = r
    OUT.write_text(json.dumps(rep, indent=1))
    for name in list(REG) + ['vulva_mirror']:
        print(f'{name:18s}', ' | '.join(f"{t}: {rep[t][name]}" for t in rep))
    print('crease profile (max concave deg by h):')
    for t in rep: print('  ', t, rep[t]['crease_profile_concave_deg_by_h'])
    for t in rep: print('  ', t, 'nipple R', rep[t]['nipple_R'], 'L', rep[t]['nipple_L'])
