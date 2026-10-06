"""Surface / seam / pubic-hair / LOD-region quality metrics, at rest and in poses. Read-only.

blender -b -P quality_metrics.py -- <out.json> <tag>=<pkg_dir> [...] [--poses rest,HUF_BAS_STAND:0,DIAG:torso_twist]
"""
from pathlib import Path
import bpy, json, sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posekit as K

argv = sys.argv[sys.argv.index('--') + 1:]
OUT = Path(argv[0])
specs = [a.split('=', 1) for a in argv[1:] if '=' in a and not a.startswith('--')]
POSES = (argv[argv.index('--poses') + 1] if '--poses' in argv else 'rest').split(',')


def face_normals(p, f):
    n = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    return n, np.linalg.norm(n, axis=1)


def regions_of(p, part, hw_fn):
    """Boolean masks (per vertex) for anatomical regions, from REST positions (game axes)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]; h = K.hh(y); ax = np.abs(x)
    hw = hw_fn(y); lat = ax > 0.55 * hw
    r = {}
    r['shoulders'] = (h > 0.90) & (h < 0.98) & (ax > 0.10) & (ax < 0.24)
    r['chest'] = (part == 'Bust') & (h > 0.80) & (h < 0.93) & (z < -0.04) & (ax < 0.15)
    r['waist_flank'] = (h > 0.70) & (h < 0.82) & lat & (ax < 0.2)
    r['waist'] = (h > 0.72) & (h < 0.80) & (ax < 0.2)
    r['iliac_crest'] = (h > 0.66) & (h < 0.72) & (ax < 0.2)
    r['lateral_hips'] = (h > 0.54) & (h < 0.68) & lat
    r['glute_apex'] = (h > 0.57) & (h < 0.69) & (z > 0.06)
    r['inner_thighs'] = (h > 0.42) & (h < 0.56) & (ax < 0.06)
    r['lateral_thigh'] = (h > 0.36) & (h < 0.54) & lat
    r['knees'] = (h > 0.33) & (h < 0.41)
    r['calves'] = (h > 0.15) & (h < 0.30)
    r['wrists'] = (ax > 0.42) & (ax < 0.47) & (h > 0.88)
    r['ankles'] = (h > 0.03) & (h < 0.09)
    return r


def edge_dihedrals(p, f):
    """Per interior edge: (v0, v1, signed dihedral deg (+convex/-concave))."""
    n, a = face_normals(p, f); n = n / np.maximum(a, 1e-30)[:, None]
    c = p[f].mean(1); edges = {}
    for fi, t in enumerate(f):
        for k in range(3):
            e = tuple(sorted((int(t[k]), int(t[(k + 1) % 3]))))
            edges.setdefault(e, []).append(fi)
    out = []
    for e, fs in edges.items():
        if len(fs) != 2: continue
        f1, f2 = fs; ang = np.degrees(np.arccos(np.clip(np.dot(n[f1], n[f2]), -1, 1)))
        convex = np.dot(n[f1], c[f2] - c[f1]) < 0
        out.append((e[0], e[1], ang if convex else -ang))
    return out


rep = {'scope': 'Offline metrics (game axes, model units). Poses: EAN keyframes / diagnostic rotations with linear skinning; no engine IK/physics.', 'packages': {}}
for tag, pkg in specs:
    R = {'lods': {}}
    base0 = K.load(pkg, 0)
    # half-width reference (rest LOD00 silhouette per height, |x|<0.2)
    allp = np.concatenate([base0[n]['p'] for n in ('Bust', 'Pants', 'Boots')])
    ys = np.arange(-0.70, 0.42, 0.005)
    hwv = np.array([np.abs(allp[(np.abs(allp[:, 1] - y) < 0.006) & (np.abs(allp[:, 0]) < 0.2), 0]).max() if np.any((np.abs(allp[:, 1] - y) < 0.006) & (np.abs(allp[:, 0]) < 0.2)) else np.nan for y in ys])
    ok = ~np.isnan(hwv); hw_fn = lambda y: np.interp(y, ys[ok], hwv[ok])
    lods = {lod: K.load(pkg, lod) for lod in range(4)}
    for pose in POSES:
        nm, fr = (pose.rsplit(':', 1) if (':' in pose and not pose.startswith('DIAG:')) or pose.count(':') > 1 else (pose, 0))
        T = K.transforms(nm, int(fr)) if pose != 'rest' else None
        prow = {}
        posed = {}
        for lod, parts in lods.items():
            row = {}
            P = {}; Nn = {}
            for pn in K.PARTS:
                P[pn], Nn[pn] = K.skin(parts[pn], T)
            posed[lod] = P
            # ---- seam crease angle (normals of adjacent faces on each side of the ring)
            seams = {}
            for a_, b_, sel in (('Bust', 'Pants', lambda v: 0.05 < v[1] < 0.12), ('Bust', 'Rist', lambda v: abs(v[0]) > 0.4), ('Pants', 'Boots', lambda v: -0.55 < v[1] < -0.5)):
                pa = parts[a_]['p']; pb = parts[b_]['p']
                fa = parts[a_]['f']; fb = parts[b_]['f']
                na, la = face_normals(P[a_], fa); nb, lb = face_normals(P[b_], fb)
                angs = []; gap = 0.0
                for i in parts[a_]['locked']:
                    if not sel(pa[i]): continue
                    j = int(np.linalg.norm(pb - pa[i], axis=1).argmin())
                    if np.linalg.norm(pb[j] - pa[i]) > 1e-6: continue
                    ma = np.any(fa == i, axis=1); mb = np.any(fb == j, axis=1)
                    va = na[ma].sum(0); vb = nb[mb].sum(0)
                    angs.append(float(np.degrees(np.arccos(np.clip(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1)))))
                    gap = max(gap, float(np.linalg.norm(P[a_][i] - P[b_][j])))
                seams[f'{a_}-{b_}'] = {'vertices': len(angs), 'crease_deg_max': max(angs), 'crease_deg_mean': float(np.mean(angs)),
                                       'crease_deg_p90': float(np.percentile(angs, 90)), 'gap_max': gap}
            row['seams'] = seams
            # ---- regional dihedrals (skin faces only)
            reg = {}
            for pn in ('Bust', 'Pants', 'Boots', 'Rist'):
                part = parts[pn]
                keep = np.array([m != 'HAIR_pubic' for m in part['mats']])
                f = part['f'][keep]
                masks = regions_of(part['p'], pn, hw_fn)
                dih = edge_dihedrals(P[pn], f)
                if not dih: continue
                v0 = np.array([d[0] for d in dih]); v1 = np.array([d[1] for d in dih]); ang = np.array([d[2] for d in dih])
                for rn, m in masks.items():
                    sel = m[v0] & m[v1]
                    if not sel.any(): continue
                    a = ang[sel]
                    cur = reg.setdefault(rn, [])
                    cur.extend(a.tolist())
            row['regions'] = {rn: {'edges': len(a), 'dihedral_p95': float(np.percentile(np.abs(a), 95)), 'dihedral_max': float(np.max(np.abs(a))),
                                   'concave_max': float(-min(0.0, min(a))), 'concave_over_20deg': int(np.sum(np.asarray(a) < -20)),
                                   'convex_over_30deg': int(np.sum(np.asarray(a) > 30))} for rn, a in reg.items()}
            # ---- pubic hair vs skin
            pt = parts['Pants']; hmask = np.array([m == 'HAIR_pubic' for m in pt['mats']])
            hf = pt['f'][hmask]; sf = pt['f'][~hmask]
            q = P['Pants']; hv = np.unique(hf)
            tree = BVHTree.FromPolygons([Vector(v) for v in q], sf.tolist(), all_triangles=True)
            sd = []
            for v in hv:
                loc, nrm, idx, dist = tree.find_nearest(Vector(q[v]))
                sd.append(dist if np.dot(np.asarray(q[v]) - np.asarray(loc), np.asarray(nrm)) >= 0 else -dist)
            sd = np.asarray(sd)
            htree = BVHTree.FromPolygons([Vector(v) for v in q], hf.tolist(), all_triangles=True)
            inter = len(tree.overlap(htree))
            row['pubic_hair'] = {'vertices': len(hv), 'signed_offset_min': float(sd.min()), 'p05': float(np.percentile(sd, 5)),
                                 'median': float(np.median(sd)), 'max': float(sd.max()), 'below_skin_count': int((sd < -1e-5).sum()),
                                 'hair_skin_triangle_overlaps': inter}
            prow[f'LOD{lod:02d}'] = row
        # ---- LOD-region deviation (LODn vs LOD00 surface), per region, this pose
        lodreg = {}
        L0 = posed[0]
        t0 = {}
        for pn in ('Bust', 'Pants', 'Boots', 'Rist'):
            keep = np.array([m != 'HAIR_pubic' for m in lods[0][pn]['mats']])
            t0[pn] = BVHTree.FromPolygons([Vector(v) for v in L0[pn]], lods[0][pn]['f'][keep].tolist(), all_triangles=True)
        for lod in (1, 2, 3):
            for pn in ('Bust', 'Pants', 'Boots', 'Rist'):
                rest_p = lods[lod][pn]['p']; masks = regions_of(rest_p, pn, hw_fn)
                q = posed[lod][pn]
                d = np.array([t0[pn].find_nearest(Vector(v))[3] for v in q])
                for rn, m in masks.items():
                    if m.sum() < 3: continue
                    e = lodreg.setdefault(rn, {}).setdefault(f'LOD{lod:02d}', [])
                    e.extend(d[m].tolist())
        prow['lod_region_deviation'] = {rn: {k: {'p95': float(np.percentile(v, 95)), 'max': float(np.max(v))} for k, v in lv.items()} for rn, lv in lodreg.items()}
        R['lods'][pose] = prow
        print('METRICS', tag, pose, flush=True)
    rep['packages'][tag] = R
    OUT.write_text(json.dumps(rep, indent=1))
print('QUALITY_METRICS_DONE')
