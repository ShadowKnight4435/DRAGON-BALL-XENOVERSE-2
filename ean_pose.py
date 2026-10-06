"""Offline game-animation posing (EAN keyframes + linear skinning from the JSON weights). Read-only.

blender -b -P ean_pose.py -- render <out_dir> <tag>=<pkg_dir> [...] --anim NAME:FRAME[,NAME:FRAME] [--lod N] [--views v1,v2]
blender -b -P ean_pose.py -- cross  <out.json> <base_tag>=<pkg_dir> <cand_tag>=<pkg_dir> --anim ALL|NAME:FRAME,... [--lods 0,1,2,3] [--frames start,mid,end]

pkg_dir must contain authoring/lod{N}_geometry.json (game axes). Not an engine test: no IK, physics or engine skinning quirks.
"""
from pathlib import Path
import bpy, json, math, sys, time
import numpy as np
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

ROOT = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930')
sys.path.insert(0, str(ROOT / 'deliverable/HUF_REVAMP511_ReferenceBody/authoring'))
from asset_io import skeleton
from skinning import AnimationFile, global_matrices, matrix, rotation
DIAG = json.loads((Path(__file__).resolve().parent / 'poses_extended.json').read_text())
DIAG.pop('_note', None)

EAN = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\HUF Assets\HUF_000_REVAMP_v5.1.1\HUF\HUF.ean')
ESK = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\HUF Assets\HUF_000_REVAMP_v5.1.1\HUF\HUF_000.esk')
argv = sys.argv[sys.argv.index('--') + 1:]
MODE = argv[0]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


specs = [a.split('=', 1) for a in argv[2:] if '=' in a and not a.startswith('--')]
anim = AnimationFile(EAN)
rig = skeleton(ESK)
cat = {r['name']: r for r in anim.catalog}
PARTS = ('Bust', 'Pants', 'Rist', 'Boots')


def load(pkg, lod):
    out = {}
    for d in json.loads((Path(pkg) / f'authoring/lod{lod}_geometry.json').read_text()):
        p = np.asarray(d['positions'], float); n = np.asarray(d['normals'], float)
        names = sorted({b for w in d['weights'] for b in w})
        W = np.zeros((len(p), len(names)))
        idx = {b: i for i, b in enumerate(names)}
        for vi, w in enumerate(d['weights']):
            for b, x in w.items():
                W[vi, idx[b]] = x
        mats = [d['materials'][m]['name'] for m in d['material_indices']]
        out[d['part']] = dict(p=p, n=n, W=W, bones=names, f=np.asarray(d['faces']), mats=mats, locked=d['locked'])
    return out


def skin(part, T):
    p = part['p']; ph = np.c_[p, np.ones(len(p))]
    M = np.einsum('vb,bij->vij', part['W'], np.stack([T[b] for b in part['bones']]))
    q = np.einsum('vij,vj->vi', M, ph)[:, :3]
    n = np.einsum('vij,vj->vi', M[:, :3, :3], part['n'])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return q, n


def frames_for(name, which):
    nfr = cat[name]['frames']
    sel = {'start': 0, 'mid': (nfr - 1) // 2, 'end': nfr - 1}
    return sorted({sel[w] for w in which})


def anim_list(spec, which=('start', 'mid', 'end')):
    out = []
    for item in spec.split(','):
        if item == 'ALL':
            out += [(r['name'], f) for r in anim.catalog if r['frames'] and r['nodes'] for f in frames_for(r['name'], which)]
        elif item == 'DIAG':
            out += [('DIAG:' + k, 0) for k in DIAG]
        elif item.startswith('DIAG:'):
            out.append((item, 0))
        elif item.startswith('EVERY:'):                       # EVERY:NAME:STEP
            _, nm, step = item.split(':')
            out += [(nm, f) for f in range(0, cat[nm]['frames'], int(step))]
        else:
            nm, fr = item.rsplit(':', 1)
            out.append((nm, int(fr)))
    return out


REST = global_matrices(rig)


def transforms(name, frame):
    if name.startswith('DIAG:'):
        pose = DIAG[name[5:]]; cache = {}

        def solve(i):
            if i in cache:
                return cache[i]
            s = rig[i]; local = matrix(s['trs'])
            for axis, deg in pose.get(s['name'], []):
                local[:3, :3] = local[:3, :3] @ rotation(axis, deg)
            par = s['hierarchy'][0]
            cache[i] = local if par in (65535, i) else solve(par) @ local
            return cache[i]
        return {s['name']: solve(i) @ np.linalg.inv(REST[s['name']]) for i, s in enumerate(rig)}
    return anim.transforms(rig, cat[name]['index'], frame)


if MODE == 'render':
    OUT = Path(argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    LOD = int(opt('--lod', 0))
    VIEWS = opt('--views', 'front,back,left,right,front34,back34').split(',')
    az = {'front': 0, 'back': 180, 'left': 90, 'right': -90, 'front34': -40, 'back34': 140, 'back34l': -140, 'front34l': 40}
    for name, frame in anim_list(opt('--anim')):
        T = transforms(name, frame)
        for tag, pkg in specs:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            sc = bpy.context.scene
            skinmat = bpy.data.materials.new('clay'); skinmat.diffuse_color = (0.80, 0.63, 0.54, 1)
            allp = []
            for pname, part in load(pkg, LOD).items():
                q, n = skin(part, T)
                keep = [i for i, m in enumerate(part['mats']) if m != 'HAIR_pubic']
                f = part['f'][keep][:, ::-1]                       # game -> Blender winding
                verts = q[:, [0, 2, 1]]; nn = n[:, [0, 2, 1]]
                me = bpy.data.meshes.new(pname); me.from_pydata(verts.tolist(), [], f.tolist()); me.update()
                for poly in me.polygons:
                    poly.use_smooth = True
                me.normals_split_custom_set_from_vertices(nn.tolist())
                me.materials.append(skinmat)
                ob = bpy.data.objects.new(pname, me); sc.collection.objects.link(ob)
                allp.append(verts[np.unique(f)])
            allp = np.concatenate(allp); lo, hi = allp.min(0), allp.max(0); c = (lo + hi) / 2
            span = float(max(hi[2] - lo[2], hi[0] - lo[0], hi[1] - lo[1])) * 1.08
            sc.render.engine = 'BLENDER_WORKBENCH'
            sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_specular_highlight = True
            sh.background_type = 'VIEWPORT'; sh.background_color = (0.30, 0.32, 0.36)
            sc.view_settings.view_transform = 'Standard'
            sc.render.resolution_x = 900; sc.render.resolution_y = 1100; sc.render.resolution_percentage = 100
            cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'; cd.ortho_scale = span
            cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
            for v in VIEWS:
                a = math.radians(az[v])
                cam.location = Vector(c) + 4 * Vector((math.sin(a), -math.cos(a), 0.12))
                cam.rotation_euler = (Vector(c) - cam.location).to_track_quat('-Z', 'Y').to_euler()
                sc.render.filepath = str(OUT / f'{tag}_LOD{LOD:02d}_{name.replace(":", "-")}_f{frame:03d}_{v}.png')
                bpy.ops.render.render(write_still=True)
        print('RENDERED', name, frame, flush=True)
    print('EAN_RENDER_DONE')

elif MODE == 'cross':
    OUT = Path(argv[1])
    lods = [int(x) for x in opt('--lods', '0').split(',')]
    which = opt('--frames', 'start,mid,end').split(',')
    poses = anim_list(opt('--anim', 'ALL'), which)

    def hits(a, b):
        """Points where edges of triangle b pierce triangle a (strict interior)."""
        e1 = a[1] - a[0]; e2 = a[2] - a[0]; pts = []
        for x, y in zip(b, np.roll(b, -1, axis=0)):
            d = y - x; h = np.cross(d, e2); det = float(np.dot(e1, h))
            if abs(det) < 1e-13: continue
            s = x - a[0]; u = float(np.dot(s, h) / det); q = np.cross(s, e1)
            v = float(np.dot(d, q) / det); t = float(np.dot(e2, q) / det)
            if 1e-6 < t < 1 - 1e-6 and 1e-6 < u and 1e-6 < v and u + v < 1 - 1e-6: pts.append(x + t * d)
        return pts

    def crosses(a, b):
        return bool(hits(a, b))

    def seg_len(a, b):
        pts = hits(a, b) + hits(b, a)
        if len(pts) < 2: return 0.0
        P = np.asarray(pts); return float(max(np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P))))

    def scan(parts, T):
        pp = []; ff = []; off = 0; seams = []
        posed = {}
        for pname in PARTS:
            part = parts[pname]; q, _ = skin(part, T); posed[pname] = q
            keep = [i for i, m in enumerate(part['mats']) if m != 'HAIR_pubic']
            pp.append(q); ff.append(part['f'][keep] + off); off += len(q)
        p = np.concatenate(pp); f = np.concatenate(ff)
        rest = np.concatenate([parts[n]['p'] for n in PARTS])
        _, weld = np.unique(np.round(rest, 6), axis=0, return_inverse=True)
        tree = BVHTree.FromPolygons([Vector(v) for v in p], f.tolist(), all_triangles=True, epsilon=0.)
        hit = {}
        for a, b in tree.overlap(tree):
            if a >= b or set(weld[f[a]]) & set(weld[f[b]]): continue
            if crosses(p[f[a]], p[f[b]]) or crosses(p[f[b]], p[f[a]]): hit[(a, b)] = seg_len(p[f[a]], p[f[b]])
        scan.rest_centroids = rest[f].mean(1)
        # seam gap: coincident rest vertices across parts must stay coincident
        gap = 0.0
        for a_, b_ in (('Bust', 'Pants'), ('Bust', 'Rist'), ('Pants', 'Boots')):
            pa = parts[a_]['p']; pb = parts[b_]['p']
            for i in parts[a_]['locked']:
                j = int(np.linalg.norm(pb - pa[i], axis=1).argmin())
                if np.linalg.norm(pb[j] - pa[i]) < 1e-6:
                    gap = max(gap, float(np.linalg.norm(posed[a_][i] - posed[b_][j])))
        return hit, gap

    (bt, bp), (ct, cp) = specs[0], specs[1]
    rep = {'base': bp, 'candidate': cp, 'ean': str(EAN), 'scope': 'Offline EAN keyframes + linear skinning of JSON weights; strict non-adjacent triangle crossings; no IK/physics.', 'rows': []}
    t0 = time.time()
    for lod in lods:
        B = load(bp, lod); C = load(cp, lod)
        for name, frame in poses:
            T = transforms(name, frame)
            hb, gb = scan(B, T); cen = scan.rest_centroids; hc, gc = scan(C, T)
            new = sorted(set(hc) - set(hb))
            # a new pair is a new clipping REGION only if no source crossing exists within 0.02 units (rest space)
            bc = np.asarray([(cen[a] + cen[b]) / 2 for a, b in hb]) if hb else np.zeros((0, 3))
            far = [x for x in new if not len(bc) or np.min(np.linalg.norm(bc - (cen[x[0]] + cen[x[1]]) / 2, axis=1)) > 0.02]
            lb, lc = sum(hb.values()), sum(hc.values())
            ok = not far and lc <= lb * 1.05 + 0.002 and gc < 1e-6
            rep['rows'].append({'lod': lod, 'anim': name, 'frame': frame, 'base_crossings': len(hb), 'cand_crossings': len(hc),
                                'new_crossings': [list(x) for x in new], 'new_region_pairs': [list(x) for x in far],
                                'resolved': len(set(hb) - set(hc)), 'intersection_length_base': lb, 'intersection_length_cand': lc,
                                'seam_gap_base': gb, 'seam_gap_cand': gc, 'severity_gate_pass': ok,
                                'worst_growth': sorted(([list(k), hb.get(k, 0.0), v] for k, v in hc.items()), key=lambda t: t[1] - t[2])[:5] if not ok else []})
            if new or not ok:
                print(f'LOD{lod} {name}:{frame} base {len(hb)}/{lb:.4f} cand {len(hc)}/{lc:.4f} new {len(new)} new-region {len(far)} {"PASS" if ok else "FAIL"}', flush=True)
        print(f'LOD{lod} done {len(poses)} poses, {time.time() - t0:.0f}s', flush=True)
        OUT.write_text(json.dumps(rep, indent=1))
    rep['summary'] = {'poses': len(poses), 'lods': lods, 'new_pairs_total': sum(len(r['new_crossings']) for r in rep['rows']),
                      'poses_with_new_pairs': sum(1 for r in rep['rows'] if r['new_crossings']),
                      'new_region_pairs_total': sum(len(r['new_region_pairs']) for r in rep['rows']),
                      'severity_gate_failures': sum(1 for r in rep['rows'] if not r['severity_gate_pass']),
                      'intersection_length_base': sum(r['intersection_length_base'] for r in rep['rows']),
                      'intersection_length_cand': sum(r['intersection_length_cand'] for r in rep['rows']),
                      'max_seam_gap_cand': max(r['seam_gap_cand'] for r in rep['rows'])}
    OUT.write_text(json.dumps(rep, indent=1))
    print('EAN_CROSS_DONE', json.dumps(rep['summary']))
