"""Write per-vertex displacements into a COPY of the production .blend; everything else stays byte-exact.

venv/bin/python apply_candidate.py <baseline.blend> <displacements.npz> <out.blend> [note.txt]
displacements.npz: key = object name, value = (V, 3) float displacement in GAME axes (x, y up, z back).

Only vertex positions change, plus the custom-normal records of the touched fans (corners of every vertex that shares
a face with a moved vertex). Target corner normals rotate with the surface: each fan vertex's stored normal is turned
by the minimal rotation between its old and new area-weighted geometric normal; locked modular (seam) vertices keep
their stored normal exactly so both sides of every seam still match. Custom-normal records outside the touched fans,
faces, edges, sharp edges, UVs, materials, smoothing, groups/weights, modifiers, custom props and the rig are verified
identical before saving. The baseline file is never written.
"""
import bpy, sys, json, hashlib
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arr(coll, attr, n, k, dt):
    a = np.zeros(n * k, dt); coll.foreach_get(attr, a); return a.reshape(n, k) if k > 1 else a


def protected(obj):
    m = obj.data; L = len(m.loops); P = len(m.polygons); E = len(m.edges); V = len(m.vertices)
    h = hashlib.sha256()
    for a in (arr(m.loops, 'vertex_index', L, 1, np.int32), arr(m.loops, 'edge_index', L, 1, np.int32),
              arr(m.polygons, 'material_index', P, 1, np.int32), arr(m.polygons, 'use_smooth', P, 1, bool),
              arr(m.edges, 'vertices', E, 2, np.int32), arr(m.uv_layers[0].data, 'uv', L, 2, np.float32)):
        h.update(a.tobytes())
    se = np.zeros(E, bool)
    if 'sharp_edge' in m.attributes:
        m.attributes['sharp_edge'].data.foreach_get('value', se)
    h.update(se.tobytes())
    w = [[(g.group, round(g.weight, 9)) for g in v.groups] for v in m.vertices]
    h.update(json.dumps(w).encode())
    h.update(json.dumps({'mats': [s.name for s in m.materials], 'groups': [g.name for g in obj.vertex_groups],
                         'mods': [(x.name, x.type, getattr(getattr(x, 'object', None), 'name', None)) for x in obj.modifiers],
                         'props': {k: str(obj[k]) for k in obj.keys()}, 'parent': obj.parent.name if obj.parent else None,
                         'coll': [c.name for c in obj.users_collection], 'mesh': m.name, 'attrs': sorted((a.name, a.domain, a.data_type) for a in m.attributes if not a.name.startswith('.'))}, sort_keys=True).encode())
    return h.hexdigest()


def vnormals(co, tri):
    fn = np.cross(co[tri[:, 1]] - co[tri[:, 0]], co[tri[:, 2]] - co[tri[:, 0]]); g = np.zeros_like(co)
    for k in range(3):
        np.add.at(g, tri[:, k], fn)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)


def rotate(v, a, b):
    """Minimal rotation taking unit a -> unit b, applied to rows of v (Rodrigues)."""
    axis = np.cross(a, b); s = np.linalg.norm(axis, axis=1); c = np.clip((a * b).sum(1), -1, 1)
    k = axis / np.maximum(s, 1e-30)[:, None]
    out = v * c[:, None] + np.cross(k, v) * s[:, None] + k * (k * v).sum(1, keepdims=True) * (1 - c)[:, None]
    return np.where((s > 1e-12)[:, None], out, v)


def main():
    src, disp_path, out = sys.argv[1:4]
    note = Path(sys.argv[4]).read_text() if len(sys.argv) > 4 else ''
    assert not Path(out).exists(), out
    src_sha = sha(src)
    bpy.ops.wm.open_mainfile(filepath=src)
    disp = np.load(disp_path)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    before = {o.name: protected(o) for o in meshes}
    rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    rig_before = json.dumps([[b.name, b.parent.name if b.parent else None, [list(r) for r in b.matrix_local]] for b in rig.data.bones])
    report = []
    for name in disp.files:
        o = bpy.data.objects[name]; m = o.data
        V = len(m.vertices); L = len(m.loops); P = len(m.polygons)
        D = np.asarray(disp[name], float)[:, [0, 2, 1]]          # game -> blender axes
        assert D.shape == (V, 3), (name, D.shape)
        co_pre = arr(m.vertices, 'co', V, 3, np.float32)
        moved = np.where(np.any((co_pre.astype(float) + D).astype(np.float32) != co_pre, axis=1))[0]   # real float32 change only
        if not len(moved):
            continue
        D[np.setdiff1d(np.arange(V), moved)] = 0.0
        locked = set(json.loads(o['locked_modular_vertices']))
        assert not (set(moved.tolist()) & locked), (name, 'locked seam vertex displaced', sorted(set(moved.tolist()) & locked)[:10])
        co0 = arr(m.vertices, 'co', V, 3, np.float32).astype(float)
        tri = arr(m.loops, 'vertex_index', L, 1, np.int32).reshape(P, 3)
        lv = tri.ravel()
        raw0 = arr(m.attributes['custom_normal'].data, 'value', L, 2, np.int16)
        vec0 = arr(m.corner_normals, 'vector', L, 3, np.float32).astype(float)
        se0 = np.zeros(len(m.edges), bool); m.attributes['sharp_edge'].data.foreach_get('value', se0)
        co1 = (co0 + D).astype(np.float32)
        fan = np.unique(tri[np.isin(tri, moved).any(1)])
        g0 = vnormals(co0, tri); g1 = vnormals(co1.astype(float), tri)
        target = vec0.copy()
        touched = np.isin(lv, fan)
        rot_v = np.array([v not in locked for v in lv])
        sel = touched & rot_v
        target[sel] = rotate(vec0[sel], g0[lv[sel]], g1[lv[sel]])
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        m.vertices.foreach_set('co', co1.ravel()); m.update()
        m.normals_split_custom_set(target.tolist()); m.update()
        raw_new = arr(m.attributes['custom_normal'].data, 'value', L, 2, np.int16)
        raw = raw0.copy(); raw[touched] = raw_new[touched]
        m.attributes['custom_normal'].data.foreach_set('value', raw.ravel())
        m.attributes['sharp_edge'].data.foreach_set('value', se0)
        m.update()
        vec1 = arr(m.corner_normals, 'vector', L, 3, np.float32).astype(float)
        raw_chk = arr(m.attributes['custom_normal'].data, 'value', L, 2, np.int16)
        assert np.array_equal(raw_chk[~touched], raw0[~touched]), (name, 'custom-normal record outside the touched fan changed')
        err_out = float(np.abs(vec1[~touched] - vec0[~touched]).max()) if (~touched).any() else 0.0
        err_t = float(np.linalg.norm(vec1[touched] - target[touched], axis=1).max())
        lk = np.array([v in locked for v in lv])
        err_lock = float(np.linalg.norm(vec1[touched & lk] - vec0[touched & lk], axis=1).max()) if (touched & lk).any() else 0.0
        assert err_out < 1e-6, (name, 'decoded normal outside fan changed', err_out)
        assert err_t < 2e-3, (name, 'normal encoding error', err_t)
        assert err_lock < 2e-3, (name, 'seam normal changed', err_lock)
        assert np.array_equal(arr(m.vertices, 'co', V, 3, np.float32), co1)
        turn = np.degrees(np.arccos(np.clip((vec1[touched] * vec0[touched]).sum(1), -1, 1)))
        report.append({'object': name, 'moved_vertices': int(len(moved)), 'max_displacement': float(np.linalg.norm(D, axis=1).max()),
                       'fan_vertices': int(len(fan)), 'touched_corners': int(touched.sum()), 'normal_encoding_error_max': err_t,
                       'seam_normal_error_max': err_lock, 'normal_turn_deg_max': float(turn.max()),
                       'untouched_custom_normal_records_exact': True, 'moved_ids': moved.tolist()})
        print('APPLIED', name, 'moved', len(moved), 'max', round(report[-1]['max_displacement'], 6), 'fan', len(fan), 'enc', round(err_t, 6), 'turn', round(float(turn.max()), 3), flush=True)
    for o in meshes:
        assert protected(o) == before[o.name], (o.name, 'protected data changed')
    assert json.dumps([[b.name, b.parent.name if b.parent else None, [list(r) for r in b.matrix_local]] for b in rig.data.bones]) == rig_before
    if note:
        t = bpy.data.texts.new('NUDE_BASE_LOCAL_REFINE_20261006B'); t.write(note)
    bpy.context.scene['anatomy_refinement'] = 'Localized nude-base crease/contour refinement (2026-10-06B)'
    bpy.context.scene['previous_baseline_sha256'] = src_sha
    bpy.context.scene['body_status'] = 'Unverified - Runtime Validation Required'
    bpy.ops.wm.save_as_mainfile(filepath=out, copy=True)
    assert sha(src) == src_sha
    Path(out + '.manifest.json').write_text(json.dumps({'source': src, 'source_sha256': src_sha, 'output': out, 'output_sha256': sha(out),
                                                       'protected_data_identical': True, 'rig_identical': True, 'changes': report}, indent=1))
    print('SAVED', out)


if __name__ == '__main__':
    main()
