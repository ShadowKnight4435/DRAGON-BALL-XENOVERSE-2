"""Dump every mesh object and the rig of a .blend into a single .npz (read-only on the .blend).

venv/bin/python dump_blend.py <file.blend> <out.npz>
Arrays are stored in Blender axes. Game axes (x lateral, y up, z back+) = blender[:, [0, 2, 1]].
"""
import bpy, sys, json, hashlib
import numpy as np

src, out = sys.argv[1], sys.argv[2]
bpy.ops.wm.open_mainfile(filepath=src)
D = {}
meta = {'source': src, 'sha256': hashlib.sha256(open(src, 'rb').read()).hexdigest(), 'objects': {}}
for o in bpy.data.objects:
    if o.type != 'MESH':
        continue
    m = o.data
    n = o.name
    V = len(m.vertices); L = len(m.loops); P = len(m.polygons); E = len(m.edges)
    co = np.zeros(V * 3, np.float32); m.vertices.foreach_get('co', co)
    lv = np.zeros(L, np.int32); m.loops.foreach_get('vertex_index', lv)
    le = np.zeros(L, np.int32); m.loops.foreach_get('edge_index', le)
    ls = np.zeros(P, np.int32); m.polygons.foreach_get('loop_start', ls)
    lt = np.zeros(P, np.int32); m.polygons.foreach_get('loop_total', lt)
    mi = np.zeros(P, np.int32); m.polygons.foreach_get('material_index', mi)
    sm = np.zeros(P, bool); m.polygons.foreach_get('use_smooth', sm)
    ed = np.zeros(E * 2, np.int32); m.edges.foreach_get('vertices', ed)
    cn = np.zeros(L * 3, np.float32); m.corner_normals.foreach_get('vector', cn)
    raw = np.zeros(L * 2, np.int16); m.attributes['custom_normal'].data.foreach_get('value', raw)
    uv = np.zeros(L * 2, np.float32); m.uv_layers[0].data.foreach_get('uv', uv)
    se = np.zeros(E, bool)
    if 'sharp_edge' in m.attributes:
        m.attributes['sharp_edge'].data.foreach_get('value', se)
    G = len(o.vertex_groups)
    W = np.zeros((V, G), np.float32)
    for v in m.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    assert np.all(lt == 3), (n, 'non-triangle polygon')
    D[n + '/co'] = co.reshape(V, 3); D[n + '/tri'] = lv.reshape(P, 3); D[n + '/loop_edge'] = le
    D[n + '/mat'] = mi; D[n + '/smooth'] = sm; D[n + '/edges'] = ed.reshape(E, 2)
    D[n + '/cn'] = cn.reshape(L, 3); D[n + '/cn_raw'] = raw.reshape(L, 2); D[n + '/uv'] = uv.reshape(L, 2)
    D[n + '/sharp_edge'] = se; D[n + '/W'] = W
    meta['objects'][n] = {'materials': [s.name for s in m.materials], 'groups': [g.name for g in o.vertex_groups],
                          'props': {k: (o[k] if isinstance(o[k], (str, int, float)) else str(o[k])) for k in o.keys()},
                          'V': V, 'F': P, 'E': E, 'L': L}
rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
bones = [b.name for b in rig.data.bones]
D['rig/matrix_local'] = np.array([[list(r) for r in b.matrix_local] for b in rig.data.bones], np.float64)
D['rig/head'] = np.array([list(b.head_local) for b in rig.data.bones]); D['rig/tail'] = np.array([list(b.tail_local) for b in rig.data.bones])
meta['rig'] = {'name': rig.name, 'bones': bones, 'parents': [b.parent.name if b.parent else None for b in rig.data.bones]}
meta['scene'] = {k: bpy.context.scene[k] for k in bpy.context.scene.keys() if isinstance(bpy.context.scene[k], (str, int, float))}
D['meta'] = np.array(json.dumps(meta))
np.savez_compressed(out, **D)
print('DUMPED', len(meta['objects']), 'meshes', len(bones), 'bones ->', out)
