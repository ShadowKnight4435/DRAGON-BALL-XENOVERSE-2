"""Pose the preserved rig with diagnostic poses and evaluate deformed meshes + strict triangle intersections.

venv/bin/python pose_eval.py <file.blend> <poses.json> <out.npz> [pose names...]
Writes, per pose and LOD, deformed vertex positions (game axes) for every part and the list of intersecting
skin-triangle pairs (non-adjacent by welded position) found with mathutils BVHTree.overlap (exact tri-tri test).
Read-only on the .blend (never saved).
"""
import bpy, sys, json, math
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

PARTS = ('Bust', 'Pants', 'Rist', 'Boots')


def set_pose(rig, spec):
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'; pb.rotation_quaternion = (1, 0, 0, 0); pb.location = (0, 0, 0); pb.scale = (1, 1, 1)
    for bone, rots in spec.items():
        pb = rig.pose.bones[bone]; B = pb.bone.matrix_local.to_3x3()
        R = Matrix.Identity(3)
        for axis, deg in rots:
            R = Matrix.Rotation(math.radians(deg), 3, axis) @ R
        pb.rotation_quaternion = (B.inverted() @ R @ B).to_quaternion()
    bpy.context.view_layer.update()


def evaluated(obj):
    dg = bpy.context.evaluated_depsgraph_get(); e = obj.evaluated_get(dg); m = e.to_mesh()
    co = np.zeros(len(m.vertices) * 3, np.float32); m.vertices.foreach_get('co', co); e.to_mesh_clear()
    return co.reshape(-1, 3).astype(float)[:, [0, 2, 1]]


def skin_tris(obj):
    m = obj.data; t = np.array([list(p.vertices) for p in m.polygons]); mi = np.array([p.material_index for p in m.polygons])
    names = [s.name for s in m.materials]
    keep = np.array(['HAIR' not in names[i] for i in mi])
    return t[keep]


def intersections(P, F):
    """Exact intersecting triangle pairs that share no welded vertex."""
    key = np.round(P, 6); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    tree = BVHTree.FromPolygons([Vector(x) for x in P], F.tolist(), all_triangles=True)
    pairs = tree.overlap(tree)
    W = inv[F]; out = []
    for a, b in pairs:
        if a >= b:
            continue
        if set(W[a]) & set(W[b]):
            continue
        out.append((a, b))
    return np.array(out, int).reshape(-1, 2)


def main():
    src, poses_path, out = sys.argv[1:4]; only = sys.argv[4:]
    bpy.ops.wm.open_mainfile(filepath=src)
    poses = json.loads(open(poses_path).read()); poses.pop('_note', None)
    poses = {'rest': {}, **poses}
    if only:
        poses = {k: v for k, v in poses.items() if k in only or k == 'rest'}
    rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    rig.data.pose_position = 'POSE'
    # LOD01-03 collections are hidden by default; hidden objects are not evaluated by the depsgraph.
    def unhide(lc):
        lc.exclude = False; lc.hide_viewport = False; lc.collection.hide_viewport = False
        for c in lc.children:
            unhide(c)
    unhide(bpy.context.view_layer.layer_collection)
    for o in bpy.data.objects:
        o.hide_viewport = False; o.hide_set(False)
    D = {}; summary = {}
    tris = {o.name: skin_tris(o) for o in bpy.data.objects if o.type == 'MESH'}
    for name, spec in poses.items():
        set_pose(rig, spec); summary[name] = {}
        for lod in range(4):
            P = []; F = []; owner = []; off = 0
            for part in PARTS:
                o = bpy.data.objects[f'HUF_{part}_LOD{lod:02d}']; q = evaluated(o)
                D[f'{name}/{lod}/{part}'] = q.astype(np.float32)
                P.append(q); F.append(tris[o.name] + off); owner += [part] * len(tris[o.name]); off += len(q)
            P = np.concatenate(P); F = np.concatenate(F); owner = np.array(owner)
            X = intersections(P, F)
            D[f'{name}/{lod}/xpairs'] = X
            D[f'{name}/{lod}/xcentroids'] = (P[F[X[:, 0]]].mean(1) + P[F[X[:, 1]]].mean(1)) / 2 if len(X) else np.zeros((0, 3))
            D[f'{name}/{lod}/xowners'] = np.array([owner[a] + '-' + owner[b] for a, b in X]) if len(X) else np.zeros(0, 'U16')
            summary[name][lod] = int(len(X))
        print('POSE', name, summary[name], flush=True)
    for part in PARTS:
        for lod in range(4):
            D[f'tris/{lod}/{part}'] = tris[f'HUF_{part}_LOD{lod:02d}']
    D['summary'] = np.array(json.dumps(summary))
    np.savez_compressed(out, **D)


if __name__ == '__main__':
    main()
