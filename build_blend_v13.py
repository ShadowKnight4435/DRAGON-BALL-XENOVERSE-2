"""Write a candidate's positions and normals into a new copy of the v13 production .blend (structural authority).

blender -b <v13.blend> -P build_blend_v13.py -- <candidate_dir>
The v13 file is never saved. Only vertices whose position or stored normal differ between v13 and the candidate are
written; protected mesh data (faces, edges, UVs, groups, weights, materials, smoothing, modifiers) and the rig must
stay identical, and custom-normal records outside the touched fans must stay byte-exact.
"""
from pathlib import Path
import bpy, hashlib, json, sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
sys.path.insert(0, str(HERE.parent / 'chunli_physique_20261004'))
from preserve_normals import preserve_directions

V13 = HERE.parent / 'reference_fit_20261005/candidate_v13'
SOURCE = V13 / 'editable/HUF_REVAMP511_ChunLiInspired.blend'
DEST = CAND / 'editable/HUF_REVAMP511_NudeBaseRefine.blend'
SOURCE_PIN = '48a898567622e9bdd4452e8a5035d81fb163899d738a022519145a2f6f873524'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def signature(obj):
    m = obj.data
    d = {'polygons': [list(p.vertices) for p in m.polygons], 'uvs': [[list(x.uv) for x in l.data] for l in m.uv_layers],
         'groups': [g.name for g in obj.vertex_groups], 'weights': [[[g.group, g.weight] for g in v.groups] for v in m.vertices],
         'materials': [s.name for s in m.materials], 'indices': [p.material_index for p in m.polygons],
         'smooth': [p.use_smooth for p in m.polygons], 'edges': [list(e.vertices) for e in m.edges],
         'modifiers': [(x.name, x.type, getattr(getattr(x, 'object', None), 'name', None)) for x in obj.modifiers]}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


assert Path(bpy.data.filepath).resolve() == SOURCE.resolve(), bpy.data.filepath
assert sha(SOURCE) == SOURCE_PIN
assert not DEST.exists(), DEST
before = {o.name: signature(o) for o in bpy.data.objects if o.type == 'MESH'}
rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
rig_before = {b.name: [list(r) for r in b.matrix_local] for b in rig.data.bones}
results = []
for lod in range(4):
    vd = {d['part']: d for d in json.loads((V13 / f'authoring/lod{lod}_geometry.json').read_text())}
    cd = {d['part']: d for d in json.loads((CAND / f'authoring/lod{lod}_geometry.json').read_text())}
    for part in ('Bust', 'Pants', 'Rist', 'Boots'):
        obj = bpy.data.objects[f'HUF_{part}_LOD{lod:02d}']; m = obj.data; assert m.users == 1
        pv = np.asarray(vd[part]['positions'])[:, [0, 2, 1]]; pc = np.asarray(cd[part]['positions'])[:, [0, 2, 1]]
        nv = np.asarray(vd[part]['normals'], float); nc = np.asarray(cd[part]['normals'], float)
        co = np.asarray([list(v.co) for v in m.vertices])
        # baseline authority check: the .blend must be exactly v13 before anything is written
        assert np.array_equal(co, np.asarray(pv, dtype=np.float32)), (obj.name, 'blend is not the v13 baseline')
        moved = np.where(np.any(np.asarray(pc, np.float32) != np.asarray(pv, np.float32), axis=1))[0]
        renorm = np.where(np.any(np.abs(nc - nv) > 1e-12, axis=1))[0]
        if not len(moved) and not len(renorm):
            continue
        f = np.asarray(cd[part]['faces'])
        oldraw = np.asarray([list(x.value) for x in m.attributes['custom_normal'].data])
        oldvec = np.asarray([list(x.vector) for x in m.corner_normals])
        for i in moved:
            m.vertices[int(i)].co = pc[i]
        m.update()
        nrm = nc[:, [0, 2, 1]]; nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        updated = set(map(int, renorm))
        loop_vertex = np.asarray([l.vertex_index for l in m.loops])
        target = oldvec.copy(); sel = np.isin(loop_vertex, list(updated)); target[sel] = nrm[loop_vertex[sel]]
        fan = set(map(int, np.unique(f[np.any(np.isin(f, moved), axis=1)]))) if len(moved) else set()
        allowed = fan | updated
        err, optimized = preserve_directions(m, target, allowed)
        m.update()
        newraw = np.asarray([list(x.value) for x in m.attributes['custom_normal'].data])
        vec = np.asarray([list(x.vector) for x in m.corner_normals])
        outside = ~np.isin(loop_vertex, list(allowed))
        assert np.array_equal(newraw[outside], oldraw[outside]), obj.name
        err = float(np.max(np.linalg.norm(vec - target, axis=1))); assert err < 0.0005, (obj.name, err)
        assert np.array_equal(np.asarray([list(v.co) for v in m.vertices]), np.asarray(pc, dtype=np.float32))
        assert signature(obj) == before[obj.name], obj.name
        results.append({'object': obj.name, 'moved_vs_v13': int(len(moved)), 'normal_updated_vs_v13': int(len(updated)),
                        'normal_vector_error_max': err, 'unrelated_custom_normal_records_exact': True,
                        'protected_mesh_data_exact': True, 'quantized_fan_corrections': len(optimized)})
        print('APPLIED', obj.name, 'moved vs v13', len(moved), 'normals', len(updated), 'err', round(err, 6), flush=True)
assert all(signature(bpy.data.objects[n]) == h for n, h in before.items())
assert {b.name: [list(r) for r in b.matrix_local] for b in rig.data.bones} == rig_before
note = bpy.data.texts.new('NUDE_BASE_REFINE_20261006')
note.write('Localized refinement of the v13 production baseline (sha256 ' + SOURCE_PIN + ').\n'
           'Authority: this .blend (structure) > 3D reference body (proportion/silhouette) > supplied .x2m (restricted local\n'
           'detail: hip transition, vulvar contour, areola). No .x2m geometry, topology, UVs, textures, materials, rig or weights used.\n'
           'Changes vs v13: hip narrowing replaced by an affine cross-section scale (no z-band plane; midline guard keeps the\n'
           'verified vulvar contour); waist minimum moved off the first loop above the Bust/Pants seam (smooth contour);\n'
           'forearm taper ends before the wrist seam; lower-glute fill lateral falloff widened (0.16 -> 0.20).\n'
           'Unchanged by evidence: vulva (within 0.002 of the .x2m, symmetric), areola/nipple, bust lowering, IMF.\n'
           'Topology, vertex order, UVs, weights, groups, materials, rig, IDs and modular seams preserved. Not installed.\n')
scene = bpy.context.scene
scene['body_status'] = 'Unverified - Runtime Validation Required'
scene['current_baseline_sha256'] = SOURCE_PIN
scene['anatomy_refinement'] = 'Nude-base localized refinement of v13 (2026-10-06)'
DEST.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(DEST), copy=True)
assert sha(SOURCE) == SOURCE_PIN
(CAND / 'blender_change_manifest.json').write_text(json.dumps(
    {'source': str(SOURCE), 'source_sha256': SOURCE_PIN, 'output': str(DEST), 'output_sha256': sha(DEST),
     'source_preserved': True, 'rig_rest_matrices_exact': True, 'changes': results}, indent=2))
print('CANDIDATE BLEND SAVED', DEST)
