"""Write the candidate's positions and normals into a new copy of the source .blend.

blender -b <source.blend> -P build_blend.py -- <candidate_dir>
The source file is never saved. Protected mesh data (faces, UVs, groups, weights, materials, modifiers)
must stay identical; custom normals change only on loops of moved / normal-updated vertices.
"""
from pathlib import Path
import bpy, hashlib, json, sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAND = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
sys.path.insert(0, str(HERE.parent / 'chunli_physique_20261004'))
from preserve_normals import preserve_directions

BASE = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930'
            r'\chunli_physique_20261004\deliverable\HUF_REVAMP511_ChunLiInspired_20261005'
            r'\HUF_REVAMP511_ChunLiInspired_20261005')
SOURCE = BASE / 'editable/HUF_REVAMP511_ChunLiInspired.blend'
DEST = CAND / 'editable/HUF_REVAMP511_ChunLiInspired.blend'
SOURCE_PIN = '26e6bdb1f4db45cd20c7e51a6457c0b60e7c01e8d405933a4b4b1d6a1df61048'


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
rig_before = {b.name: [list(r) for r in b.matrix_local] for b in next(o for o in bpy.data.objects if o.type == 'ARMATURE').data.bones}
results = []
for c in json.loads((CAND / 'localized_changes.json').read_text()):
    lod, part = c['lod'], c['part']
    obj = bpy.data.objects[f'HUF_{part}_LOD{lod:02d}']; m = obj.data; assert m.users == 1
    d = next(d for d in json.loads((CAND / f'authoring/lod{lod}_geometry.json').read_text()) if d['part'] == part)
    oldraw = np.asarray([list(x.value) for x in m.attributes['custom_normal'].data])
    oldvec = np.asarray([list(x.vector) for x in m.corner_normals])
    p = np.asarray(d['positions'])[:, [0, 2, 1]]
    nrm = np.asarray(d['normals'])[:, [0, 2, 1]]
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    for i in c['changed_source_vertex_ids']:
        m.vertices[i].co = p[i]
    m.update()
    updated = set(c['normal_updated_vertex_ids'])
    loop_vertex = np.asarray([l.vertex_index for l in m.loops])
    target = oldvec.copy()
    sel = np.isin(loop_vertex, list(updated))
    target[sel] = nrm[loop_vertex[sel]]
    allowed = set(c['reencoded_blender_normal_vertex_ids']) | updated
    err, optimized = preserve_directions(m, target, allowed)
    m.update()
    newraw = np.asarray([list(x.value) for x in m.attributes['custom_normal'].data])
    vec = np.asarray([list(x.vector) for x in m.corner_normals])
    outside = ~np.isin(loop_vertex, list(allowed))
    assert np.array_equal(newraw[outside], oldraw[outside]), obj.name
    err = float(np.max(np.linalg.norm(vec - target, axis=1))); assert err < 0.0005, (obj.name, err)
    assert np.array_equal(np.asarray([list(v.co) for v in m.vertices]), np.asarray(p, dtype=np.float32))
    assert signature(obj) == before[obj.name], obj.name
    results.append({'object': obj.name, 'moved_vertices': len(c['changed_source_vertex_ids']),
                    'normal_updated_vertices': len(updated), 'normal_vector_error_max': err,
                    'unrelated_custom_normal_records_exact': True, 'protected_mesh_data_exact': True,
                    'quantized_fan_corrections': len(optimized)})
    print('APPLIED', obj.name, 'moved', len(c['changed_source_vertex_ids']), 'normal err', err, flush=True)
assert all(signature(bpy.data.objects[n]) == h for n, h in before.items())
rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
assert {b.name: [list(r) for r in b.matrix_local] for b in rig.data.bones} == rig_before
note = bpy.data.texts.new('REFERENCE_FIT_20261005')
note.write('Reference-fit derivative of the approved Chun-Li source (sha256 ' + SOURCE_PIN + ').\n'
           'Driven by the supplied turnaround reference sheet (reference_fit_20261005/reference).\n'
           'Edits (smooth spatial fields, identical on all four LODs): hip/trochanter lateral narrowing; waist minimum moved\n'
           'down to just above the navel with a broader lower ribcage; breast mound lowered (areola translated rigidly, no\n'
           'scale change); moderate lower-glute fill; forearm dorsal/palmar thickening.\n'
           'Height unchanged: body-internal proportions already match the reference; head-based ratios are not usable\n'
           '(the HUF anime face differs from the reference face). Stored normals rotate with the surface; seams stay exact.\n'
           'Topology, vertex order, UVs, weights, groups, materials, rig, IDs and modular seams preserved. Not installed.\n')
scene = bpy.context.scene
scene['body_status'] = 'Unverified - Runtime Validation Required'
scene['current_baseline_sha256'] = SOURCE_PIN
scene['anatomy_refinement'] = 'Reference-fit Chun-Li physique (turnaround sheet, 2026-10-05)'
DEST.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(DEST), copy=True)
assert sha(SOURCE) == SOURCE_PIN
(CAND / 'blender_change_manifest.json').write_text(json.dumps(
    {'source': str(SOURCE), 'source_sha256': SOURCE_PIN, 'output': str(DEST), 'output_sha256': sha(DEST),
     'source_preserved': True, 'rig_rest_matrices_exact': True, 'changes': results}, indent=2))
print('CANDIDATE BLEND SAVED', DEST)
