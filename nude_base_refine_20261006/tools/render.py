"""Clay / heat-map orthographic renders of the body (Cycles CPU; read-only on the .blend).

venv/bin/python render.py <file.blend> <jobs.json> <out_dir>
jobs.json: [{"name": str, "lod": 0, "pose": {bone: [[axis, deg]]} | null, "view": "front|back|left|right|front34|back34|top|bottom|custom",
             "center_h": h (normalized height) or "center": [x, y, z] (game axes), "scale": ortho size (model units),
             "res": 900, "heat": {"<object>": "path.npy"} (optional per-vertex 0..1 scalar), "elev": deg, "azim": deg,
             "hair": true, "lods_visible": [0]}]
"""
import bpy, sys, json, math
from pathlib import Path
import numpy as np
from mathutils import Matrix, Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_eval import set_pose

H = 1.0761; SOLE = -0.697
VIEWS = {'front': (0, 0), 'back': (180, 0), 'left': (90, 0), 'right': (-90, 0), 'front34': (35, 0), 'back34': (145, 0),
         'front34r': (-35, 0), 'back34r': (-145, 0), 'low': (0, -35), 'top': (0, 80)}


def clay():
    m = bpy.data.materials.new('QA_clay'); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    attr = nt.nodes.new('ShaderNodeAttribute'); attr.attribute_name = 'qa_heat'; attr.attribute_type = 'GEOMETRY'
    ramp = nt.nodes.new('ShaderNodeValToRGB'); r = ramp.color_ramp
    r.elements[0].position = 0.0; r.elements[0].color = (0.50, 0.44, 0.40, 1)
    r.elements[1].position = 1.0; r.elements[1].color = (0.85, 0.05, 0.02, 1)
    nt.links.new(attr.outputs['Fac'], ramp.inputs['Fac']); nt.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.55
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    hair = bpy.data.materials.new('QA_hair'); hair.diffuse_color = (0.08, 0.06, 0.05, 1); hair.use_nodes = True
    hair.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.05, 0.04, 0.03, 1)
    return m, hair


def main():
    src, jobs_path, out_dir = sys.argv[1:4]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=src)
    jobs = json.loads(Path(jobs_path).read_text())
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = 24; sc.cycles.use_denoising = False
    sc.render.film_transparent = False; sc.view_settings.view_transform = 'Standard'
    w = sc.world or bpy.data.worlds.new('W'); sc.world = w; w.use_nodes = True
    w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.32, 0.33, 0.35, 1); w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.30
    def unhide(lc):
        lc.exclude = False; lc.hide_viewport = False
        for c in lc.children:
            unhide(c)
    unhide(bpy.context.view_layer.layer_collection)
    clay_m, hair_m = clay()
    rig = next(o for o in bpy.data.objects if o.type == 'ARMATURE'); rig.data.pose_position = 'POSE'
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    orig_mats = {}
    for o in meshes:
        o.hide_viewport = False; o.hide_set(False)
        orig_mats[o.name] = [s.name for s in o.data.materials]
        for i, name in enumerate(orig_mats[o.name]):
            o.data.materials[i] = hair_m if 'HAIR' in name else clay_m
        if 'qa_heat' not in o.data.attributes:
            o.data.attributes.new('qa_heat', 'FLOAT', 'POINT')
    cam_d = bpy.data.cameras.new('qa_cam'); cam_d.type = 'ORTHO'; cam = bpy.data.objects.new('qa_cam', cam_d); sc.collection.objects.link(cam); sc.camera = cam
    key = bpy.data.lights.new('qa_key', 'SUN'); key.energy = float(__import__('os').environ.get('QA_KEY', 2.0)); key.angle = math.radians(12); kobj = bpy.data.objects.new('qa_key', key); sc.collection.objects.link(kobj)
    fill = bpy.data.lights.new('qa_fill', 'SUN'); fill.energy = 0.45; fobj = bpy.data.objects.new('qa_fill', fill); sc.collection.objects.link(fobj)
    for job in jobs:
        lods = job.get('lods_visible', [job.get('lod', 0)])
        for o in meshes:
            o.hide_render = int(o.name[-2:]) not in lods
            vals = np.zeros(len(o.data.vertices), np.float32)
            hp = (job.get('heat') or {}).get(o.name)
            if hp:
                vals = np.clip(np.load(hp).astype(np.float32), 0, 1)
            o.data.attributes['qa_heat'].data.foreach_set('value', vals)
            if not job.get('hair', True):
                for i, name in enumerate(orig_mats[o.name]):
                    if 'HAIR' in name:
                        pass
        set_pose(rig, job.get('pose') or {})
        az, el = VIEWS.get(job.get('view', 'front'), (0, 0))
        az = job.get('azim', az); el = job.get('elev', el)
        if 'center' in job:
            cx, cy, cz = job['center']
        else:
            cx, cy, cz = 0.0, job.get('center_h', 0.55) * H + SOLE, 0.0
        target = Vector((cx, cz, cy))  # game -> blender (x, z, y)
        a = math.radians(az); e = math.radians(el)
        d = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))  # front = camera at -Y
        cam.location = target + d * 4.0
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        cam_d.ortho_scale = job.get('scale', 1.3); cam_d.clip_end = 20
        kobj.rotation_euler = ((-(d + Vector((job.get('key_x', 0.35), 0.0, 0.9)))).normalized()).to_track_quat('-Z', 'Y').to_euler()
        fobj.rotation_euler = ((-(d + Vector((-0.8, 0.0, -0.2)))).normalized()).to_track_quat('-Z', 'Y').to_euler()
        res = job.get('res', 900); sc.render.resolution_x = int(res * job.get('aspect', 1.0)); sc.render.resolution_y = res
        sc.render.filepath = str(out_dir / (job['name'] + '.png'))
        bpy.ops.render.render(write_still=True)
        print('RENDERED', job['name'], flush=True)


if __name__ == '__main__':
    main()
