"""Rest-pose region close-ups (stored normals, Workbench) for JSON packages or raw EMD folders. Read-only.

blender -b -P render_regions.py -- <out_dir> <tag>=<pkg_dir | emd:<dir>:<prefix>> [...] [--views name,...]
"""
from pathlib import Path
import bpy, json, math, sys
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_io import emd

argv = sys.argv[sys.argv.index('--') + 1:]
OUT = Path(argv[0]); OUT.mkdir(parents=True, exist_ok=True)
specs = [a.split('=', 1) for a in argv[1:] if '=' in a and not a.startswith('--')]
# (name, target game-axes (x,y,z), azimuth deg (0=front), elevation deg, ortho scale)
VIEWS = {
    'groin_front': ((0, -0.03, -0.06), 0, 0, 0.16), 'groin_low': ((0, -0.04, -0.06), 0, -40, 0.16),
    'groin_34': ((0, -0.03, -0.05), -35, -15, 0.18), 'groin_side': ((0, -0.03, -0.05), -90, -10, 0.18),
    'areola_front': ((0.075, 0.222, -0.13), 0, 0, 0.09), 'areola_side': ((0.075, 0.222, -0.13), -90, 0, 0.09),
    'areola_34': ((0.075, 0.222, -0.13), -40, 10, 0.10), 'breast_side': ((0.07, 0.23, -0.09), -90, 0, 0.22),
    'hip_front': ((0, -0.03, 0.0), 0, 0, 0.42), 'hip_back': ((0, -0.03, 0.0), 180, 0, 0.42),
    'hip_side': ((0, -0.03, 0.0), -90, 0, 0.42), 'hip_34back': ((0, -0.03, 0.0), 145, 0, 0.42),
    'vulva_front': ((0.0, -0.055, -0.05), 0, -10, 0.075), 'vulva_low': ((0.0, -0.062, -0.035), 0, -45, 0.075),
    'vulva_34': ((0.01, -0.058, -0.045), -35, -20, 0.085), 'vulva_side': ((0.0, -0.058, -0.04), -80, -10, 0.09),
    'ing_front': ((0.04, -0.02, -0.06), 0, 0, 0.12), 'ing_34': ((0.05, -0.02, -0.06), -40, -10, 0.13), 'ing_side': ((0.05, -0.02, -0.06), -80, 0, 0.14),
    'flank': ((-0.09, 0.12, 0.0), -100, 0, 0.22), 'flank_back': ((-0.08, 0.12, 0.03), -150, 0, 0.22),
    'hip_lat': ((-0.15, -0.02, 0.02), -100, 0, 0.24), 'hip_lat_back': ((-0.13, -0.03, 0.05), -135, 5, 0.24),
}
sel = argv[argv.index('--views') + 1].split(',') if '--views' in argv else list(VIEWS)


def meshes_from_pkg(pkg):
    out = []
    for d in json.loads((Path(pkg) / 'authoring/lod0_geometry.json').read_text()):
        p = np.asarray(d['positions']); n = np.asarray(d['normals'], float); f = np.asarray(d['faces'])
        mats = [d['materials'][m]['name'] for m in d['material_indices']]
        out.append((d['part'], p, n, f, mats))
    return out


def meshes_from_emd(folder, prefix):
    out = []
    for part in ('Bust', 'Pants', 'Rist', 'Boots'):
        fn = Path(folder) / f'{prefix}_{part}.emd'
        if not fn.exists():
            continue
        for s in emd(fn):
            p = np.asarray(s['positions']); n = np.asarray(s['normals'], float)
            f = np.concatenate([np.asarray(g['indices']).reshape(-1, 3) for g in s['groups']])
            out.append((part, p, n, f, [s['name']] * len(f)))
    return out


for tag, src in specs:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    clay = bpy.data.materials.new('clay'); clay.diffuse_color = (0.80, 0.63, 0.54, 1)
    tint = bpy.data.materials.new('tint'); tint.diffuse_color = (0.68, 0.45, 0.42, 1)
    hair = bpy.data.materials.new('hair'); hair.diffuse_color = (0.15, 0.12, 0.11, 1)
    parts = meshes_from_emd(*src[4:].rsplit(':', 1)) if src.startswith('emd:') else meshes_from_pkg(src)
    for i, (pn, p, n, f, mats) in enumerate(parts):
        v = p[:, [0, 2, 1]]; nn = n[:, [0, 2, 1]]; nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        me = bpy.data.meshes.new(f'{pn}{i}'); me.from_pydata(v.tolist(), [], f[:, ::-1].tolist()); me.update()
        for poly, m in zip(me.polygons, mats):
            poly.use_smooth = True
            poly.material_index = 2 if m == 'HAIR_pubic' else (1 if m in ('SKIN_nipple', 'SKIN_bust_CB') else 0)
        me.normals_split_custom_set_from_vertices(nn.tolist())
        for m in (clay, tint, hair):
            me.materials.append(m)
        ob = bpy.data.objects.new(f'{pn}{i}', me); sc.collection.objects.link(ob)
        if '--wire' in argv:
            wm = bpy.data.materials.new('wire'); wm.diffuse_color = (0.05, 0.05, 0.08, 1)
            w2 = bpy.data.objects.new(f'{pn}{i}_wire', me.copy()); sc.collection.objects.link(w2)
            w2.data.materials.clear(); w2.data.materials.append(wm)
            mod = w2.modifiers.new('w', 'WIREFRAME'); mod.thickness = 0.00035; mod.use_replace = True
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_specular_highlight = True
    if '--matcap' in argv:      # e.g. toon.exr (cel-like bands) or reflection_check_vertical.exr (zebra continuity)
        sh.light = 'MATCAP'; sh.studio_light = argv[argv.index('--matcap') + 1]; sh.color_type = 'SINGLE'
        sh.single_color = (0.85, 0.85, 0.85)
    sh.background_type = 'VIEWPORT'; sh.background_color = (0.30, 0.32, 0.36)
    sc.view_settings.view_transform = 'Standard'
    sc.render.resolution_x = 640; sc.render.resolution_y = 640; sc.render.resolution_percentage = 100
    cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    for name in sel:
        (gx, gy, gz), az, el, scale = VIEWS[name]
        t = Vector((gx, gz, gy)); a = math.radians(az); e = math.radians(el)
        cd.ortho_scale = scale
        cam.location = t + 3 * Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        cam.rotation_euler = (t - cam.location).to_track_quat('-Z', 'Y').to_euler()
        mc = ('_' + argv[argv.index('--matcap') + 1].split('.')[0]) if '--matcap' in argv else ''
        sc.render.filepath = str(OUT / f'{tag}_{name}{mc}.png')
        bpy.ops.render.render(write_still=True)
    print('REGIONS_DONE', tag, flush=True)
