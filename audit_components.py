"""Baseline audit of unreferenced vertices, seam rings and the pubic-hair component (read-only).

python audit_components.py <package_dir> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np

PKG = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(PKG / 'authoring')); sys.path.insert(0, str(Path(__file__).resolve().parent))
from asset_io import emd

rep = {'package': str(PKG), 'lods': []}
for lod in range(4):
    ds = {d['part']: d for d in json.loads((PKG / f'authoring/lod{lod}_geometry.json').read_text())}
    row = {'lod': lod, 'parts': {}}
    for part, d in ds.items():
        p = np.asarray(d['positions']); f = np.asarray(d['faces']); mi = np.asarray(d['material_indices'])
        names = [m['name'] for m in d['materials']]
        used = np.zeros(len(p), bool); used[np.unique(f)] = True
        unref = np.where(~used)[0]
        fn = f'HUF_2998_{part}' + (f'_LOD{lod:02d}' if lod else '') + '.emd'
        subs = emd(PKG / 'data/chara/HUF' / fn)
        emd_pos = np.concatenate([np.asarray(s['positions']) for s in subs]) if subs else np.zeros((0, 3))
        info = []
        for v in unref:
            dmin = float(np.min(np.linalg.norm(emd_pos - p[v], axis=1))) if len(emd_pos) else None
            twin = np.where((np.linalg.norm(p - p[v], axis=1) < 1e-6) & used)[0]
            info.append({'index': int(v), 'position': p[v].round(5).tolist(), 'weights': d['weights'][v],
                         'nearest_emd_vertex_distance': dmin, 'coincident_referenced_vertices': twin.tolist(),
                         'locked': int(v) in set(d['locked'])})
        part_row = {'vertices': len(p), 'faces': len(f), 'unreferenced': info, 'locked_count': len(d['locked']),
                    'materials': {n: int((mi == k).sum()) for k, n in enumerate(names)}}
        if 'HAIR_pubic' in names:
            k = names.index('HAIR_pubic')
            hv = np.unique(f[mi == k]); sv = np.unique(f[mi != k])
            part_row['pubic'] = {'faces': int((mi == k).sum()), 'vertices': len(hv),
                                 'shares_vertices_with_skin': int(len(np.intersect1d(hv, sv))),
                                 'bounds': [p[hv].min(0).round(4).tolist(), p[hv].max(0).round(4).tolist()]}
        row['parts'][part] = part_row
    # seam rings: locked vertices shared (coincident) between modular parts
    def ring(a, b, sel):
        pa = np.asarray(ds[a]['positions']); pb = np.asarray(ds[b]['positions'])
        la = [i for i in ds[a]['locked'] if sel(pa[i])]
        pairs = []
        for i in la:
            dd = np.linalg.norm(pb - pa[i], axis=1); j = int(dd.argmin()); pairs.append((i, j, float(dd[j])))
        return {'count': len(la), 'left': sum(1 for i, _, _ in pairs if pa[i][0] < -1e-6),
                'right': sum(1 for i, _, _ in pairs if pa[i][0] > 1e-6), 'centre': sum(1 for i, _, _ in pairs if abs(pa[i][0]) <= 1e-6),
                'max_gap': max(g for _, _, g in pairs),
                'normals_equal': all(np.allclose(ds[a]['normals'][i], ds[b]['normals'][j], atol=1e-6) for i, j, _ in pairs),
                'weights_equal': all(ds[a]['weights'][i] == ds[b]['weights'][j] for i, j, _ in pairs)}
    row['seams'] = {'Bust-Pants (waist)': ring('Bust', 'Pants', lambda v: 0.05 < v[1] < 0.12),
                    'Bust-Rist (wrists)': ring('Bust', 'Rist', lambda v: abs(v[0]) > 0.4),
                    'Pants-Boots (ankles)': ring('Pants', 'Boots', lambda v: -0.55 < v[1] < -0.5)}
    rep['lods'].append(row)
OUT.write_text(json.dumps(rep, indent=1))
for row in rep['lods']:
    print(f"LOD{row['lod']}:", {k: (v['count'], v['left'], v['right'], round(v['max_gap'], 9), v['normals_equal'], v['weights_equal']) for k, v in row['seams'].items()})
    for part, pr in row['parts'].items():
        if pr['unreferenced']:
            print(f"   {part}: unreferenced", [(u['index'], u['position'], round(u['nearest_emd_vertex_distance'], 6), u['coincident_referenced_vertices']) for u in pr['unreferenced']])
        if 'pubic' in pr:
            print(f"   {part}: pubic", pr['pubic'])
