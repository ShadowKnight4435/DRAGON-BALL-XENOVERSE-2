"""Tune the local smoothing on v13 LOD00 Pants against the .x2m crease metrics (no files written except a report).

python tune_vulva.py <params.json> [iters,...]
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import local_smooth as LS
import region_compare as RC  # noqa: reuses dihedrals/stats/REG (module body only defines functions when imported with argv guard)

V13 = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930\reference_fit_20261005\candidate_v13')
P = json.loads(Path(sys.argv[1]).read_text())
d = [x for x in json.loads((V13 / 'authoring/lod0_geometry.json').read_text()) if x['part'] == 'Pants'][0]
p = np.asarray(d['positions']); f = np.asarray(d['faces']); mats = [d['materials'][m]['name'] for m in d['material_indices']]
skin = np.array([m != 'HAIR_pubic' for m in mats])
iters_list = [int(i) for i in sys.argv[2].split(',')] if len(sys.argv) > 2 else [P['iterations']]
for it in iters_list:
    P['iterations'] = it
    D = LS.lod0_delta({'p': p, 'f': f, 'mats': mats}, P, d['locked'])
    q = p + D
    key, inv = np.unique(np.round(q, 6), axis=0, return_inverse=True)
    rows = RC.dihedrals(key, inv[f[skin]])
    out = {k: RC.stats(rows, RC.REG[k]) for k in ('vulva_central', 'vulva_crease_line', 'inguinal_folds', 'pubic_mound')}
    prof = {}
    for mid, ang, _ in rows:
        if RC.REG['vulva_crease_line'](mid):
            b = f'{round(RC.h(mid) / 0.01) * 0.01:.2f}'; prof[b] = max(prof.get(b, 0.0), max(0.0, -ang))
    mv = np.linalg.norm(D, axis=1)
    print(f'iters {it:2d}  moved {int((mv > 1e-7).sum())} max {mv.max():.4f}')
    for k, v in out.items():
        print(f'   {k:18s} p95 {v["abs_p95"]:6.1f}  concave-max {v["concave_max"]:6.1f}  convex>40 {v["convex_over_40"]:2d}  concave>40 {v["concave_over_40"]:2d}')
    print('   crease profile', {k: round(v, 1) for k, v in sorted(prof.items())})
    print('   mirror', RC.mirror_error(key, RC.REG['vulva_central']))
