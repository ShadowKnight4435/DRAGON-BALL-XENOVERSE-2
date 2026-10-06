"""Compare stored-normal rotation with true surface rotation (area-weighted geometric normals).

python check_normals.py <candidate_dir>
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CAND = HERE / sys.argv[1]
BASE = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930'
            r'\chunli_physique_20261004\deliverable\HUF_REVAMP511_ChunLiInspired_20261005'
            r'\HUF_REVAMP511_ChunLiInspired_20261005')


def geo_normals(p, f):
    fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def ang(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


report = []
for lod in range(4):
    old = {d['part']: d for d in json.loads((BASE / f'authoring/lod{lod}_geometry.json').read_text())}
    new = {d['part']: d for d in json.loads((CAND / f'authoring/lod{lod}_geometry.json').read_text())}
    for part in ('Bust', 'Pants'):
        a, b = old[part], new[part]
        f = np.asarray(a['faces']); p0 = np.asarray(a['positions']); p1 = np.asarray(b['positions'])
        n0 = np.asarray(a['normals']); n1 = np.asarray(b['normals'])
        moved = np.linalg.norm(p1 - p0, axis=1) > 1e-8
        g0 = geo_normals(p0, f); g1 = geo_normals(p1, f)
        surf_turn = ang(g0, g1); stored_turn = ang(n0, n1)
        dev0 = ang(n0, g0); dev1 = ang(n1, g1)          # stored-vs-geometric deviation before/after
        worse = dev1 - dev0
        i = int(np.argmax(stored_turn))
        row = dict(lod=lod, part=part, moved=int(moved.sum()),
                   surface_turn_max=float(surf_turn[moved].max()), stored_turn_max=float(stored_turn.max()),
                   mismatch_p95=float(np.percentile(np.abs(surf_turn - stored_turn)[moved], 95)),
                   deviation_increase_max=float(worse[moved].max()), deviation_increase_p99=float(np.percentile(worse[moved], 99)),
                   deviation_before_p99=float(np.percentile(dev0, 99)), deviation_after_p99=float(np.percentile(dev1, 99)),
                   max_turn_vertex=i, max_turn_position=p0[i].round(4).tolist(), max_turn_h=float((p0[i, 1] + 0.697) / 1.0761),
                   max_turn_surface=float(surf_turn[i]))
        report.append(row)
        print(f"LOD{lod} {part:5s} moved {row['moved']:5d} surface-turn max {row['surface_turn_max']:5.1f} stored-turn max "
              f"{row['stored_turn_max']:5.1f} |diff| p95 {row['mismatch_p95']:.2f}  stored-vs-geo deviation p99 "
              f"{row['deviation_before_p99']:.1f} -> {row['deviation_after_p99']:.1f} (worst increase {row['deviation_increase_max']:.1f})"
              f"  max-turn at {row['max_turn_position']} h {row['max_turn_h']:.3f} (surface {row['max_turn_surface']:.1f})")
(CAND / 'normal_consistency.json').write_text(json.dumps(report, indent=1))
