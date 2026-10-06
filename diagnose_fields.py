"""Per-edit triangle distortion diagnostics for the field definitions (no files written except a report).

python diagnose_fields.py [fields.json]
"""
import json, sys, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE / 'build_candidate.py').read_text()
head = src.split("changes = []; summary = []")[0]
head = head.replace("assert not CAND.exists(), CAND", "")
head = head.replace("(CAND / 'authoring').mkdir(parents=True)", "").replace("(CAND / 'data/chara/HUF').mkdir(parents=True)", "")
sys.argv = ['x', '_diag_'] + sys.argv[1:]
ns = {'__file__': str(HERE / 'build_candidate.py'), '__name__': 'diag'}
exec(compile(head, 'build_candidate_head', 'exec'), ns)
BASE = ns['BASE']; field = ns['field']
for lod in range(4):
    for d in json.loads((BASE / f'authoring/lod{lod}_geometry.json').read_text()):
        if d['part'] not in ('Bust', 'Pants'):
            continue
        p = np.asarray(d['positions']); f = np.asarray(d['faces'])
        D, parts = field(d['part'], p)
        locked = np.asarray(d['locked'], int); D[locked] = 0
        q = p + D
        on = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        nn = np.cross(q[f[:, 1]] - q[f[:, 0]], q[f[:, 2]] - q[f[:, 0]])
        ratio = np.linalg.norm(nn, axis=1) / np.maximum(np.linalg.norm(on, axis=1), 1e-30)
        cos = (on * nn).sum(1) / np.maximum(np.linalg.norm(on, axis=1) * np.linalg.norm(nn, axis=1), 1e-30)
        worst = np.argsort(ratio)[:4]
        flips = np.where(cos <= 0.5)[0]
        print(f'LOD{lod} {d["part"]:5s} area ratio min {ratio.min():.3f} max {ratio.max():.3f}  faces turned >60deg: {len(flips)}')
        for fi in list(worst) + list(flips[:4]):
            c = p[f[fi]].mean(0)
            mags = {k: float(v[f[fi]].max()) for k, v in parts.items()}
            dom = max(mags, key=mags.get)
            print(f'    face {fi:5d} ratio {ratio[fi]:.3f} cos {cos[fi]:+.3f} centre ({c[0]:+.4f},{c[1]:+.4f},{c[2]:+.4f}) '
                  f'h {(c[1] + 0.697) / 1.0761:.3f} dominant {dom} {mags[dom]:.4f} '
                  f'edge {np.linalg.norm(p[f[fi]] - np.roll(p[f[fi]], 1, 0), axis=1).round(4).tolist()}')
