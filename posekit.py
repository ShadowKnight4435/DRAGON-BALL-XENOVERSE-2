"""Shared posing helpers: JSON geometry loading, EAN / diagnostic transforms, linear skinning (game axes)."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930')
import sys
sys.path.insert(0, str(ROOT / 'deliverable/HUF_REVAMP511_ReferenceBody/authoring'))
from asset_io import skeleton
from skinning import AnimationFile, global_matrices, matrix, rotation

EAN = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\HUF Assets\HUF_000_REVAMP_v5.1.1\HUF\HUF.ean')
ESK = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\HUF Assets\HUF_000_REVAMP_v5.1.1\HUF\HUF_000.esk')
PARTS = ('Bust', 'Pants', 'Rist', 'Boots')
H = 1.0761; SOLE = -0.697

_anim = None; _rig = None; _rest = None
DIAG = json.loads((Path(__file__).resolve().parent / 'poses_extended.json').read_text()); DIAG.pop('_note', None)


def rig():
    global _rig, _rest
    if _rig is None:
        _rig = skeleton(ESK); _rest = global_matrices(_rig)
    return _rig


def anim():
    global _anim
    if _anim is None:
        _anim = AnimationFile(EAN)
    return _anim


def load(pkg, lod):
    out = {}
    for d in json.loads((Path(pkg) / f'authoring/lod{lod}_geometry.json').read_text()):
        p = np.asarray(d['positions'], float); n = np.asarray(d['normals'], float)
        names = sorted({b for w in d['weights'] for b in w}); idx = {b: i for i, b in enumerate(names)}
        W = np.zeros((len(p), len(names)))
        for vi, w in enumerate(d['weights']):
            for b, x in w.items():
                W[vi, idx[b]] = x
        out[d['part']] = dict(p=p, n=n, W=W, bones=names, f=np.asarray(d['faces']),
                              mats=[d['materials'][m]['name'] for m in d['material_indices']], locked=list(d['locked']))
    return out


def transforms(name, frame=0):
    r = rig()
    if name == 'rest':
        return None
    if name.startswith('DIAG:'):
        pose = DIAG[name[5:]]; cache = {}

        def solve(i):
            if i in cache:
                return cache[i]
            s = r[i]; local = matrix(s['trs'])
            for axis, deg in pose.get(s['name'], []):
                local[:3, :3] = local[:3, :3] @ rotation(axis, deg)
            par = s['hierarchy'][0]
            cache[i] = local if par in (65535, i) else solve(par) @ local
            return cache[i]
        return {s['name']: solve(i) @ np.linalg.inv(_rest[s['name']]) for i, s in enumerate(r)}
    a = anim(); cat = {x['name']: x for x in a.catalog}
    return a.transforms(r, cat[name]['index'], frame)


def skin(part, T):
    if T is None:
        return part['p'].copy(), part['n'] / np.maximum(np.linalg.norm(part['n'], axis=1, keepdims=True), 1e-12)
    p = part['p']; ph = np.c_[p, np.ones(len(p))]
    M = np.einsum('vb,bij->vij', part['W'], np.stack([T[b] for b in part['bones']]))
    q = np.einsum('vij,vj->vi', M, ph)[:, :3]
    n = np.einsum('vij,vj->vi', M[:, :3, :3], part['n'])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return q, n


def hh(y):
    return (np.asarray(y) - SOLE) / H
