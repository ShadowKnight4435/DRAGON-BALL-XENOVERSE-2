"""Resolution-independent comparison of the external vulvar / inguinal contour: planar cross-sections
(horizontal slices at fixed h, plus the mid-sagittal slice) of LOD00 Pants skin for several sources.

python section_profiles.py <out.png> <tag>=<pkg_dir | emd:<dir>:<prefix> | tune:<params.json>:<iters>> [...]
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from asset_io import emd

H = 1.0761; SOLE = -0.697
V13 = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930\reference_fit_20261005\candidate_v13')


def load(src):
    if src.startswith('emd:'):
        folder, prefix = src[4:].rsplit(':', 1)
        P, F = [], []; off = 0
        for s in emd(Path(folder) / f'{prefix}_Pants.emd'):
            if 'HAIR' in s['name']:
                continue
            p = np.asarray(s['positions']); f = np.concatenate([np.asarray(g['indices']).reshape(-1, 3) for g in s['groups']])
            P.append(p); F.append(f + off); off += len(p)
        return np.concatenate(P), np.concatenate(F)
    if src.startswith('tune:'):
        import local_smooth as LS
        prm, it = src[5:].rsplit(':', 1); prm = json.loads(Path(prm).read_text()); prm['iterations'] = int(it)
        d = [x for x in json.loads((V13 / 'authoring/lod0_geometry.json').read_text()) if x['part'] == 'Pants'][0]
        p = np.asarray(d['positions']); f = np.asarray(d['faces']); mats = [d['materials'][m]['name'] for m in d['material_indices']]
        p = p + LS.lod0_delta({'p': p, 'f': f, 'mats': mats}, prm, d['locked'])
        skin = np.array([m != 'HAIR_pubic' for m in mats])
        return p, f[skin]
    d = [x for x in json.loads((Path(src) / 'authoring/lod0_geometry.json').read_text()) if x['part'] == 'Pants'][0]
    mats = [d['materials'][m]['name'] for m in d['material_indices']]
    skin = np.array([m != 'HAIR_pubic' for m in mats])
    return np.asarray(d['positions']), np.asarray(d['faces'])[skin]


def slice_mesh(p, f, axis, c):
    """Segments of the mesh cut by plane p[:,axis]==c."""
    s = p[:, axis] - c; segs = []
    for t in f:
        pts = []
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            if (s[a] < 0) != (s[b] < 0):
                u = s[a] / (s[a] - s[b]); pts.append(p[a] + u * (p[b] - p[a]))
        if len(pts) == 2:
            segs.append(pts)
    return np.asarray(segs)


if __name__ == '__main__':
    out = Path(sys.argv[1]); specs = [a.split('=', 1) for a in sys.argv[2:]]
    meshes = {t: load(s) for t, s in specs}
    hs = [0.570, 0.585, 0.595, 0.605, 0.615, 0.625, 0.635, 0.645]
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(3, 4, figsize=(20, 14))
    for k, hh in enumerate(hs):
        ax = axs.flat[k]; y = SOLE + hh * H
        for i, (t, (p, f)) in enumerate(meshes.items()):
            S = slice_mesh(p, f, 1, y)
            if not len(S):
                continue
            m = (np.abs(S[:, :, 0]).max(1) < 0.11) & (S[:, :, 2].max(1) < 0.04)
            for sg in S[m]:
                ax.plot(sg[:, 0], -sg[:, 2], color=cols[i], lw=1.4, label=t)
        h_, l_ = ax.get_legend_handles_labels(); u = dict(zip(l_, h_))
        ax.legend(u.values(), u.keys(), fontsize=8); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_title(f'horizontal slice h={hh:.3f} (x vs -z, front up)'); ax.set_xlim(-0.11, 0.11)
    for k, xs in enumerate((0.0, 0.012, 0.024, 0.040)):
        ax = axs.flat[8 + k]
        for i, (t, (p, f)) in enumerate(meshes.items()):
            S = slice_mesh(p, f, 0, xs)
            m = (S[:, :, 1].min(1) > SOLE + 0.52 * H) & (S[:, :, 1].max(1) < SOLE + 0.70 * H) & (S[:, :, 2].max(1) < 0.02)
            for sg in S[m]:
                ax.plot(-sg[:, 2], (sg[:, 1] - SOLE) / H, color=cols[i], lw=1.4, label=t)
        h_, l_ = ax.get_legend_handles_labels(); u = dict(zip(l_, h_))
        ax.legend(u.values(), u.keys(), fontsize=8); ax.grid(alpha=.3)
        ax.set_title(f'sagittal slice x={xs:.3f} (-z vs h)')
        ax.set_aspect(1 / H)
    fig.tight_layout(); fig.savefig(out, dpi=80)
    print('wrote', out)
