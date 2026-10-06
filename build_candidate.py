"""Reference-fit candidate: smooth spatial displacement fields applied identically to all four LODs.

python build_candidate.py <label> [fields.json]

Only vertex positions change. Locked modular seams (waist, wrists, ankles, neck) are kept exact.
Native EMDs are patched in place: position bytes plus any bounding volume that must expand.
The source package is read-only.
"""
from pathlib import Path
import copy, hashlib, json, struct, sys
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930'
            r'\chunli_physique_20261004\deliverable\HUF_REVAMP511_ChunLiInspired_20261005'
            r'\HUF_REVAMP511_ChunLiInspired_20261005')
LABEL = sys.argv[1]
FIELDS = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else HERE / 'fields.json').read_text())
CAND = HERE / LABEL
assert not CAND.exists(), CAND
(CAND / 'authoring').mkdir(parents=True)
(CAND / 'data/chara/HUF').mkdir(parents=True)
sys.path.insert(0, str(BASE / 'authoring'))
from emd_writer import write_emd
from asset_io import emd


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def smooth(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10)


def step3(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def curve(knots, values):
    f = PchipInterpolator(knots, values, extrapolate=False)
    return lambda x: np.nan_to_num(f(np.asarray(x, float)), nan=0.0)


def halfwidth(d, ys, xlimit=None, ymask=None):
    """Silhouette half-width |x| of a mesh at each height by exact triangle-plane intersection."""
    p = np.asarray(d['positions']); f = np.asarray(d['faces'])
    keep = np.array([d['materials'][m]['name'] != 'HAIR_pubic' for m in d['material_indices']])
    t = p[f[keep]]
    out = []
    for y in ys:
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            k = (t[:, a, 1] - y) * (t[:, b, 1] - y) < 0
            aa = t[k, a]; bb = t[k, b]; s = (y - aa[:, 1]) / (bb[:, 1] - aa[:, 1])
            pts.append(aa[:, 0] + (bb[:, 0] - aa[:, 0]) * s)
        x = np.abs(np.concatenate(pts))
        if xlimit is not None:
            x = x[x < xlimit]
        out.append(x.max() if len(x) else np.nan)
    out = np.asarray(out)
    good = ~np.isnan(out)
    out = np.interp(ys, ys[good], out[good])
    k = np.ones(5) / 5
    return np.convolve(np.pad(out, 2, mode='edge'), k, mode='valid')


high = {d['part']: d for d in json.loads((BASE / 'authoring/lod0_geometry.json').read_text())}
# Fixed spatial reference functions from LOD00 (same field for every LOD).
ys_p = np.arange(-0.20, 0.0901, 0.002)
hw_pants = halfwidth(high['Pants'], ys_p)
ys_b = np.arange(0.075, 0.2001, 0.002)
hw_bust = halfwidth(high['Bust'], ys_b, xlimit=0.2)
bp = np.asarray(high['Bust']['positions']); bf = np.asarray(high['Bust']['faces'])
xs_a = np.arange(0.27, 0.4561, 0.005)
arm_c = []
tb = bp[bf]
for x0 in xs_a:
    yy = []
    for sgn in (-1, 1):  # both arms, cross-section plane |x| = x0
        tx = tb[:, :, 0] * sgn
        for a, b in ((0, 1), (1, 2), (2, 0)):
            k = (tx[:, a] - x0) * (tx[:, b] - x0) < 0
            s = (x0 - tx[k, a]) / (tx[k, b] - tx[k, a])
            yy.append(tb[k, a, 1] + (tb[k, b, 1] - tb[k, a, 1]) * s)
    yy = np.concatenate(yy)
    arm_c.append([(yy.min() + yy.max()) / 2, (yy.max() - yy.min()) / 2])
arm_c = np.asarray(arm_c)
arm_c[:, 0] = np.convolve(np.pad(arm_c[:, 0], 2, mode='edge'), np.ones(5) / 5, mode='valid')
arm_c[:, 1] = np.convolve(np.pad(arm_c[:, 1], 2, mode='edge'), np.ones(5) / 5, mode='valid')

F = FIELDS
SNAP = F.get('_snap', 0.0)   # opt-in: displacements of the new terms below this are numerically meaningless -> exactly zero
# Native templates patched in place (default: the Chun-Li source). Geometry is always computed from the source JSON;
# a production-baseline template (e.g. v13) makes the payload differ from that baseline only where vertices changed.
TEMPLATE = Path(F.get('_template_dir', BASE))
# Pelvis cross-section centre (z) per height, from LOD00 Pants (used by the radial hip correction).
_pp = np.asarray(high['Pants']['positions'])
ys_c = np.arange(-0.20, 0.0901, 0.005)
zc_p = np.array([(lambda s: (s[:, 2].min() + s[:, 2].max()) / 2 if len(s) else np.nan)(_pp[(np.abs(_pp[:, 1] - yy) < 0.006) & (np.abs(_pp[:, 0]) < 0.2)]) for yy in ys_c])
zc_p = np.interp(ys_c, ys_c[~np.isnan(zc_p)], zc_p[~np.isnan(zc_p)])
zc_p = np.convolve(np.pad(zc_p, 3, mode='edge'), np.ones(7) / 7, mode='valid')


def bust_drop(y, BL):
    y = np.asarray(y, float)
    d = np.zeros_like(y)
    d += np.where(y < BL['y_imf'], BL['imf_drop'] * step3(BL['y_below_start'], BL['y_imf'], y), 0)
    mid = (y >= BL['y_imf']) & (y < BL['y_plateau_lo'])
    d = np.where(mid, BL['imf_drop'] + (BL['apex_drop'] - BL['imf_drop']) * step3(BL['y_imf'], BL['y_plateau_lo'], y), d)
    d = np.where((y >= BL['y_plateau_lo']) & (y <= BL['y_plateau_hi']), BL['apex_drop'], d)
    d = np.where(y > BL['y_plateau_hi'], BL['apex_drop'] * (1 - step3(BL['y_plateau_hi'], BL['y_top'], y)), d)
    return d


def baseline_v13_terms(part, p, D, parts):
    """Field terms carried unchanged from the v13 production baseline (bust lowering, glute fill, v13 hip)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    if part == 'Pants' and 'hip_narrowing' in F:
        HN = F['hip_narrowing']; hipA = curve(HN['knots_y'], HN['inward'])
        w = step3(HN['lateral_start'], 1.1, np.abs(x) / np.interp(y, ys_p, hw_pants))
        w = w * (1 - HN['anterior_protection'] * smooth(HN['anterior_z_free'], HN['anterior_z_protected'], z))
        w = w * (1 - smooth(HN.get('posterior_z_free', 0.9), HN.get('posterior_z_protected', 1.0), z) * smooth(HN.get('posterior_y_free', -0.11), HN.get('posterior_y_protected', -0.14), y))
        dx = -np.sign(x) * hipA(y) * w
        D[:, 0] += dx; parts['hip_narrowing'] = np.abs(dx)
    if part == 'Pants' and 'glute_lower_fill' in F:
        GL = F['glute_lower_fill']; gluteG = curve(GL['knots_y'], GL['backward'])
        wp = smooth(GL['posterior_z_zero'], GL['posterior_z_full'], z)
        wl = 1 - smooth(GL['lateral_full'], GL['lateral_zero'], np.abs(x))
        perineum = (1 - smooth(GL['perineum_x_zero'], GL['perineum_x_full'], np.abs(x))) * smooth(GL['perineum_y_free'], GL['perineum_y_locked'], y)
        dz = gluteG(y) * wp * wl * (1 - perineum)
        D[:, 2] += dz; parts['glute_lower_fill'] = np.abs(dz)
    if part == 'Bust' and 'bust_lowering' in F:
        BL = F['bust_lowering']; torso = np.abs(x) < 0.2
        RL = 1 - smooth(BL['radial_plateau'], BL['radial_outer'], np.abs(x + BL['centre_x']))
        RR = 1 - smooth(BL['radial_plateau'], BL['radial_outer'], np.abs(x - BL['centre_x']))
        R = 1 - (1 - RL) * (1 - RR)
        Z = smooth(BL['front_z_zero'], BL['front_z_full'], z)
        if 'below_z_zero' in BL:
            Zb = smooth(BL['below_z_zero'], BL['below_z_full'], z)
            blend = step3(BL['y_imf'] - 0.02, BL['y_imf'] + 0.005, y)
            Z = Z * blend + Zb * (1 - blend)
        dy = -bust_drop(y, BL) * R * Z * torso
        D[:, 1] += dy; parts['bust_lowering'] = np.abs(dy)


def window(theta, a0, a1, b0, b1):
    """1 inside [a1, b0], smooth falloff to 0 at a0 (below) and b1 (above); degrees."""
    return smooth(a0, a1, theta) * (1 - smooth(b0, b1, theta))


def field(part, p):
    """Return (displacement, per-edit magnitudes) for positions p (game axes). Absent sections are inactive."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    D = np.zeros_like(p)
    parts = {}
    baseline_v13_terms(part, p, D, parts)
    if part == 'Pants' and 'hip_radial' in F:
        HR = F['hip_radial']; A = curve(HR['knots_y'], HR['inward'])(y)
        zc = np.interp(y, ys_c, zc_p)
        th = np.degrees(np.arctan2(z - zc, np.abs(x)))          # 0 = lateral, + toward posterior, - toward anterior
        wa = window(th, HR['anterior_zero_deg'], HR['anterior_full_deg'], HR['posterior_full_deg'], HR['posterior_zero_deg'])
        wr = step3(HR['radial_start'], HR['radial_full'], np.abs(x) / np.interp(y, ys_p, hw_pants))
        m = A * wa * wr
        r = np.hypot(np.abs(x), z - zc); r = np.maximum(r, 1e-9)
        D[:, 0] += -np.sign(x) * m * np.abs(x) / r              # radial shrink toward the pelvis axis (keeps lateral roundness)
        D[:, 2] += -m * (z - zc) / r
        parts['hip_radial'] = m
    if part == 'Pants' and 'hip_xscale' in F:
        HX = F['hip_xscale']; A = curve(HX['knots_y'], HX['inward'])(y)
        # affine horizontal scale of the pelvis cross-section about the midline: a smooth elliptical squash,
        # no angular window (no shear band); silhouette point moves by A, centre line not at all
        dx = -A * x / np.interp(y, ys_p, hw_pants)
        if 'anterior_residual' in HX:   # broad (low-shear) attenuation toward the anterior hip crease
            dx = dx * (1 - (1 - HX['anterior_residual']) * smooth(HX['anterior_z_free'], HX['anterior_z_min'], z))
        if 'midline_x_full' in HX:      # leave the verified vulvar / perineal midline untouched (smooth, low-shear)
            dx = dx * smooth(HX.get('midline_x_zero', 0.0), HX['midline_x_full'], np.abs(x))
        dx = np.where(np.abs(dx) < SNAP, 0.0, dx)
        D[:, 0] += dx; parts['hip_xscale'] = np.abs(dx)
    if part == 'Bust':
        if 'waist_ribcage' in F:
            WB = curve(F['waist_ribcage']['knots_y'], F['waist_ribcage']['outward'])
            torso = np.abs(x) < 0.2
            w = step3(0.0, 1.1, np.abs(x) / np.interp(y, ys_b, hw_bust))
            WR = F['waist_ribcage']
            if 'anterior_residual' in WR:   # broad (low-shear) attenuation toward the anterior waist (flexion folds)
                w = w * (1 - (1 - WR['anterior_residual']) * smooth(WR['anterior_z_free'], WR['anterior_z_min'], z))
            dx = np.where(torso, np.sign(x) * WB(y) * w, 0)
            dx = np.where(np.abs(dx) < SNAP, 0.0, dx)
            D[:, 0] += dx; parts['waist_ribcage'] = np.abs(dx)
        if 'forearm_thickening' in F:
            FA = F['forearm_thickening']
            armF = curve(FA['knots_absx'], FA['outward']); armF_top = curve(FA['top_knots_absx'], FA['top_outward'])
            ax = np.abs(x)
            on_arm = (ax > 0.27) & (ax < 0.46)
            cy = np.interp(ax, xs_a, arm_c[:, 0]); ry = np.interp(ax, xs_a, arm_c[:, 1])
            s = np.clip((y - cy) / ry, -1.2, 1.2)
            dya = np.where(on_arm, np.where(s > 0, armF_top(ax), armF(ax)) * s, 0)
            dya = np.where(np.abs(dya) < SNAP, 0.0, dya)
            D[:, 1] += dya; parts['forearm_thickening'] = np.abs(dya)
    return D, parts


class LocalOffsets:
    """Explicit LOD00 Pants vertex offsets (e.g. crease refinement solved on the post-field LOD00 surface), carried to
    LOD01-03 and to the pubic-hair cards: exact for coincident positions, otherwise a Gaussian-weighted average of the
    LOD00 skin offsets (zeros included, so the region never overshoots)."""

    def __init__(self, spec):
        from scipy.spatial import cKDTree
        o = json.loads(Path(spec['file']).read_text()); self.sigma = spec.get('sigma', 0.004)
        d0 = high['Pants']; p0 = np.asarray(d0['positions'])
        D0, _ = field('Pants', p0); D0[np.asarray(d0['locked'], int)] = 0
        q0 = p0 + D0
        skin_v = np.unique(np.asarray(d0['faces'])[[d0['materials'][m]['name'] != 'HAIR_pubic' for m in d0['material_indices']]])
        DL = np.zeros_like(q0)
        for k, v in o['displacement'].items():
            k = int(k)
            assert np.linalg.norm(q0[k] - np.asarray(o['lod0_rest_positions'][str(k)])) < 1e-9, ('offsets were solved on a different surface', k)
            assert k in set(skin_v.tolist()), k
            DL[k] = v
        self.Q = q0[skin_v]; self.DL = DL[skin_v]; self.tree = cKDTree(self.Q); self.stats = {}

    def displacement_for(self, lod, q):
        out = np.zeros_like(q); dist, idx = self.tree.query(q)
        exact = dist < 1e-6; out[exact] = self.DL[idx[exact]]
        n_interp = 0
        for i in np.where(~exact)[0]:
            nb = self.tree.query_ball_point(q[i], 3 * self.sigma)
            if not nb:
                continue
            w = np.exp(-(np.linalg.norm(self.Q[nb] - q[i], axis=1) / self.sigma) ** 2)
            v = (w[:, None] * self.DL[nb]).sum(0) / max(w.sum(), 1e-30)
            if np.linalg.norm(v) > 0:
                out[i] = v; n_interp += 1
        self.stats[lod] = {'exact_nonzero': int((np.linalg.norm(out[exact], axis=1) > 0).sum()), 'interpolated_nonzero': n_interp}
        return out


LO = LocalOffsets(F['local_offsets']) if 'local_offsets' in F else None
changes = []; summary = []
for lod in range(4):
    ds = json.loads((BASE / f'authoring/lod{lod}_geometry.json').read_text())
    newds = copy.deepcopy(ds)
    for d in newds:
        part = d['part']
        fn = f'HUF_2998_{part}' + (f'_LOD{lod:02d}' if lod else '') + '.emd'
        src = TEMPLATE / 'data/chara/HUF' / fn
        raw = src.read_bytes()
        p = np.asarray(d['positions']); f = np.asarray(d['faces'])
        D, parts = field(part, p)
        if part == 'Pants' and LO is not None:
            Dz = D.copy(); Dz[np.asarray(d['locked'], int)] = 0
            dl = LO.displacement_for(lod, p + Dz)
            D += dl; parts['local_crease_offsets'] = np.linalg.norm(dl, axis=1)
        locked = np.asarray(d['locked'], int)
        locked_field = float(np.linalg.norm(D[locked], axis=1).max()) if len(locked) else 0.0
        D[locked] = 0
        q = p + D
        moved = np.linalg.norm(q - p, axis=1) > 1e-8
        q[~moved] = p[~moved]
        d['positions'] = q.tolist()
        # Normals follow the deformed surface: each moved vertex's stored (artist/custom) normal is rotated by
        # the minimal rotation of its area-weighted geometric normal, preserving the original offset between the
        # stored and geometric normals. Locked seam normals stay exact so both modular sides still match.
        n0 = np.asarray(d['normals'], float)
        n1 = n0.copy()
        if moved.any():
            def geo(pp):
                fn_ = np.cross(pp[f[:, 1]] - pp[f[:, 0]], pp[f[:, 2]] - pp[f[:, 0]])
                g = np.zeros_like(pp)
                for k in range(3):
                    np.add.at(g, f[:, k], fn_)
                return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-30)
            g0 = geo(p); g1 = geo(q)
            axis = np.cross(g0, g1); s = np.linalg.norm(axis, axis=1); c = (g0 * g1).sum(1)
            k = axis / np.maximum(s, 1e-30)[:, None]
            # Rodrigues: v' = v c + (k x v) s + k (k.v)(1-c)
            nt = n0 * c[:, None] + np.cross(k, n0) * s[:, None] + k * (k * n0).sum(1, keepdims=True) * (1 - c)[:, None]
            rot = moved & (s > 1e-12)
            rot[locked] = False
            n1[rot] = nt[rot]
            d['normals'] = n1.tolist()
        cosn = (n0 * n1).sum(1) / np.maximum(np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1), 1e-30)
        normal_turn = np.degrees(np.arccos(np.clip(cosn, -1, 1)))
        if not moved.any():
            (CAND / 'data/chara/HUF' / fn).write_bytes(raw)
            continue
        oldn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        newn = np.cross(q[f[:, 1]] - q[f[:, 0]], q[f[:, 2]] - q[f[:, 0]])
        cosang = (oldn * newn).sum(1) / np.maximum(np.linalg.norm(oldn, axis=1) * np.linalg.norm(newn, axis=1), 1e-30)
        assert np.all(cosang > 0), 'Face orientation reversed'
        area_ratio = np.linalg.norm(newn, axis=1) / np.maximum(np.linalg.norm(oldn, axis=1), 1e-30)
        trial = write_emd(d); aa = emd(trial); bb = emd(raw)
        patched = bytearray(raw); ranges = []; allowed = np.zeros(len(raw), bool)
        for a, b in zip(aa, bb):
            for k in a:
                if k not in ('positions', 'normals', 'vo', 'mesh_offset', 'sub_offset', 'groups'):
                    assert a[k] == b[k], (fn, k)
            # normals may differ only on vertices of moved source vertices (checked via byte ranges below)
            assert [(g['indices'], g['bones']) for g in a['groups']] == [(g['indices'], g['bones']) for g in b['groups']]
            for i in range(a['count']):
                ss = a['vo'] + i * a['stride']; dd = b['vo'] + i * b['stride']
                if trial[ss:ss + 12] != raw[dd:dd + 12]:
                    patched[dd:dd + 12] = trial[ss:ss + 12]; allowed[dd:dd + 12] = True; ranges.append([dd, 12, 'position'])
                if trial[ss + 12:ss + 18] != raw[dd + 12:dd + 18]:
                    patched[dd + 12:dd + 18] = trial[ss + 12:ss + 18]; allowed[dd + 12:dd + 18] = True
                    ranges.append([dd + 12, 6, 'normal'])
        bounds = {}
        for a, b in zip(aa, bb):
            bounds[b['sub_offset']] = list(a['positions'])
            bounds.setdefault(b['mesh_offset'], []).extend(a['positions'])
        for off, pts in bounds.items():
            pts = np.asarray(pts); box = np.asarray(struct.unpack_from('<12f', raw, off)).reshape(3, 4)
            lo = np.minimum(box[1, :3], pts.min(0)); hi = np.maximum(box[2, :3], pts.max(0))
            if np.array_equal(lo, box[1, :3]) and np.array_equal(hi, box[2, :3]):
                continue
            for j, vec in enumerate(((lo + hi) / 2, lo, hi)):
                struct.pack_into('<3f', patched, off + j * 16, *vec); allowed[off + j * 16:off + j * 16 + 12] = True
                ranges.append([off + j * 16, 12, 'bounding_volume'])
        diff = np.frombuffer(raw, 'u1') != np.frombuffer(patched, 'u1')
        assert not np.any(diff & ~allowed)
        dest = CAND / 'data/chara/HUF' / fn; dest.write_bytes(patched)
        changes.append({'lod': lod, 'part': part, 'file': fn,
                        'changed_source_vertex_ids': np.where(moved)[0].tolist(),
                        'reencoded_blender_normal_vertex_ids': np.unique(f[np.any(moved[f], axis=1)]).tolist(),
                        'max_displacement': float(np.linalg.norm(q - p, axis=1).max()),
                        'per_edit_max': {k: float(v.max()) for k, v in parts.items()},
                        'field_at_locked_before_zeroing': locked_field,
                        'area_ratio_min': float(area_ratio.min()), 'area_ratio_max': float(area_ratio.max()),
                        'normal_updated_vertex_ids': np.where(normal_turn > 1e-6)[0].tolist(),
                        'normal_turn_deg_max': float(normal_turn.max()),
                        'normal_turn_deg_p95_moved': float(np.percentile(normal_turn[moved], 95)),
                        'native_byte_ranges': ranges, 'before_sha256': sha(src), 'after_sha256': sha(dest),
                        'nonposition_nonnormal_vertex_bytes_exact': True})
        print(f'{fn:28s} normals turned max {normal_turn.max():5.2f} deg  ', end='')
        print(f'moved {int(moved.sum()):5d}  max {changes[-1]["max_displacement"]:.5f}  '
              f'locked-field {locked_field:.2e}  area {area_ratio.min():.3f}-{area_ratio.max():.3f}  '
              + ' '.join(f'{k}={v.max():.4f}' for k, v in parts.items()), flush=True)
    (CAND / f'authoring/lod{lod}_geometry.json').write_text(json.dumps(newds, indent=2))
for src in (TEMPLATE / 'data/chara/HUF').glob('*.em*'):
    dest = CAND / 'data/chara/HUF' / src.name
    if not dest.exists():
        dest.write_bytes(src.read_bytes())
(CAND / 'localized_changes.json').write_text(json.dumps(changes, indent=2))
(CAND / 'fields_used.json').write_text(json.dumps(FIELDS, indent=2))
(CAND / 'field_reference_functions.json').write_text(json.dumps({
    'pants_halfwidth': {'y': ys_p.tolist(), 'x': hw_pants.tolist()},
    'bust_halfwidth': {'y': ys_b.tolist(), 'x': hw_bust.tolist()},
    'forearm_axis': {'absx': xs_a.tolist(), 'centre_y': arm_c[:, 0].tolist(), 'half_height': arm_c[:, 1].tolist()}}, indent=1))
print('CANDIDATE BUILT (not accepted, not installed):', CAND)
