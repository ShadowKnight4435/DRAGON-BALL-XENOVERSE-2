"""Overlay LOD00 Pants vertex ids on a render_regions.py image (same orthographic camera). Read-only.

python annotate_view.py <image.png> <pkg_dir> <view> <out.png> [--xmin 0] [--ids 1,2,3]
"""
import json, math, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
IMG, PKG, VIEW, OUT = sys.argv[1:5]
argv = sys.argv[5:]
src = Path(__file__).resolve().parent.joinpath('render_regions.py').read_text()
VIEWS = eval(src.split('VIEWS = ')[1].split('\n}')[0] + '\n}')
(gx, gy, gz), az, el, scale = VIEWS[VIEW]
t = np.array([gx, gz, gy]); a = math.radians(az); e = math.radians(el)
loc = t + 3 * np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
fwd = (t - loc) / np.linalg.norm(t - loc); right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
im = cv2.imread(IMG); R = im.shape[0]
d = [x for x in json.loads((Path(PKG) / 'authoring/lod0_geometry.json').read_text()) if x['part'] == 'Pants'][0]
p = np.asarray(d['positions']); B = p[:, [0, 2, 1]]
xmin = float(argv[argv.index('--xmin') + 1]) if '--xmin' in argv else 0.0
only = set(map(int, argv[argv.index('--ids') + 1].split(','))) if '--ids' in argv else None
used = np.unique(np.asarray(d['faces']))
seen = set()
for v in used:
    if p[v, 0] < xmin or (only and v not in only):
        continue
    k = tuple(np.round(p[v], 5))
    if k in seen:
        continue
    seen.add(k)
    q = B[v] - t
    if np.dot(q, -fwd) < -0.02:      # behind the surface region facing the camera (rough cull)
        continue
    u = int((0.5 + np.dot(q, right) / scale) * R); w = int((0.5 - np.dot(q, up) / scale) * R)
    if 0 <= u < R and 0 <= w < R:
        cv2.circle(im, (u, w), 2, (0, 255, 255), -1)
        cv2.putText(im, str(v), (u + 3, w - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 255), 1)
cv2.imwrite(OUT, im); print('wrote', OUT)
