"""Compare quality_metrics outputs between packages.

python summarize_quality.py <metrics.json> [<metrics2.json> ...] [--lod LOD00] [--regions a,b]
"""
import json, sys

files = [a for a in sys.argv[1:] if a.endswith('.json')]
lod = sys.argv[sys.argv.index('--lod') + 1] if '--lod' in sys.argv else 'LOD00'
pk = {}
for f in files:
    pk.update(json.load(open(f))['packages'])
tags = list(pk)
poses = list(next(iter(pk.values()))['lods'])
regions = (sys.argv[sys.argv.index('--regions') + 1].split(',') if '--regions' in sys.argv else
           ['waist_flank', 'waist', 'iliac_crest', 'lateral_hips', 'glute_apex', 'lateral_thigh', 'inner_thighs', 'chest', 'shoulders', 'knees'])
for pose in poses:
    print(f'\n=== {pose} ({lod})')
    print('seam crease max/mean deg: ' + ' | '.join(
        f"{t}: " + ', '.join(f"{k} {v['crease_deg_max']:.1f}/{v['crease_deg_mean']:.1f}" for k, v in pk[t]['lods'][pose][lod]['seams'].items()) for t in tags))
    print('region: ' + ' | '.join(f'{t}: p95 / concave-max / n(concave>20)' for t in tags))
    for r in regions:
        cells = []
        for t in tags:
            x = pk[t]['lods'][pose][lod]['regions'].get(r)
            cells.append(f"{x['dihedral_p95']:5.1f} / {x['concave_max']:5.1f} / {x['concave_over_20deg']:3d}" if x else '   n/a   ')
        print(f'  {r:14s} ' + ' | '.join(cells))
    print('pubic: ' + ' | '.join(f"{t}: min {pk[t]['lods'][pose][lod]['pubic_hair']['signed_offset_min']:+.4f} med {pk[t]['lods'][pose][lod]['pubic_hair']['median']:+.4f} below {pk[t]['lods'][pose][lod]['pubic_hair']['below_skin_count']} overlaps {pk[t]['lods'][pose][lod]['pubic_hair']['hair_skin_triangle_overlaps']}" for t in tags))
    if pose == 'rest':
        print('LOD region deviation p95 (LOD01/02/03): ')
        for r in ['shoulders', 'chest', 'waist', 'iliac_crest', 'lateral_hips', 'glute_apex', 'inner_thighs', 'knees', 'calves', 'wrists', 'ankles']:
            cells = []
            for t in tags:
                d = pk[t]['lods'][pose]['lod_region_deviation'].get(r, {})
                cells.append('/'.join(f"{d[k]['p95']:.4f}" if k in d else '  -   ' for k in ('LOD01', 'LOD02', 'LOD03')))
            print(f'  {r:13s} ' + ' | '.join(cells))
