"""Print a compact table from final_metrics.py output.  python print_final_metrics.py <final_metrics.json>"""
import json, sys

r = json.load(open(sys.argv[1]))
for band, views in r['bands'].items():
    for view, x in views.items():
        print(f"{band:22s} {view:6s} mean |err| {x['mean_abs_error_before']:.4f} -> {x['mean_abs_error_after']:.4f}   "
              f"max {x['max_abs_error_before']:.4f} -> {x['max_abs_error_after']:.4f}")
print('width scale k', round(r['width_scale_fit_before']['k'], 4), '->', round(r['width_scale_fit_after']['k'], 4))
print('waist minimum (h, half-width):', r['waist_minimum']['front'])
print('bust apex profile:', r['bust_apex_profile'])
