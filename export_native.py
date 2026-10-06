"""Rebuild the 16 sealed EMDs from source export JSON and the original binary templates.

Reference-fit revision: vertex positions AND vertex normals (half-float, bytes 12-17 of the 36-byte 0x8207 record)
come from the JSON; every other byte (UVs, bone indices/weights, palettes, indices, materials, headers) is taken
from the original template. Bounding volumes only ever expand.
"""
from pathlib import Path
import argparse, hashlib, json, struct, sys, tempfile
sys.dont_write_bytecode = True
from emd_writer import write_emd
from asset_io import emd
ROOT = Path(__file__).resolve().parent.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rebuild():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'authoring/reexported'); args = ap.parse_args()
    contract = json.loads((ROOT / 'authoring/source_contract.json').read_text())
    for rel, pin in contract['required_inputs'].items():
        if sha(ROOT / rel) != pin:
            raise ValueError('Source changed; re-audit required: ' + rel)
    result = {}; rows = []
    for lod in range(4):
        for d in json.loads((ROOT / f'authoring/lod{lod}_geometry.json').read_text()):
            name = f"HUF_2998_{d['part']}" + (f'_LOD{lod:02d}' if lod else '') + '.emd'
            source = ROOT / 'authoring/native_templates' / name; old = source.read_bytes(); out = bytearray(old)
            with tempfile.TemporaryDirectory() as temp:
                trial = Path(temp) / name; new = write_emd(d); trial.write_bytes(new); aa, bb = emd(trial), emd(source)
            if len(aa) != len(bb):
                raise ValueError('Submesh count changed')
            bounds = {}
            for a, b in zip(aa, bb):
                for key in a:
                    if key not in ('positions', 'normals', 'vo', 'mesh_offset', 'sub_offset', 'groups') and a[key] != b[key]:
                        raise ValueError('Protected data differs: ' + name + ' ' + key)
                if [(g['indices'], g['bones']) for g in a['groups']] != [(g['indices'], g['bones']) for g in b['groups']]:
                    raise ValueError('Triangle palettes or indexing changed')
                for i in range(a['count']):
                    src = a['vo'] + i * a['stride']; dst = b['vo'] + i * b['stride']
                    out[dst:dst + 18] = new[src:src + 18]
                bounds[b['sub_offset']] = list(a['positions'])
                bounds.setdefault(b['mesh_offset'], []).extend(a['positions'])
            for offset, points in bounds.items():
                box = list(struct.unpack_from('<12f', old, offset))
                lo = [min(box[4 + j], min(p[j] for p in points)) for j in range(3)]
                hi = [max(box[8 + j], max(p[j] for p in points)) for j in range(3)]
                if lo == box[4:7] and hi == box[8:11]:
                    continue
                for j, xyz in enumerate(([.5 * (a + b) for a, b in zip(lo, hi)], lo, hi)):
                    struct.pack_into('<3f', out, offset + j * 16, *xyz)
            pin = hashlib.sha256(out).hexdigest()
            if pin != contract['output_sha256'][name]:
                raise ValueError('Native reconstruction mismatch: ' + name)
            result[name] = out; rows.append({'file': name, 'sha256': pin, 'exact': True})
    destination = args.output.resolve(); destination.mkdir(parents=True, exist_ok=True)
    for name, b in result.items():
        p = destination / name
        if p.exists() and p.read_bytes() != b:
            raise ValueError('Preserving an unrelated export at ' + str(p))
        p.write_bytes(b)
    print(json.dumps({'reconstructed_native_files': len(rows), 'all_exact': True, 'files': rows}, indent=2))


if __name__ == '__main__':
    rebuild()
