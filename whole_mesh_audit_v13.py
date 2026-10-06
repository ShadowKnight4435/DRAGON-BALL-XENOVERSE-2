"""Fresh neutral whole-mesh QA. Reads production files; writes only audit evidence."""
from pathlib import Path
from collections import Counter,defaultdict
import bpy,json,hashlib,sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.dont_write_bytecode=True
W=Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\_HUF_REVAMP511_ReferenceBody_20260930')
O=W/'production_reaudit_20261005'
P=W/'nude_base_audit_20261006'/sys.argv[sys.argv.index('--')+1]
B=W/'reference_fit_20261005'/'candidate_v13'   # v13 production baseline (structural authority)
sys.path.insert(0,str(W/'scripts'))
from asset_io import emd,skeleton,material_names
from skinning import influences,verify_bind
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
BLEND=P/'editable/HUF_REVAMP511_NudeBaseRefine.blend'
def weld(p):
    keys=np.round(p,6);unique,idx,inv=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    return inv,p[idx]
def topology(p,f):
    wi,wp=weld(p);wf=wi[f];e=Counter(tuple(sorted((int(a),int(b)))) for t in wf for a,b in zip(t,np.roll(t,-1)))
    directed=Counter((int(a),int(b)) for t in wf for a,b in zip(t,np.roll(t,-1)))
    parent=list(range(len(wp)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in e:
        a=root(a);b=root(b)
        if a!=b:parent[b]=a
    groups=defaultdict(list)
    for idx,t in enumerate(wf):groups[root(int(t[0]))].append(idx)
    comps=[]
    for faces in groups.values():
        verts=np.unique(f[np.array(faces)]);pts=p[verts]
        comps.append({'faces':len(faces),'vertices':len(verts),'bounds':[pts.min(0).tolist(),pts.max(0).tolist()]})
    bd=np.array(sorted({v for (a,b),c in e.items() if c==1 for v in (a,b)}),dtype=int)
    area=.5*np.linalg.norm(np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]]),axis=1)
    lengths=np.stack([np.linalg.norm(p[f[:,a]]-p[f[:,b]],axis=1) for a,b in ((0,1),(1,2),(2,0))])
    quality=4*np.sqrt(3)*area/(lengths**2).sum(0)
    return {'vertices':len(p),'welded_vertices':len(wp),'triangles':len(f),'unused_vertices':len(p)-len(np.unique(f)),
      'zero_area_faces':int((area<1e-12).sum()),'nonmanifold_edges':sum(v>2 for v in e.values()),
      'duplicate_faces':len(wf)-len({tuple(sorted(t)) for t in wf}),
      'inconsistent_winding_edges':sum(c==2 and (directed[(a,b)]!=1 or directed[(b,a)]!=1) for (a,b),c in e.items()),
      'boundary_vertices':len(bd),'boundary_bounds':[wp[bd].min(0).tolist(),wp[bd].max(0).tolist()] if len(bd) else None,
      'components':sorted(comps,key=lambda c:-c['faces']),
      'triangle_quality_min':float(quality.min()),'triangle_quality_p01':float(np.percentile(quality,1)),
      'bounds':[p.min(0).tolist(),p.max(0).tolist()]}
def mirrored_surface(p,f):
    tree=BVHTree.FromPolygons([Vector(x) for x in p],f.tolist(),all_triangles=True)
    samples=np.concatenate([p,p[f].mean(1)]);samples[:,0]*=-1
    ds=np.array([tree.find_nearest(Vector(x))[3] for x in samples])
    i=int(ds.argmax())
    return {'max':float(ds.max()),'p95':float(np.percentile(ds,95)),'max_sample':samples[i].tolist(),'scope':'Every vertex and triangle centroid mirrored to the actual triangle surface; not nearest-vertex correspondence.'}
def crosssection(p,f,y):
    t=p[f];out=[]
    for a,b in ((0,1),(1,2),(2,0)):
        keep=(t[:,a,1]-y)*(t[:,b,1]-y)<0
        aa=t[keep,a];bb=t[keep,b];tt=(y-aa[:,1])/(bb[:,1]-aa[:,1]);out.extend(aa+(bb-aa)*tt[:,None])
    pts=np.array(out);pts=pts[abs(pts[:,0])<.215]
    return {'y':y,'width':float(np.ptp(pts[:,0])),'depth':float(np.ptp(pts[:,2])),'front':float(pts[:,2].min()),'rear':float(pts[:,2].max())} if len(pts) else None
report={'source':str(BLEND),'source_sha256':sha(BLEND),'units':'native model units, Y vertical; no real-world or reference scale inferred','runtime':False,'levels':[],'native':[]}
sk=skeleton(Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\HUF Assets\HUF_000_REVAMP_v5.1.1\HUF\HUF_000.esk'))
bn={r['name'] for r in sk};report['rig']={'bones':len(sk),'inverse_bind_max_error':verify_bind(sk)}
for lod in range(4):
    source=json.loads((P/f'authoring/lod{lod}_geometry.json').read_text())
    baseline=json.loads((B/f'authoring/lod{lod}_geometry.json').read_text())
    pp=[];ff=[];offset=0;permat=[];patches=[];source_changes=[]
    for d,bd in zip(source,baseline):
        p=np.array(d['positions']);f=np.array(d['faces']);mi=np.array(d['material_indices'])
        normals=np.array(d['normals']);face_norm=np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])
        avg=normals[f].mean(1);dot=(avg*face_norm).sum(1)
        skin=np.array([d['materials'][i]['name'].startswith('SKIN') for i in mi])
        kept=np.array([d['materials'][i]['name']!='HAIR_pubic' for i in mi])
        for idx,mat in enumerate(d['materials']):
            fs=f[mi==idx];ids=np.unique(fs);uv=np.array(d['uv'])[mi==idx]
            a=uv[:,1]-uv[:,0];b=uv[:,2]-uv[:,0];ua=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])*.5
            stat=topology(p[ids],np.searchsorted(ids,fs));stat.update(part=d['part'],material=mat['name'],zero_area_UV_triangles=int((ua<1e-12).sum()),normals_against_face=int((dot[mi==idx]<0).sum()))
            permat.append(stat)
            if mat['name']=='SKIN_nipple':
                for side,label in [(-1,'left'),(1,'right')]:
                    vs=p[ids][p[ids,0]*side>0]
                    center=(vs.max(0)+vs.min(0))/2
                    patches.append({'side':label,'unique_positions':len(np.unique(np.round(vs,7),axis=0)),'bounds':[vs.min(0).tolist(),vs.max(0).tolist()],'bbox_center':center.tolist(),'axis_span':np.ptp(vs,axis=0).tolist(),'caution':'Geometry assigned SKIN_nipple; not a calibrated visual areola diameter.'})
        wsum=np.array([sum(w.values()) for w in d['weights']])
        assert np.isfinite(p).all() and np.isfinite(normals).all() and np.max(abs(wsum-1))<1e-6
        assert max(map(len,d['weights']))<=4 and all(n in bn for row in d['weights'] for n in row)
        changed=np.linalg.norm(p-np.array(bd['positions']),axis=1)>1e-10
        chest_material=np.array([m['name'] in ('SKIN_nipple','SKIN_bust_C','SKIN_bust_CB') for m in d['materials']])
        chest_ids=np.unique(f[chest_material[mi]]) if chest_material.any() else np.array([],int)
        source_changes.append({'part':d['part'],'moved_vs_confirmed':int(changed.sum()),'vertical_coordinates_exact':bool(np.array_equal(p[:,1],np.array(bd['positions'])[:,1])),'chest_material_vertices_moved':int(changed[chest_ids].sum()),'weight_sum_error':float(np.max(abs(wsum-1)))})
        pp.extend(p.tolist());ff.extend((f[kept]+offset).tolist());offset+=len(p)
        fn=f"HUF_2998_{d['part']}"+(f'_LOD{lod:02d}' if lod else '')+'.emd'
        subs=emd(P/'data/chara/HUF'/fn);mnames=material_names((P/'data/chara/HUF'/f"HUF_2998_{d['part']}.emm").read_bytes())
        native={'file':fn,'submeshes':len(subs),'EMD_materials_bound':all(s['name'] in mnames for s in subs),'all_bones_resolve':all(n in bn for s in subs for g in s['groups'] for n in g['bones']),'finite_attributes':all(np.isfinite(np.asarray(s[k])).all() for s in subs for k in ('positions','normals','uvs','weights'))}
        assert native['EMD_materials_bound'] and native['all_bones_resolve'] and native['finite_attributes']
        report['native'].append(native)
    p=np.array(pp);f=np.array(ff);used=np.unique(f);p=p[used];f=np.searchsorted(used,f);t=topology(p,f)
    level={'lod':lod,'assembled_body_including_nails_excluding_aux_hair':t,'mirror_surface':mirrored_surface(p,f),'material_regions':permat,'areola_material_regions':patches,'v13_comparison':source_changes,'sections':[crosssection(p,f,y) for y in (.30,.25,.20,.14,.08,.02,-.02,-.12,-.20,-.295,-.40,-.52)]}
    assert all(t[k]==0 for k in ('zero_area_faces','nonmanifold_edges','duplicate_faces','inconsistent_winding_edges'))
    assert t['boundary_vertices']==26
    report['levels'].append(level)
    print('NEUTRAL',lod,'components',len(t['components']),'boundary',t['boundary_vertices'],'mirror max',level['mirror_surface']['max'],flush=True)
report['source_files_unmodified']=sha(BLEND)==report['source_sha256']
(P/'whole_mesh_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('WHOLE MESH TECHNICAL AUDIT WRITTEN',flush=True)

