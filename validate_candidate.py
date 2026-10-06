"""Fresh source, posed-surface and LOD regression on the two actual blend files.

Diagnostic rotations are explicit test inputs, not recorded DBXV2 animation/IK.
Neither input blend is saved by this audit.
"""
from pathlib import Path
import bpy, hashlib, json, sys
sys.dont_write_bytecode=True
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

HERE=Path(__file__).resolve().parent; WORK=HERE.parent; CAND=HERE/sys.argv[sys.argv.index('--')+1]
BASE=Path('D:\\Codebase\\Projects\\DRAGON BALL XENOVERSE 2\\_HUF_REVAMP511_ReferenceBody_20260930\\chunli_physique_20261004\\deliverable\\HUF_REVAMP511_ChunLiInspired_20261005\\HUF_REVAMP511_ChunLiInspired_20261005')
sys.path.insert(0,str(WORK/'scripts'))
from asset_io import skeleton
from skinning import global_matrices,rotation
S=Matrix(((1,0,0,0),(0,0,1,0),(0,1,0,0),(0,0,0,1)))
poses=json.loads((HERE/'poses_extended.json').read_text()); poses.pop('_note',None)
changes={(c['lod'],c['part']):c for c in json.loads((CAND/'localized_changes.json').read_text())}
report={'method':{'poses':poses,'intersection_scope':'Body triangles and nails; excludes auxiliary hair, shared rest vertices, coplanar and boundary-only contacts. BVH broad phase, strict segment/triangle test.',
    'runtime':False,'sources_saved':False,'blender':bpy.app.version_string},'sources':{},'pose_regression':[],'lod_surfaces':[]}
cache={}; source_pins={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sig(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def save(): (CAND/'fresh_blender_regression.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
def expose(layer):
    layer.exclude=False;layer.hide_viewport=False
    for child in layer.children:expose(child)
def crosses(a,b):
    e1=a[1]-a[0];e2=a[2]-a[0]
    for x,y in zip(b,np.roll(b,-1,axis=0)):
        direction=y-x;h=np.cross(direction,e2);det=float(np.dot(e1,h))
        if abs(det)<1e-13:continue
        delta=x-a[0];u=float(np.dot(delta,h)/det);q=np.cross(delta,e1)
        v=float(np.dot(direction,q)/det);t=float(np.dot(e2,q)/det)
        if 1e-6<t<1-1e-6 and 1e-6<u and 1e-6<v and u+v<1-1e-6:return True
    return False
def intersections(p,f,weld):
    tree=BVHTree.FromPolygons([Vector(v) for v in p],f.tolist(),all_triangles=True,epsilon=0.)
    hit=[];tested=0
    for a,b in tree.overlap(tree):
        if a>=b or set(weld[f[a]])&set(weld[f[b]]):continue
        tested+=1;ta=p[f[a]];tb=p[f[b]]
        if crosses(ta,tb) or crosses(tb,ta):hit.append((a,b))
    return sorted(hit),tested
def area(p,f):
    t=p[f];return .5*np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)

for label,package,blend in [('baseline',BASE,BASE/'editable/HUF_REVAMP511_ChunLiInspired.blend'),
                             ('refined',CAND,CAND/'editable/HUF_REVAMP511_ChunLiInspired.blend')]:
    source_pins[str(blend)]=sha(blend);bpy.ops.wm.open_mainfile(filepath=str(blend))
    rigs=[o for o in bpy.data.objects if o.type=='ARMATURE'];assert len(rigs)==1;rig=rigs[0]
    sk=skeleton(Path(r'D:\Codebase\Projects\DRAGON BALL XENOVERSE 2\HUF Assets\HUF_000_REVAMP_v5.1.1\HUF\HUF_000.esk'));rest=global_matrices(sk)
    re=max(float(np.max(abs(np.asarray(rig.data.bones[n].matrix_local)-np.asarray(S@Matrix(g)@S)))) for n,g in rest.items())
    assert set(rest)==set(rig.data.bones.keys()) and re<2e-5
    assert all((rig.data.bones[r['name']].parent.name if rig.data.bones[r['name']].parent else None)==
        (None if r['hierarchy'][0] in (65535,r['index']) else sk[r['hierarchy'][0]]['name']) for r in sk)
    row={'file':str(blend),'sha256':sha(blend),'rig_bones':len(rig.data.bones),'rest_matrix_error':re,'meshes':[],'poses':[]}
    report['sources'][label]=row; meshes={};scan={};seams=[]
    for lod in range(4):
        ds=json.loads((package/f'authoring/lod{lod}_geometry.json').read_text());pp=[];ff=[];offset=0
        for d in ds:
            name=f"HUF_{d['part']}_LOD{lod:02d}";o=bpy.data.objects[name];m=o.data;meshes[name]=o
            p=np.asarray([list(v.co) for v in m.vertices]);f=np.asarray([list(t.vertices) for t in m.polygons])
            uv=[[[float(m.uv_layers.active.data[i].uv[0]),1-float(m.uv_layers.active.data[i].uv[1])] for i in reversed(t.loop_indices)] for t in m.polygons]
            weights=[{o.vertex_groups[g.group].name:g.weight for g in v.groups if g.weight>0} for v in m.vertices]
            assert np.array_equal(p,np.array(d['positions'],dtype=np.float32)[:,[0,2,1]])
            assert f[:,::-1].tolist()==d['faces'] and [t.material_index for t in m.polygons]==d['material_indices']
            assert np.max(abs(np.asarray(uv)-d['uv']))<1e-7
            assert max(abs(a.get(n,0)-b.get(n,0)) for a,b in zip(weights,d['weights']) for n in set(a)|set(b))<1e-7
            assert len(o.modifiers)==1 and o.modifiers[0].type=='ARMATURE' and o.modifiers[0].object==rig
            assert not o.modifiers[0].use_deform_preserve_volume and m.shape_keys is None
            assert np.max(abs(np.asarray(o.matrix_world)-np.eye(4)))<1e-7
            protected={'faces':f.tolist(),'uv_layers':[[list(x.uv) for x in l.data] for l in m.uv_layers],
                'weights':weights,'groups':list(o.vertex_groups.keys()),'material_indices':[t.material_index for t in m.polygons],
                'smoothing':[t.use_smooth for t in m.polygons],'materials':list(m.materials.keys()),
                'edges':[list(e.vertices) for e in m.edges]}
            raw=np.asarray([list(x.value) for x in m.attributes['custom_normal'].data]); normals=np.asarray([list(n.vector) for n in m.corner_normals])
            reference=np.asarray(d['normals'])[:,[0,2,1]][np.asarray([l.vertex_index for l in m.loops])]
            normal_error=float(np.max(np.linalg.norm(normals-reference,axis=1)))
            assert normal_error<.003,(name,normal_error)
            if label=='baseline':cache[name]={'protected':sig(protected),'raw':raw,'p':p,'area':area(p,f),'vectors':normals}
            else:
                assert sig(protected)==cache[name]['protected'],name
                allowed=(set(changes[(lod,d['part'])]['reencoded_blender_normal_vertex_ids'])|set(changes[(lod,d['part'])]['normal_updated_vertex_ids'])) if (lod,d['part']) in changes else set()
                outside=[l.index for l in m.loops if l.vertex_index not in allowed]
                assert np.array_equal(raw[outside],cache[name]['raw'][outside]),name
                # Reference-fit policy: moved surfaces carry rotated normals (checked against the candidate JSON above);
                # loops outside the moved/normal-updated set must keep the baseline vectors exactly.
                keep=[l.index for l in m.loops if l.vertex_index not in allowed]
                assert np.max(np.linalg.norm(normals[keep]-cache[name]['vectors'][keep],axis=1))<.0005,name
                actual=set(np.where(np.linalg.norm(p-cache[name]['p'],axis=1)>1e-10)[0])
                expected=set(changes[(lod,d['part'])]['changed_source_vertex_ids']) if (lod,d['part']) in changes else set()
                assert actual==expected,(name,actual^expected)
            aa=area(p,f);rat=aa/cache[name]['area']
            row['meshes'].append({'name':name,'vertices':len(p),'triangles':len(f),'normal_export_error_max':normal_error,
                'protected_signature':sig(protected),'area_ratio_min':float(rat.min()),'area_ratio_max':float(rat.max()),
                'zero_area_faces':int((aa<1e-12).sum())})
            assert np.min(aa)>1e-12
            kept=[j for j in range(len(f)) if d['materials'][d['material_indices'][j]]['name']!='HAIR_pubic']
            pp.extend(p.tolist());ff.extend((f[kept]+offset).tolist());offset+=len(p)
        pp=np.asarray(pp);ff=np.asarray(ff);_,weld=np.unique(np.round(pp,6),axis=0,return_inverse=True)
        scan[lod]=(ff,weld)
        for a,b,kind in [('Bust','Pants','waist'),('Bust','Rist','wrist'),('Pants','Boots','ankle')]:
            a=meshes[f'HUF_{a}_LOD{lod:02d}'];b=meshes[f'HUF_{b}_LOD{lod:02d}'];tree=KDTree(len(b.data.vertices))
            for v in b.data.vertices:tree.insert(v.co,v.index)
            tree.balance();pairs=[]
            for i in json.loads(a['locked_modular_vertices']):
                p=a.data.vertices[i].co
                if kind=='waist' and not .05<p.z<.12:continue
                if kind=='wrist' and not abs(p.x)>.4:continue
                if kind=='ankle' and not -.55<p.z<-.5:continue
                _,j,dist=tree.find(p);assert dist<1e-7;pairs.append((i,j))
            assert pairs;seams.append((a.name,b.name,kind,lod,pairs))
    row['images']=[{'name':i.name,'packed_sha256':hashlib.sha256(i.packed_file.data).hexdigest() if i.packed_file else None} for i in bpy.data.images if i.name not in ('Render Result','Viewer Node')]
    row['materials']=[{'name':m.name,'properties':dict(m.items()),'nodes':[(n.name,n.bl_idname,getattr(getattr(n,'image',None),'name',None)) for n in m.node_tree.nodes],
        'links':[(l.from_node.name,l.from_socket.name,l.to_node.name,l.to_socket.name) for l in m.node_tree.links]} for m in bpy.data.materials]
    if label=='refined':
        assert row['images']==report['sources']['baseline']['images']
        assert row['materials']==report['sources']['baseline']['materials']
    for c in bpy.data.collections:c.hide_viewport=False
    for o in meshes.values():o.hide_set(False);o.hide_viewport=False
    expose(bpy.context.view_layer.layer_collection)
    for pose_name,rotations in poses.items():
        for pb in rig.pose.bones:pb.matrix_basis=Matrix.Identity(4)
        for name,rr in rotations.items():
            mat=np.eye(4)
            for axis,degrees in rr:
                r=np.eye(4);r[:3,:3]=rotation(axis,degrees);mat=mat@r
            rig.pose.bones[name].matrix_basis=S@Matrix(mat)@S
        bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get();coords={};stats=[]
        for name,o in meshes.items():
            ev=o.evaluated_get(deps);m=ev.to_mesh()
            try:
                p=np.asarray([list(v.co) for v in m.vertices]);f=np.asarray([list(t.vertices) for t in m.polygons]);aa=area(p,f)
                assert len(p)==len(o.data.vertices) and np.isfinite(p).all() and (aa>1e-12).all()
                coords[name]=p;delta=0.;unrelated=0.
                if label=='baseline':cache[(pose_name,name)]=p
                else:
                    dist=np.linalg.norm(p-cache[(pose_name,name)],axis=1);delta=float(dist.max())
                    mask=np.linalg.norm(cache[name]['p']-np.asarray([list(v.co) for v in o.data.vertices]),axis=1)<1e-10
                    unrelated=float(dist[mask].max());assert unrelated<1e-7
                stats.append({'name':name,'zero_area_faces':int((aa<1e-12).sum()),'min_area':float(aa.min()),'max_refinement_displacement':delta,'unchanged_vertices_max_displacement':unrelated})
            finally:ev.to_mesh_clear()
        maxgap=max(float(np.linalg.norm(coords[a][i]-coords[b][j])) for a,b,k,l,pairs in seams for i,j in pairs)
        assert maxgap<1e-6
        scans=[]
        for lod in range(4):
            p=np.concatenate([coords[f'HUF_{part}_LOD{lod:02d}'] for part in ('Bust','Pants','Rist','Boots')]);f,weld=scan[lod]
            hit,tested=intersections(p,f,weld);new=[]
            if label=='baseline':cache[(pose_name,lod,'crossings')]=set(hit)
            else:new=sorted(set(hit)-cache[(pose_name,lod,'crossings')])
            scans.append({'lod':lod,'tested_nonadjacent_pairs':tested,'strict_crossings':hit,'new_crossings':new})
        prow={'pose':pose_name,'seam_gap_max':maxgap,'meshes':stats,'intersections':scans};row['poses'].append(prow)
        print(label,pose_name,'seam',maxgap,'crossings',[len(s['strict_crossings']) for s in scans],'new',[len(s['new_crossings']) for s in scans],flush=True);save()
    for part in ('Bust','Pants'):
        for lod in (1,2,3):
            a=meshes[f'HUF_{part}_LOD00'].data;b=meshes[f'HUF_{part}_LOD{lod:02d}'].data
            pa=np.asarray([list(v.co) for v in a.vertices]);fa=np.asarray([list(t.vertices) for t in a.polygons])
            pb=np.asarray([list(v.co) for v in b.vertices]);fb=np.asarray([list(t.vertices) for t in b.polygons])
            ta=BVHTree.FromPolygons([Vector(v) for v in pa],fa.tolist(),all_triangles=True)
            tb=BVHTree.FromPolygons([Vector(v) for v in pb],fb.tolist(),all_triangles=True)
            da=np.array([tb.find_nearest(Vector(v))[3] for v in np.concatenate([pa,pa[fa].mean(1)])])
            db=np.array([ta.find_nearest(Vector(v))[3] for v in np.concatenate([pb,pb[fb].mean(1)])])
            report['lod_surfaces'].append({'source':label,'part':part,'lod':lod,'high_to_low_max':float(da.max()),'high_to_low_p95':float(np.percentile(da,95)),
                'low_to_high_max':float(db.max()),'low_to_high_max_sample':np.concatenate([pb,pb[fb].mean(1)])[int(db.argmax())].tolist(),'sample_index':int(db.argmax()),'low_vertices':len(pb),'policy':'Every vertex and triangle centroid; model units, neutral actual Blender meshes.'})
    save()
assert all(sha(Path(p))==h for p,h in source_pins.items())
report['new_strict_crossings_total']=sum(len(s['new_crossings']) for p in report['sources']['refined']['poses'] for s in p['intersections'])
report['neutral_strict_crossings']={label:sum(len(s['strict_crossings']) for s in report['sources'][label]['poses'][0]['intersections']) for label in report['sources']}
report['source_files_unchanged']=True;save()
assert report['new_strict_crossings_total']==0,'New crossing needs review; candidate is not accepted.'
assert report['neutral_strict_crossings']['refined']==0
print('FRESH BLENDER REGRESSION PASS: protected source data, 13 diagnostic poses x 4 LODs, neutral surface, no new strict crossings.')
