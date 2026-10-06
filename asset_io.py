"""Strict, read-only readers for the supplied little-endian XV2 assets.

Format reference: Olganix/LibXenoverse2 EMD.h, ESK.h and EAN.h.
No importer/exporter round trip is used to write runtime assets.
"""
import struct
import math
from pathlib import Path

def u16(b,o): return struct.unpack_from('<H',b,o)[0]
def u32(b,o): return struct.unpack_from('<I',b,o)[0]
def string(b,o):
    if not 0 <= o < len(b): raise ValueError(('string outside file',o,len(b)))
    return b[o:b.index(0,o)].decode('ascii')
def signature(b,s):
    if b[:4]!=s or b[4:6]!=b'\xfe\xff': raise ValueError(('signature/endian',b[:6],s))

def emd(b):
    if isinstance(b,(Path,str)): b=Path(b).read_bytes()
    signature(b,b'#EMD')
    result=[]
    for mi in range(u16(b,18)):
        model=u32(b,u32(b,20)+4*mi)
        model_name=string(b,u32(b,u32(b,24)+4*mi))
        for mei in range(u16(b,model+2)):
            mesh=model+u32(b,model+u32(b,model+4)+4*mei)
            for si in range(u16(b,mesh+54)):
                sub=mesh+u32(b,mesh+u32(b,mesh+56)+4*si)
                flags,stride,count=struct.unpack_from('<III',b,sub+48)
                vo=sub+u32(b,sub+60)
                if count and (flags,stride)!=(0x8207,36): raise ValueError(('unsupported layout',hex(flags),stride))
                if vo+stride*count>len(b): raise ValueError('vertex buffer outside file')
                groups=[]
                for ti in range(u16(b,sub+70)):
                    tri=sub+u32(b,sub+u32(b,sub+76)+4*ti)
                    n,nb,io,bo=struct.unpack_from('<IIII',b,tri)
                    # LibXenoverse EMDTriangles::read selects index width by count.
                    code='I' if n>65535 else 'H'
                    indices=struct.unpack_from('<'+str(n)+code,b,tri+(io or 16))
                    if n%3 or (indices and max(indices)>=count): raise ValueError('invalid indices')
                    bones=[string(b,tri+u32(b,tri+bo+4*k)) for k in range(nb)]
                    groups.append({'indices':indices,'bones':bones,'offset':tri})
                textures=[]
                for ti in range(b[sub+69]):
                    to=sub+u32(b,sub+72)+12*ti
                    textures.append(struct.unpack_from('<4B2f',b,to))
                result.append({'model_index':mi,'model_name':model_name,'mesh_offset':mesh,'sub_offset':sub,
                    'name':string(b,sub+u32(b,sub+64)),'flags':flags,'stride':stride,'vo':vo,'count':count,
                    'positions':[struct.unpack_from('<3f',b,vo+stride*i) for i in range(count)],
                    'normals':[struct.unpack_from('<3e',b,vo+stride*i+12) for i in range(count)],
                    'uvs':[struct.unpack_from('<2e',b,vo+stride*i+20) for i in range(count)],
                    'bone_indices':[struct.unpack_from('<4B',b,vo+stride*i+24) for i in range(count)],
                    'weights':[(*struct.unpack_from('<3e',b,vo+stride*i+28),1-sum(struct.unpack_from('<3e',b,vo+stride*i+28))) for i in range(count)],
                    'groups':groups,'textures':textures})
    return result

def skeleton(b,offset=None):
    if isinstance(b,(Path,str)): b=Path(b).read_bytes()
    if offset is None:
        signature(b,b'#ESK');offset=u32(b,16)
    n=u16(b,offset)
    h,no,rel,absolute=struct.unpack_from('<4I',b,offset+4)
    records=[]
    for i in range(n):
        records.append({'index':i,'name':string(b,offset+u32(b,offset+no+4*i)),
            'hierarchy':struct.unpack_from('<4H',b,offset+h+8*i),
            'trs':struct.unpack_from('<12f',b,offset+rel+48*i),
            'inverse_bind':struct.unpack_from('<16f',b,offset+absolute+64*i) if absolute else None})
    if any(any(j!=65535 and j>=n for j in r['hierarchy'][:3]) for r in records): raise ValueError('invalid hierarchy')
    if any(not all(math.isfinite(v) for v in r['trs']) for r in records): raise ValueError('invalid transform')
    return records

def animation_catalog(b):
    if isinstance(b,(Path,str)): b=Path(b).read_bytes()
    signature(b,b'#EAN')
    result=[]
    for i in range(u16(b,18)):
        offset=u32(b,u32(b,24)+i*4)
        name=string(b,u32(b,u32(b,28)+i*4))
        result.append({'index':i,'name':name,'offset':offset,'frames':u32(b,offset+4),'nodes':u32(b,offset+8),'index_size':b[offset+2],'float_size':b[offset+3]})
    return result

def material_names(b):
    signature(b,b'#EMM')
    section=u32(b,12)
    return [string(b,section+u32(b,section+4+4*i)) for i in range(u32(b,section))]

def emb_entries(b):
    signature(b,b'#EMB')
    n=u32(b,12);table=u32(b,24)
    result=[]
    for i in range(n):
        at=table+8*i
        offset,size=struct.unpack_from('<II',b,at)
        data=b[at+offset:at+offset+size]
        if len(data)!=size: raise ValueError('EMB entry outside buffer')
        result.append({'bytes':size,'signature':data[:4].decode('ascii',errors='replace'),
            'height':u32(data,12) if data[:4]==b'DDS ' else None,'width':u32(data,16) if data[:4]==b'DDS ' else None})
    return result
