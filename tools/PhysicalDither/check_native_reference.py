"""Independent numerical checks for a native before/after conversion pair.
Usage: python check_native_reference.py --before report.json --after report.json --image layout.png
Requires the reference tool's NumPy/SciPy/Pillow development dependencies.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, convolve1d
from physical_dither import responses

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--before',type=Path,required=True)
p.add_argument('--after',type=Path,required=True)
p.add_argument('--image',type=Path,required=True)
a=p.parse_args()
before=json.loads(a.before.read_text());after=json.loads(a.after.read_text());c=after['settings']
assert before['algorithm']==after['algorithm']
assert before['sourceHash']==after['sourceHash']==hashlib.sha256(a.image.read_bytes()).hexdigest().upper()
assert before['result']['size']==after['result']['size']
assert before['settings']['passes']==0
for key,value in c.items():
    if key!='passes':assert before['settings'][key]==value,(key,'configuration mismatch')
profile={'canvas_radius_mm':after['canvasRadiusMm'],'track_pitch_nm':after['trackPitchNm'],
         'textures':[],'views':[],'wavelengths':[[w['nm'],w['weight']] for w in c['wavelengths']]}
for t in c['textures']:
    profile['textures'].append({'diffuse':t['diffuse'],'specular':t['specular'],'diffraction':t['diffraction'],
        'specular_width':t['specularWidth'],'diffraction_width_radial':t['radialWidth'],'diffraction_width_tangent':t['tangentWidth']})
for v in c['views']:
    profile['views'].append({'camera_mm':[v['cameraX'],v['cameraY'],v['cameraZ']], 'lights':[
        {'center_mm':[l['x'],l['y'],l['z']], 'size_mm':[l['width'],l['height']],
         'samples_per_axis':l['samples'],'radiance':l['radiance']} for l in v['lights']]})
m=c['mapSize'];n=after['result']['size'];maps=responses(profile,m)
expected=np.array([[q[m//2,m//2],q[m//2,3*m//4]] for q in maps.reshape(-1,m,m)])
np.testing.assert_allclose(expected,after['opticalCheckpoints'],rtol=5e-6,atol=1e-12)
source=np.asarray(Image.open(a.image).convert('RGBA'),dtype=float)/255
rgb=source[:,:,:3];linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
source_gray=(linear@np.array([.2126,.7152,.0722]))*source[:,:,3]+1-source[:,:,3]
# Upsampled fixture: same bilinear convention and source float rounding as native.
assert source_gray.shape[0]<=n,'Use an upsampled square fixture for this reference check.'
y,x=np.meshgrid(np.linspace(0,source_gray.shape[0]-1,n),np.linspace(0,source_gray.shape[1]-1,n),indexing='ij')
gray=map_coordinates(source_gray.astype(np.float32),[y,x],order=1,mode='nearest').astype(float)**c['toneExponent']
y,x=np.meshgrid(np.linspace(0,m-1,n),np.linspace(0,m-1,n),indexing='ij')
fields=np.asarray([map_coordinates(q,[y,x],order=1,mode='nearest') for q in maps.reshape(-1,m,m)]).reshape(len(c['views']),2,n,n)
def load_codes(report):
    return np.asarray(Image.open(report['result']['binaryPath']),dtype=float)/255
b0=load_codes(before);b1=load_codes(after)
mask=np.asarray(Image.open(after['result']['maskedPath']).convert('RGBA'))[:,:,3]>0
cell=after['result']['cellUm']/1000
loss=[]
for codes in [b0,b1]:
    value=0.
    for v in range(len(c['views'])):
        r0,r1=fields[v];target=after['referenceLow'][v]+(after['referenceHigh'][v]-after['referenceLow'][v])*gray
        error=np.where(mask,r0+(r1-r0)*codes-target,0)
        for scale,weight in [(.02,1),(.06,.5),(.18,.25)]:
            sigma=scale/cell;radius=max(1,int(np.ceil(3*sigma)))
            q=np.arange(-radius,radius+1);k=np.exp(-.5*q*q/(sigma*sigma));k/=sum(k)
            filtered=convolve1d(convolve1d(error,k,axis=0,mode='constant'),k,axis=1,mode='constant')
            value+=c['views'][v]['weight']*weight*np.sum(filtered[mask]**2)
    loss.append(float(value))
assert loss[1]<=loss[0]+max(1e-15,loss[0]*1e-7),loss
print(json.dumps({'optical_checkpoint_max_relative_error':float(np.max(abs(expected-np.asarray(after['opticalCheckpoints']))/np.maximum(abs(expected),1e-20))),
    'multiscale_loss_before':loss[0],'multiscale_loss_after':loss[1],
    'independent_loss_reduction':loss[0]-loss[1],
    'native_accepted_reduction':sum(after['optimizationAcceptedReduction'])},indent=2))
