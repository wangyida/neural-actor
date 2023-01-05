import cv2
import numpy as np
import glob
import os
import json
import yaml
from yaml.loader import SafeLoader

def read(intri_file, key, dt='mat'):
    if dt == 'mat':
        output = intri_file.getNode(key).mat()
    elif dt == 'list':
        results = []
        n = intri_file.getNode(key)
        for i in range(n.size()):
            val = n.at(i).string()
            if val == '':
                val = str(int(n.at(i).real()))
            if val != 'none':
                results.append(val)
                output = results
    return output

def get_cams():
    intri = cv2.FileStorage('intri.yml', cv2.FILE_STORAGE_READ)
    camnames = read(intri, 'names', dt='list')
    extri = cv2.FileStorage('extri.yml', cv2.FILE_STORAGE_READ)
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    # for i in range(90):
    for i in camnames:
        cams['K'].append(intri.getNode('K_{}'.format(i)).mat())
        cams['D'].append(
            intri.getNode('dist_{}'.format(i)).mat().T)
        cams['R'].append(extri.getNode('Rot_{}'.format(i)).mat())
        cams['T'].append(extri.getNode('T_{}'.format(i)).mat() * 1000)
    return camnames, cams


def get_img_paths(camnames):
    all_ims = []
    # 90 is the number of camera views
    # for i in range(90):
    for i in camnames:
        data_root = '{}'.format(i)
        ims = glob.glob(os.path.join(data_root, '*.png'))
        if ims == []:
            ims = glob.glob(os.path.join(data_root, '*.jpg'))
        ims = np.array(sorted(ims))
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


camnames, cams = get_cams()
img_paths = get_img_paths(camnames)

annot = {}
annot['cams'] = cams

ims = []
"""
for img_path, kpt in zip(img_paths, kpts2d):
    data = {}
    data['ims'] = img_path.tolist()
    ims.append(data)
"""
for img_path in img_paths:
    data = {}
    data['ims'] = img_path.tolist()
    ims.append(data)
annot['ims'] = ims

np.save('annots.npy', annot)
np.save('annots_python2.npy', annot, fix_imports=True)
