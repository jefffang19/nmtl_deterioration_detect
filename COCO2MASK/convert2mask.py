'''
source:
https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/coco_annotation.py
'''


import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
import cv2

"""hyper parameters"""
json_file_path = 'train_20210928.json'
output_path = 'mask_label/'

"""load json file"""
name_box_id = defaultdict(list)
id_hw = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)


"""generate labels"""
images = data['images']
annotations = data['annotations']


'''save image height width'''
# sanity check
for im in tqdm(images):
    id = im['id']
    width = im['width']
    height = im['height']

    id_hw[id] = [width, height]

for ant in tqdm(annotations):
    id = ant['image_id']
    cat = ant['category_id']

    name_box_id[id].append([ant['bbox'], cat-1, id_hw[id][0], id_hw[id][1]])


# collect shapes
shape_collects = []
"""write to png"""
for key in tqdm(name_box_id.keys()):
    box_infos = name_box_id[key]

    # init shape
    mask = np.zeros(shape=(box_infos[0][3], box_infos[0][2]), dtype='int')
    shape_collects.append([box_infos[0][3], box_infos[0][2]])

    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        mask[y_min:y_max, x_min:x_max] = int(info[1]) + 1

    cv2.imwrite(os.path.join(output_path, key + '.png'), mask)

# calculate max width and height
shape_collects = np.array(shape_collects).T

max_w = shape_collects.max(axis=0)
max_h = shape_collects.max(axis=1)

# print(shape_collects)
print('max_w : {}\nmax_h: {}'.format(max_w, max_h))
