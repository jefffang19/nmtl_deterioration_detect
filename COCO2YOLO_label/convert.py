'''
source:
https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/tool/coco_annotation.py
'''


import json
from collections import defaultdict
from tqdm import tqdm
import os

"""hyper parameters"""
json_file_path = 'train_coco.json'
output_path = 'yolo_label/'

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


"""write to txt"""
for key in tqdm(name_box_id.keys()):
    with open(os.path.join(output_path, key + '.txt'), 'w') as f:
        box_infos = name_box_id[key]
        for info in box_infos:
            x_min = int(info[0][0])
            y_min = int(info[0][1])
            x_max = x_min + int(info[0][2])
            y_max = y_min + int(info[0][3])

            x_center = (x_min + (x_max - x_min)/2)/info[2]
            y_center = (y_min + (y_max - y_min)/2)/info[3]
            _width = (x_max - x_min)/info[2]
            _height = (y_max - y_min)/info[3]

            box_info = " %d %f %f %f %f\n" % (
                int(info[1]), x_center, y_center, _width, _height)
            f.write(box_info)
