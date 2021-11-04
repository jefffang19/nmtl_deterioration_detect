import numpy as np
from tqdm import tqdm
import os
from cv2 import cv2
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, regionprops_table

# load mask
read_path = 'mask_patches'
save_path = 'tile_yolo_label'

# load all the tiles
patches = os.listdir(read_path)

# load mask image
for _patch in tqdm(patches):
    mask = cv2.imread(os.path.join(read_path, _patch), 0)

    # use to save txt yolo labels
    yolo_labels = []

    # decompose masks
    masks = []
    for i in range(1, mask.max()+1):
        masks.append(np.where(mask == i, 1, 0))

    for c in range(mask.max()):
        # if label is empty
        if masks[c].max() == 0:
            continue

        # get bbox
        label_img = label(masks[c])
        regions = regionprops(label_img)

        for props in regions:
            miny, minx, maxy, maxx = props.bbox

            # yolo label
            # <object-class> <x_center> <y_center> <width> <height>
            # https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
            x_center = round((minx+(maxx-minx)/2)/2048, 6)
            y_center = round((miny+(maxy-miny)/2)/2048, 6)
            width = round((maxx-minx)/2048, 6)
            height = round((maxy-miny)/2048, 6)

            yolo_labels.append("{} {} {} {} {}\n".format(
                c, x_center, y_center, width, height))

    # write yolo label
    f = open(os.path.join(save_path, _patch.split('.')[0]+'.txt'), 'w')
    for _label in yolo_labels:
        f.write(_label)
    f.close()
