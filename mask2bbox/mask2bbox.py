'''
the code aims to visualize the output of prediction.csv
'''
import pandas as pd
from cv2 import cv2
import matplotlib.pyplot as plt
import os
import glob
from skimage.measure import label, regionprops
import numpy as np
from tqdm import tqdm

# define images path
mask_path = 'predictions_test/'
images_name = 'test_list.txt'


if __name__ == '__main__':

    # iterate through all the images
    f = open(images_name)
    im_names = f.readlines()
    im_names = [i[:-1] for i in im_names]

    # write csv file
    f = open('predictions.csv', 'w')
    f.write(',image_filename,label_id,x,y,w,h,confidence\n')
    f.close()

    # iterate through all the image
    for idx, im_name in tqdm(enumerate(im_names)):
        # get mask corresponding to this image
        masks = glob.glob(os.path.join(
            mask_path, '{}*.png'.format(im_name[:-4])))

        # sort masks classes from 1 to 5
        masks = sorted(masks)

        # now parse 5 classes one by one
        f = open('predictions.csv', 'a')
        for _cls in range(5):
            # load mask
            msk = cv2.imread(masks[_cls], 0)

            # thresholding
            thresh_msk = np.where(msk > 0, 255, 0).astype(np.uint8)
            '''
                # the fourth class is usually bigger, so we often need to concat masks from many patches
                # it might be a good idea to do dilate and erode  to combine some disconnected parts
                '''
            if _cls == 4:
                kernel = np.ones((3, 3), np.uint8)
                dilation = cv2.dilate(thresh_msk, kernel, iterations=1)
                thresh_msk = cv2.erode(dilation, kernel, iterations=1)

            # now we get connected component
            label_img = label(thresh_msk)

            # we get regions
            regions = regionprops(label_img)

            # get bbox
            for props in regions:
                # get coord for bbox
                miny, minx, maxy, maxx = props.bbox

                # calculate cofidence
                confidence = np.sum(
                    msk[miny:maxy, minx:maxx]) / np.sum(thresh_msk[miny:maxy, minx:maxx])
                confidence = round(confidence, 3)

                # calculate
                f.write('{},{},{},{},{},{},{},{}\n'.format(
                    idx, im_name[:-4], _cls+1, minx, miny, maxx-minx, maxy-miny, confidence))

        f.close()
