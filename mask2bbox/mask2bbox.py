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

# define images path
mask_path = 'predictions/'
images_name = 'train_list.txt'

if __name__ == '__main__':

    # iterate through all the images
    f = open(images_name)
    im_names = f.readlines()
    im_names = [i[:-1] for i in im_names]

    # load image with cv2
    image = cv2.imread(os.path.join('../train', im_names[0]))
    print(image.shape)
    # change to rgb for matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get mask corresponding to this image
    masks = glob.glob(os.path.join(
        mask_path, '{}*.png'.format(im_names[0][:-4])))

    # sort masks classes from 1 to 5
    masks = sorted(masks)

    # now parse 5 classes one by one
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

        # draw bonding box
        fig, ax = plt.subplots()
        ax.imshow(image)

        for props in regions:

            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-b', linewidth=2.5)

        plt.show()

    # define label index
    # label_dict = {
    #     'Moth_eaten': 0,
    #     'Mold': 1,
    #     'Biological_exclusion': 2,
    #     'Brown_spots': 3,
    #     'Water_stains': 4,
    # }
