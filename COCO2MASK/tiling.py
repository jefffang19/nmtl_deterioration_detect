import numpy as np
from tqdm import tqdm
import os
from cv2 import cv2

whole_masks_path = 'mask_label'
tiling_msk_path = 'mask_patches'
tiling_img_path = 'img_patches'
image_path = '../train'

whole_masks = os.listdir(whole_masks_path)

for whole in tqdm(whole_masks):
    # image as bgr
    img = cv2.imread(os.path.join(
        image_path, whole.split('.')[0]+'.jpg'))

    # mask as grayscale
    msk = cv2.imread(os.path.join(whole_masks_path, whole), 0)

    '''
    tiling strategy:
    patch size = 416*416
    shift = 208
    will be divided into = 18*24 = 432 patches (per img)
    '''
    patch_size = 416
    shift_size = 208
    tiling_shape = [17, 23]

    # save mask and image into patches
    for y in range(tiling_shape[0]):
        for x in range(tiling_shape[1]):
            if x == tiling_shape[1]-1 and y == tiling_shape[0]-1:
                cv2.imwrite('{}/{}[{}][{}].png'.format(tiling_msk_path,
                                                       whole.split('.')[0], y, x), msk[-patch_size:, -patch_size:])
                cv2.imwrite('{}/{}[{}][{}].png'.format(tiling_img_path,
                                                       whole.split('.')[0], y, x), img[-patch_size:, -patch_size:, :])
            elif x == tiling_shape[1]-1:
                cv2.imwrite('{}/{}[{}][{}].png'.format(tiling_msk_path, whole.split('.')[
                            0], y, x), msk[shift_size*y: shift_size*y+patch_size, -patch_size:])
                cv2.imwrite('{}/{}[{}][{}].png'.format(tiling_img_path, whole.split('.')[
                            0], y, x), img[shift_size*y: shift_size*y+patch_size, -patch_size:, :])
            elif y == tiling_shape[0]-1:
                cv2.imwrite('{}/{}[{}][{}].png'.format(tiling_msk_path, whole.split('.')[
                            0], y, x), msk[-patch_size:, shift_size*x: patch_size+shift_size*x])
                cv2.imwrite('{}/{}[{}][{}].png'.format(tiling_img_path, whole.split('.')[
                            0], y, x), img[-patch_size:, shift_size*x: patch_size+shift_size*x, :])
            else:
                cv2.imwrite('{}/{}[{}][{}].png'.format(tiling_msk_path, whole.split('.')[
                    0], y, x), msk[shift_size*y: shift_size*y+patch_size, shift_size*x: patch_size+shift_size*x])
                cv2.imwrite('{}/{}[{}][{}].png'.format(tiling_img_path, whole.split('.')[
                    0], y, x), img[shift_size*y: shift_size*y+patch_size, shift_size*x: patch_size+shift_size*x, :])
