import cv2
import numpy as np

img = cv2.imread('yolo/image_00077.jpg', 0)

print('avg intensity: {}\nstd intensity: {}'.format(np.mean(img), np.std(img)))
