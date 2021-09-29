import cv2
import numpy as np

img = cv2.imread('yolo/image_00009.jpg')
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl0 = clahe.apply(img[..., 0])
cl1 = clahe.apply(img[..., 1])
cl2 = clahe.apply(img[..., 2])

cl = np.stack([cl0, cl1, cl2], axis=-1)

print(cl.shape)

cv2.imwrite('clahe_2.jpg',cl)
