import cv2
import numpy as np

img = cv2.imread('yolo/image_00009.jpg')


alpha = 5.0 # Simple contrast control
beta = 100    # Simple brightness control

new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl0 = clahe.apply(new_image[..., 0])
cl1 = clahe.apply(new_image[..., 1])
cl2 = clahe.apply(new_image[..., 2])

new_image = np.stack([cl0, cl1, cl2], axis=-1)

cv2.imwrite('br_cr_clahe.jpg',new_image)
