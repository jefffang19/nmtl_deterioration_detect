import cv2
import numpy as np

img = cv2.imread('yolo/image_00077.jpg')


alpha = 1.0 # Simple contrast control
beta = 30    # Simple brightness control

new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

cv2.imwrite('br_cr.jpg',new_image)
