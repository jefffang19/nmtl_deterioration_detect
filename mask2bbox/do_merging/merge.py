import os
import cv2

path1 = "in1"
path2 = "in2"
path_out = "out1"

pics = os.listdir(path1)

for i in pics:
	img1 = cv2.imread(os.path.join(path1, i))
	img2 = cv2.imread(os.path.join(path2, i))

	cv2.imwrite(os.path.join(path_out, i), img1+img2)
