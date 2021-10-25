'''
the code aims to visualize the output of prediction.csv
'''
import pandas as pd
from cv2 import cv2
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# define images path
images_path = '../test'

if __name__ == '__main__':
    # check if images path exist
    if len(glob.glob(os.path.join(images_path, '*.jpg'))) == 0:
        print('images_path error ! No jpg file in given path')
        print('please modify \'images_path\' arg in code')
        exit()

    # load csv
    print('input csv name')
    csv_name = str(input())

    df = pd.read_csv(csv_name)
    print('load csv success')

    # make a folder for the csv results
    if not os.path.isdir(csv_name[:-4]):
        os.mkdir(csv_name[:-4])

    # get all the images
    for image_name in tqdm(os.listdir(images_path)):
        # load image with cv2
        image = cv2.imread(os.path.join(images_path, image_name))
        # change to rgb for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # filter the needed rows correspond to image_name
        sub_df = df[df['image_filename'] == image_name]

        # draw bounding box
        bbox_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (255, 255, 0), (0, 255, 255)]

        # define label index
        label_dict = [
            'Moth_eaten',
            'Mold',
            'Biological_exclusion',
            'Brown_spots',
            'Water_stains',
        ]

        # iterate through all the bbox and draw it
        for row in range(len(sub_df)):
            # get center x,y and w,h
            label, x, y, w, h, conf = int(sub_df.iloc[row]['label_id']), int(sub_df.iloc[row]['x']), int(
                sub_df.iloc[row]['y']), int(sub_df.iloc[row]['w']), int(sub_df.iloc[row]['h']), sub_df.iloc[row]['confidence']

            # define bbox color
            color = bbox_colors[label-1]

            # get (x1,y1) upper left and (x2,y2) lower right
            x1, y1 = x, y
            x2, y2 = x+w, y+h

            # draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            text = str(label_dict[label-1]) + str(conf)  # put label name
            cv2.putText(image, text, (x2+10, y2),
                        0, 1, color)
        # print(sub_df)

        # write image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("{}/{}".format(csv_name[:-4], image_name), image)
