from shutil import copyfile
import os
from tqdm import tqdm

src_path = 'tile_yolo_label'
dst_path = 'img_patches'
txts = os.listdir(src_path)

for src in tqdm(txts):
    copyfile(os.path.join(src_path, src), os.path.join(dst_path, src))
