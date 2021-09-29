import glob
import os
from tqdm import tqdm

'''
train
'''
# check train split perviously made
f = open('mytrain.txt')

lines = f.readlines()
lines = [i[:-1].split('/')[-1] for i in lines]  # remove \n and path

# array use to collect patches name
collect = []

for l in tqdm(lines):
    # get all the patches of this image
    related_patches = glob.glob(os.path.join('img_patches', l[:-4]+'*.png'))

    # push all the patches in
    for r in related_patches:
        collect.append('/home/NMTL/yolo/' + r + '\n')

    f.close()

# write the patch label file
f = open('patch_train.txt', 'w')
for i in collect:
    f.write(i)

f.close()

'''
test
'''

# check test split perviously made
f = open('myvalid.txt')

lines = f.readlines()
lines = [i[:-1].split('/')[-1] for i in lines]  # remove \n and path

# array use to collect patches name
collect = []

for l in tqdm(lines):
    # get all the patches of this image
    related_patches = glob.glob(os.path.join('img_patches', l[:-4]+'*.png'))

    # push all the patches in
    for r in related_patches:
        collect.append('/home/NMTL/yolo/' + r + '\n')

    f.close()

# write the patch label file
f = open('patch_test.txt', 'w')
for i in collect:
    f.write(i)

f.close()
