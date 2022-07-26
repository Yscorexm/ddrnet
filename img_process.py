from glob import glob
from PIL import Image
import numpy as np
import pathlib
import cv2
import os
from matplotlib import pyplot as plt

from_folder = 'dataset/hololens/group1/low_depth_map/*.png'
mask_dir = 'dataset/hololens/group1/mask'
# to_folder = 'dataset/face/Tester_1/group1/high_quality_depth/'
files = glob(from_folder)

# if not os.path.isdir(to_folder): 
#     os.mkdir(to_folder)

for img_path in files:
    filename = img_path.split('\\')[-1]
    mask = np.array(Image.open(f'{mask_dir}/{filename}'))
    img = np.array(Image.open(img_path))
    print(img.dtype)
    img = (img / 255 * mask).astype('uint16')
    print(img.dtype)
    non_zero = img[np.nonzero(img)]
    # print(list(non_zero))
    print(non_zero.max(), non_zero.min())
    plt.imshow(img)
    plt.show()
    exit(0)
