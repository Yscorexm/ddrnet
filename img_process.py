from glob import glob
from PIL import Image
import numpy as np
import pathlib
import cv2
import os
from matplotlib import pyplot as plt

from_folder = 'dataset/face/Tester_1/refined_depth_map/dt_*'
# to_folder = 'dataset/face/Tester_1/group1/high_quality_depth/'
files = glob(from_folder)
print(files)

# if not os.path.isdir(to_folder):
#     os.mkdir(to_folder)

for img_path in files:
    filename = img_path.split('\\')[1]
    gray_img = np.array(Image.open(img_path))
    img = np.array(Image.open(img_path))
    # plt.imshow(img)
    processed = np.where(gray_img > 0, gray_img + 300, np.zeros_like(gray_img))
    non_zero = img[np.nonzero(img)]
    # print(processed.shape)
    print(non_zero.max(), non_zero.min())
    # plt.imshow(processed)
    # plt.show()
    # plt.savefig('frame1.png')

    # cv2.imwrite(to_folder + filename, processed)

