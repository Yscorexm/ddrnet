from collections import deque
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
import glob


def depth2mask(depth):
    seed = (130, 150)
    mask = np.zeros_like(depth, dtype='uint8')
    dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    dq = deque([seed])
    mask[seed] = 255
    non_zero = []
    while dq:
        node = dq.pop()
        non_zero.append(depth[node])
        for direction in dirs:
            nd = (direction[0] + node[0], direction[1] + node[1])
            if mask[nd] == 255: continue
            if nd[0] > 150: continue
            if (abs(depth[nd] - depth[node]) > 100): continue
            dq.append(nd)
            mask[nd] = 255
    # print(max(non_zero), min(non_zero))
    return mask


if __name__ == '__main__':
    base_dir = 'dataset/hololens'
    for k in range(1, 4):
        test_dir = f'{base_dir}/group{k}'
        files = glob.glob(test_dir + '/color_map/*.png')
        filenames = [file_path.split('\\')[-1] for file_path in files]
        depth_dir = f'{test_dir}/raw_depth_map'
        mask_dir = f'{test_dir}/mask'
        for filename in filenames:
            depth_path = f'{depth_dir}/{filename}'
            depth = np.array(Image.open(depth_path))
            mask = depth2mask(depth)
            depth * mask / 255
            mask_path = f'{mask_dir}/{filename}'
            save_im = Image.fromarray(mask)
            save_im.save(mask_path)
