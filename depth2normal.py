from turtle import color
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def center_crop(input, patch_size=400):
    if len(input.shape) == 3:
        H, W, _ = input.shape
    else:
        H, W = input.shape
    h_min = (H - patch_size) // 2
    w_min = (W - patch_size) // 2
    if len(input.shape) == 3:
        output = input[h_min:h_min + patch_size, w_min:w_min + patch_size, :]
    else:
        output = np.array(input, dtype=np.float32)[h_min:h_min + patch_size, w_min:w_min + patch_size]
    return output

def compute_normal(depth, mask):
    depth *= mask
    d_im = depth.astype("float64")

    zy, zx = np.gradient(d_im)

    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    # zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)     
    # zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255

    normal = normal.astype('uint8')
    gray_img = cv2.cvtColor(normal[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    # gray_img *= mask
    gray_img = np.where(mask == 1, gray_img, 255)
    return gray_img

def show_result(dir, origin, x):
    # path = 'dataset/face/Tester_1/depth_map/pose_0.png'
    filename = 'frame_000001.png' if origin else f'pose_{x}.png'
    color_path = f'{dir}/color_map/{filename}'
    depth_path = f'{dir}/depth_map/{filename}'
    # depth_path = f'{dir}/high_quality_depth/{filename}'
    dn_path = f'{dir}/refined_depth_map/dn_{filename}'
    dt_path = f'{dir}/refined_depth_map/dt_{filename}'
    mask_path = f'{dir}/mask/{filename}'
    paths = [depth_path, dn_path, dt_path]
    mask = center_crop(np.array(Image.open(mask_path))).astype('uint8')
    mask //= 255

    title = ['Raw Depth', 'Denoised Depth', 'Refined Depth']
    fig, ax = plt.subplots(1, 4)
    fig.set_figheight(4.5)
    fig.set_figwidth(17)
    color = np.array(Image.open(color_path))
    color = center_crop(color)
    ax[0].imshow(color)
    ax[0].set_title("Color Image", fontsize=25)
    ax[0].axis('off')
    
    for k, p in enumerate(paths):
        depth = np.array(Image.open(p))
        if mask.shape[0] < min(depth.shape):
            depth = center_crop(depth)
        gray_img = compute_normal(depth, mask)
        ax[k + 1].set_title(title[k], fontsize=25)
        ax[k + 1].imshow(gray_img, cmap='gray', vmin=0, vmax=255)
        ax[k + 1].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    mp = True
    img_dir = f'{dir}/origin_normal' if mp else f'{dir}/normal'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig(f'{img_dir}/{filename}')
    plt.close()


if __name__ == '__main__':
    # basedir = 'dataset/face/Tester_1'
    basedir = 'dataset/20170907/group2'
    show_result(basedir, True, 1)
