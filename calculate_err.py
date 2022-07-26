from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def center_crop(input, patch_size=400):
    H, W = input.shape
    h_min = (H - patch_size) // 2
    w_min = (W - patch_size) // 2
    output = np.array(input, dtype=np.float32)[h_min:h_min + patch_size, w_min:w_min + patch_size]
    return output


dir = 'dataset/face/Tester_1'
for i in range(20):
    filename = f'pose_{i}.png'
    depth_path = f'{dir}/depth_map/{filename}'
    high_path = f'{dir}/high_quality_depth/{filename}'
    output_path = f'{dir}/refined_depth_map/dt_{filename}'
    mask_path = f'{dir}/mask/{filename}'
    mask = np.array(Image.open(mask_path))
    ref = np.array(Image.open(high_path))
    mask, ref = center_crop(mask), center_crop(ref)
    mask /= 255
    out = np.array(Image.open(output_path))

    # print(mask.shape, ref.shape, out.shape)

    # plt.subplot(1, 3, 1)
    # plt.imshow(ref)
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask)
    # plt.subplot(1, 3, 3)
    # plt.imshow(out)
    # plt.show()


    diff = np.abs(ref * mask - out * mask)
    mean_err = np.sum(diff) / np.count_nonzero(mask)
    # print(mean_err)
    non_zero = ref[np.nonzero(ref)]
    # print(non_zero.max(), non_zero.min())
    non_zero_out = out[np.nonzero(out)]
    # print(non_zero_out.max(), non_zero_out.min())
    out = (out - 1) / (259 - 1) * (224 - 18) + 18
    diff = np.abs(ref * mask - out * mask)
    mean_err = np.sum(diff) / np.count_nonzero(mask)
    print(mean_err / 255 * 10)
