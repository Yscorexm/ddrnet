from PIL import Image
import numpy as np

from matplotlib import pyplot as plt

def center_crop(input, patch_size=400):
    H, W = input.shape
    h_min = (H - patch_size) // 2
    w_min = (W - patch_size) // 2
    output = np.array(input, dtype=np.float32)[h_min:h_min + patch_size, w_min:w_min + patch_size]
    return output


ref = 'dataset/20170907/group2/high_quality_depth/frame_000001.png'
out = 'sample/dt_frame_000001.png'
mask = 'dataset/20170907/group2/mask/frame_000001.png'



mask = np.array(Image.open(mask))
ref = np.array(Image.open(ref))
mask, ref = center_crop(mask), center_crop(ref)
out = np.array(Image.open(out))

# print(mask.shape, ref.shape, out.shape)

# plt.subplot(1, 3, 1)
# plt.imshow(ref)
# plt.subplot(1, 3, 2)
# plt.imshow(mask)
# plt.subplot(1, 3, 3)
# plt.imshow(out)
# plt.show()

diff = np.abs(ref * mask // 255.0 - out * mask // 255.0)
mean_err = np.sum(diff) / np.count_nonzero(mask)
print(mean_err)
non_zero = ref[np.nonzero(ref)]
print(non_zero.max(), non_zero.min())