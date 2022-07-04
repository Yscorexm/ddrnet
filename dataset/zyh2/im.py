from PIL import Image
import numpy as np

from matplotlib import pyplot as plt

color_path = 'dataset\zyh2\group1\color_map\pose_0.png'
img_path = ['dataset\zyh2\group1\high_quality_depth\pose_0.png', 'dataset\zyh2\group1\depth_map\pose_0.png', 'dataset/zyh2/group1/refined_depth_map/dt_pose_0.png']
title = ['high depth', 'depth', 'output']
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(np.array(Image.open(color_path)))
ax[0, 0].set_title("RGB")

mask = np.array(Image.open('dataset\zyh2\group1\mask\pose_0.png'))
mask //= 255

mins = maxx = -1
for k, p in enumerate(img_path):
    img = Image.open(p)
    img = np.array(img)
    if k != 2:
        img = img * mask

    data = []
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] != 0:
                data.append(img[i][j])

    if k == 0:
        mins = min(data)
        maxx = max(data)

    img_show = np.zeros((x,y), dtype=np.float64)

    for i in range(x):
        for j in range(y):
            if img[i][j] != 0:
                img_show[i][j] = (img[i][j] - mins) / (maxx - mins)

                if k == 0:
                    print(img[i][j])
                

    i1, i2 = (k + 1) // 2, (k + 1) % 2
    ax[i1, i2].set_title(title[k])
    ax[i1, i2].imshow(img_show)

fig.tight_layout()
# plt.show()
plt.savefig('dataset\zyh2\im.jpg')