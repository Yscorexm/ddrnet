from PIL import Image
import numpy as np

from matplotlib import pyplot as plt


img_path = ['dataset/20170907/group1/high_quality_depth/frame_000001.png', 'dataset/20170907/group1/depth_map/frame_000001.png',
            'dataset/20170907/group1/depth_filled/frame_000001.png', 'sample/dt_frame_000001.png']
mins = maxx = -1
for k, p in enumerate(img_path):
    img = Image.open(p)

    img = np.array(img)

    data = []
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] != 0:
                data.append(img[i][j])

    mins = min(data)
    maxx = max(data)

    img_show = np.zeros((x,y), dtype=np.float64)

    for i in range(x):
        for j in range(y):
            if img[i][j] != 0:
                img_show[i][j] = (img[i][j] - mins) / (maxx - mins)
                # print(img_show[i][j])

    plt.subplot(2, 2, k + 1)
    plt.imshow(img_show)

# plt.show()
plt.savefig('output.jpg')