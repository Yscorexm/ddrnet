from PIL import Image
import numpy as np

for i in range(20):
    img_path = f'dataset/zyh2/group1/mask/pose_{i}'
    im1 = Image.open(img_path + '.jpg')
    npim = np.array(im1)
    npim[npim > 127] = 255
    npim[npim <= 127] = 0
    Image.fromarray(npim).save(img_path + '.png')
