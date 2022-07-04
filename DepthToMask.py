import os
import numpy as np
import imageio
from matplotlib import image
import matplotlib.pyplot as plt
import cv2

def get_binary_img(img):
    # gray img to bin image
    bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            bin_img[i][j] = 255 if img[i][j] > 0 else 0
    return bin_img

# 计算灰度图的直方图
# def calchist_for_gray(imgname):
#     img = cv2.imread(imgname,  cv2.IMREAD_GRAYSCALE)
#     hist = cv2.calcHist([img], [0], None, [256], [0, 255])
#     plt.plot(hist, color="r")
#     plt.savefig("result_gray.jpg")

if __name__ == "__main__":
    h = 480
    w = 640
    # directs = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1),(-1,1),(-1,0)]
    directs = [(0,-1), (1,0), (0,1), (-1,0)]
    for j in range(1, 3):
        path = "E:\\zhan\\VE450\\RGBD_Dataset\\toU1\\ttt\\Tester_" + str(j) + "\\depth_map"
        maskPath = "E:\\zhan\\VE450\\RGBD_Dataset\\toU1\\ttt\\Tester_" + str(j) + "\\mask_map"
        isExists=os.path.exists(maskPath)
        if not isExists:
            os.makedirs(maskPath)
    
        for i in range(0, 20):
            seeds = [(310,430)]
            seeds.append((310,240))
            imgname = path + "\\pose_" + str(i) + ".png"
            # img = image.imread(imgname)
            img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
            depth = img.copy()
            # calchist_for_gray(imgname)
            # bin_img = get_binary_img(img)
            visited = np.zeros(shape=(img.shape), dtype=np.uint8)
            out_img = np.zeros(shape=(img.shape), dtype=np.uint8)
            while len(seeds):
                seed = seeds.pop(0)
                x = seed[0]
                y = seed[1]
                # visit point (x,y)
                visited[y][x] = 1
                for direct in directs:
                    cur_x = x + direct[0]
                    cur_y = y + direct[1]
    	            # 非法
                    if cur_x < 0 or cur_y < 0 or cur_x >= w or cur_y >= h:
                        continue
                    diff = abs(int(img[cur_y][cur_x]) - int(img[y][x]))
                    if (not visited[cur_y][cur_x]) and (diff < 4):
                        out_img[cur_y][cur_x] = 255
                        visited[cur_y][cur_x] = 1
                        seeds.append((cur_x,cur_y))
                    # else:
                    #     depth[cur_y][cur_x] = 0
            # for k in range(h):
            #     for q in range(w):
            #         # area that is the different with the seeds goes to white
            #         if (out_img[k][q] == 0):
            #             depth[k][q] = 255
            #         else:
            #             depth[k][q] = 0
            # depth = get_binary_img(depth)
            # imageio.imwrite(maskPath + "\\pose_" + str(i) + ".png", depth)
            imageio.imwrite(maskPath + "\\pose_" + str(i) + ".png", out_img)
            # calchist_for_gray(maskPath + "\\pose_" + str(i) + ".png")
