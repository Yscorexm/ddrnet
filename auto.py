from cgi import test
import subprocess
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import depth2mesh
from depth2normal import show_result

def depth_process(src, dst, fn, mask_dir=None):
    files = glob.glob(f'{src}/*')
    for img_path in files:
        filename = img_path.split('\\')[-1]
        img = np.array(Image.open(img_path))
        if mask_dir:
            mask = np.array(Image.open(f'{mask_dir}/{filename}'))
            img = img * mask // 255
        processed = np.where(img > 0, fn(img), np.zeros_like(img)).astype('uint16')
        # if '2' in filename:
        #     plt.imshow(processed)
        #     plt.show()
        save_im = Image.fromarray(processed)
        save_im.save(f'{dst}/{filename}')

def img_show(dir, filename):
    color_path = f'{dir}/color_map/{filename}'
    depth_path = f'{dir}/depth_map/{filename}'
    high_path = f'{dir}/high_quality_depth/{filename}'
    output_path = f'{dir}/refined_depth_map/dt_{filename}'
    paths = [high_path, depth_path, output_path]
    title = ['high depth', 'depth', 'output']
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(np.array(Image.open(color_path)))
    ax[0, 0].set_title("RGB")
    
    for k, p in enumerate(paths):
        img = np.array(Image.open(p))
        # img_show = np.where(img > 0, img - 300, np.zeros_like(img))
        non_zero = img[np.nonzero(img)]
        mins, maxx = min(non_zero), max(non_zero)
        img_show = np.where(img > 0, (img - mins) / (maxx - mins), np.zeros_like(img))
        i1, i2 = (k + 1) // 2, (k + 1) % 2
        ax[i1, i2].set_title(title[k])
        ax[i1, i2].imshow(img_show)
    fig.tight_layout()
    mp = False
    img_dir = f'{dir}/origin_img' if mp else f'{dir}/img'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig(f'{img_dir}/{filename}')
    plt.close()

def get_model(dir, origin, x):
    model_dir = f'{dir}/model'
    filename = 'frame_000001.png' if origin else f'pose_{x}.png'
    depth_path = f'{dir}/depth_map/{filename}'
    high_path = f'{dir}/high_quality_depth/{filename}'
    output_path = f'{dir}/refined_depth_map/dt_{filename}'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    for dp, name in zip([depth_path, high_path, output_path], ['depth', 'high_depth', 'output']):
        depth2mesh.create_obj(depthPath=dp, objPath=f'{model_dir}/{name}.obj', mtlPath='', matName='', useMaterial=False)

if __name__ == '__main__':

    base_dir = 'dataset/face'
    # Generate train.csv
    # train_csv_path = f'{base_dir}/train10.csv'
    # csv_contents = []
    # for k in range(1, 11):
    #     # Generate csv files
    #     test_dir = f'{base_dir}/Tester_{k}'
    #     csv_base_dir = f'../{test_dir}'
    #     dir_names = ['depth_map', 'high_quality_depth', 'color_map', 'mask']
    #     for i in range(20):
    #         csv_contents.append(','.join(f'{csv_base_dir}/{name}/pose_{i}.png' for name in dir_names) + '\n')
    # with open(train_csv_path, 'w') as fw:
    #     fw.writelines(csv_contents)

    first = [1, 67, 68, 69, 70]
    two = [144, 145, 146, 147]

    for k in [1]:
        # Generate csv files
        test_dir = f'{base_dir}/Tester_{k}'
        csv_path = f'{test_dir}/test_{k}.csv'
        csv_contents = []
        csv_base_dir = f'../{test_dir}'
        dir_names = ['depth_map', 'high_quality_depth', 'color_map', 'mask']
        for i in range(20):
            csv_contents.append(','.join(f'{csv_base_dir}/{name}/pose_{i}.png' for name in dir_names) + '\n')
        with open(csv_path, 'w') as fw:
            fw.writelines(csv_contents)
        
        # Generate depth map that fit DDRNet
        depth_folders = [f'{test_dir}/{name}' for name in dir_names[:2]]
        raw_depth_folders = [f'{test_dir}/raw_{name}' for name in dir_names[:2]]
        if not os.path.exists(raw_depth_folders[0]):
            for i in range(2):
                src, dst = depth_folders[i], raw_depth_folders[i]
                os.rename(src, dst)
                os.mkdir(src)
            
        depth_process(raw_depth_folders[0], depth_folders[0], lambda img: (img - 1) * 3)
        depth_process(raw_depth_folders[1], depth_folders[1], lambda img: img)

        origin = False
        low = 0
        up = 255
        if origin:
            test_dir = 'dataset/20170907/group2'
            csv_path = 'dataset/test.csv'
            low = 500
            up = 3000
        output_dir = f'{test_dir}/refined_depth_map'
        checkpoint_dir = '../download/split/'
        # checkpoint_dir = '../download/face10/'
        # checkpoint_dir = '../log/cscd/noBN_L1_sd100_B16/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        os.chdir('src')
        subprocess.run([
            'python', 
            'evaluate.py', 
            '--dnnet=convResnet', 
            '--dtnet=hypercolumn', 
            f'--sample_dir=../{output_dir}',
            f'--checkpoint_dir={checkpoint_dir}',
            f'--csv_path=../{csv_path}',
            f'--low_thres={low}',
            f'--up_thres={up}',
            '--image_size=400'
            ], shell=True)
        os.chdir('..')
        for i in range(20):
            img_show(test_dir, f'pose_{i}.png')
            show_result(test_dir, f'pose_{i}.png')
        # get_model(test_dir, origin, k)
