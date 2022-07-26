import glob
import os
import shutil
import subprocess
from auto import img_show, depth_process
from depth2normal import show_result, center_crop
from PIL import Image, ImageFilter
from get_mask import depth2mask
import numpy as np

def project(low, up):
    def helper(img):
        return (img - low) / (up - low) * 1200 + 1000
    return helper

def project2uint8(low, up):
    def helper(img):
        return (img - low) * 255 // (up - low)
    return helper

def gaussian_filter(src, dst):
    filenames = glob.glob(f'{src}/*.png')
    for imgname in filenames:
        filename = imgname.split('\\')[-1]
        new_file_name = f'{dst}/{filename}'
        img = Image.open(imgname).convert(mode='L')
        im2 = img.filter(ImageFilter.GaussianBlur)
        im2.save(new_file_name)

def make_gif(dir):
    files = glob.glob(f'{dir}/*.png')
    frames = [Image.open(file) for file in files]
    frame_one = frames[0]
    frame_one.save(f"{dir}/result.gif", format="GIF", append_images=frames,
               save_all=True, duration=500, loop=0)

def generate_mask(test_dir, depth_dir):
    filenames = [x.split('\\')[-1] for x in glob.glob(f'{depth_dir}/*.png')]
    mask_dir = f'{test_dir}/mask'
    for filename in filenames:
        depth_path = f'{depth_dir}/{filename}'
        depth = np.array(Image.open(depth_path))
        mask = depth2mask(depth)
        mask_path = f'{mask_dir}/{filename}'
        save_im = Image.fromarray(mask)
        save_im.save(mask_path)

def depth_error(test_dir, filenames, low, up, crop_size=260, num=50):
    depth_dir = f'{test_dir}/refined_depth_map'
    ref_dir = f'{test_dir}/high_quality_depth'
    mask_dir = f'{test_dir}/mask'
    result = 0
    for filename in filenames:
        depth_path = f'{depth_dir}/dt_{filename}'
        ref_path = f'{ref_dir}/{filename}'
        mask_path = f'{mask_dir}/{filename}'
        depth = np.array(Image.open(depth_path))
        ref = center_crop(np.array(Image.open(ref_path)), patch_size=crop_size)
        mask = center_crop(np.array(Image.open(mask_path)), patch_size=crop_size).astype('uint8')
        mask //= 255
        depth = depth * mask
        ref = ref * mask
        diff = np.abs(depth - ref)
        mean_err = np.sum(diff) / np.count_nonzero(mask)
        mean_err_cm = mean_err / 1200 * (up - low) / num
        result += mean_err_cm
    return result, len(filenames)
    

base_dir = 'dataset/hololens_all'
folders = glob.glob(f'{base_dir}/*')
depth_results = [[0, 0], [0, 0], [0, 0]]
time_results = [[0, 0], [0, 0], [0, 0]]
output_time_dir = 'time'
if not os.path.exists(output_time_dir):
    os.mkdir(output_time_dir)
for folder in folders:
    foldername = folder.split('\\')[-1]
    k = foldername[foldername.find("group") + 5:]
    test_dir = folder.replace('\\', '/')
    print(test_dir)

    dir_names = ['depth_map', 'high_quality_depth', 'color_map', 'mask']
    train_dir_names = ['low_depth_map', 'high_quality_depth', 'color_map', 'mask']
    mask_dir = f'{test_dir}/mask'
    depth_folders = [f'{test_dir}/{name}' for name in dir_names[:2]]
    raw_depth_folders = [f'{test_dir}/raw_{name}' for name in dir_names[:2]]
    
    # rename filename
    if not os.path.exists(depth_folders[0]):
        files = glob.glob(test_dir + '/depth/*.png')
        filenames = [file_path.split('\\')[-1] for file_path in files]
        if 'frame' not in filenames[0]:
            for i, filename in enumerate(filenames):
                new_filename = f'frame_{i}.png'
                for final_path in [f'{test_dir}/rgb', f'{test_dir}/depth']:
                    src = f'{final_path}/{filename}'
                    dst = f'{final_path}/{new_filename}'
                    os.rename(src, dst)
    
    # change to DDRNet format
    if not os.path.exists(depth_folders[0]):
        shutil.copytree(f'{test_dir}/depth', depth_folders[0])
        os.rename(f'{test_dir}/depth', depth_folders[1])
    if not os.path.exists(f'{test_dir}/color_map'):
        os.rename(f'{test_dir}/rgb', f'{test_dir}/color_map')
    if not os.path.exists(raw_depth_folders[0]):
        for i in range(2):
            src, dst = depth_folders[i], raw_depth_folders[i]
            os.rename(src, dst)
            os.mkdir(src)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
        generate_mask(test_dir, f'{test_dir}/raw_depth_map')
    
    # Generate train.csv and test.csv
    csv_path = f'{test_dir}/test_{k}.csv'
    train_csv_path = f'{test_dir}/train_{k}.csv'
    csv_contents = []
    train_csv_contents = []
    csv_base_dir = f'../{test_dir}'
    files = glob.glob(test_dir + '/color_map/*.png')
    filenames = [file_path.split('\\')[-1] for file_path in files]
    for filename in filenames:
        csv_contents.append(','.join(f'{csv_base_dir}/{name}/{filename}' for name in dir_names) + '\n')
    with open(csv_path, 'w') as fw:
        fw.writelines(csv_contents)
    
    for filename in filenames:
        train_csv_contents.append(','.join(f'{csv_base_dir}/{name}/{filename}' for name in train_dir_names) + '\n')
    with open(train_csv_path, 'w') as fw:
        fw.writelines(train_csv_contents)
    
    
    # normalization
    distancemap = {
        '1_0_5': (2500, 3700), '1_1_0': (4200, 5400), '1_1_5': (6800, 8000),
        '2_0_5': (2400, 3500), '2_1_0': (4200, 5400), '2_1_5': (6200, 7600),
        '3_0_5': (2300, 3900), '3_1_0': (4300, 6500), '3_1_5': (6600, 8600),
        '4_0_5': (2500, 3800), '4_1_0': (3900, 5500), '4_1_5': (6000, 8400),
        '5_0_5': (2750, 4100), '5_1_0': (4400, 6000), '5_1_5': (6900, 8600),
        '6_0_5': (2350, 3700), '6_1_0': (4400, 6200), '6_1_5': (6900, 8900),
        '7_0_5': (2500, 3900), '7_1_0': (4400, 5700), '7_1_5': (6700, 8500)
    }
    low, up = distancemap[k]
    # depth value 50 -> 1cm

    # Generate processed depth map
    uint8_dir = f'{test_dir}/uint8'
    middle_dir = f'{test_dir}/low_uint8'
    low_depth_dir = f'{test_dir}/low_depth_map'
    if not os.path.exists(uint8_dir):
        os.mkdir(uint8_dir)
    if not os.path.exists(middle_dir):
        os.mkdir(middle_dir)
    if not os.path.exists(low_depth_dir):
        os.mkdir(low_depth_dir)
    depth_process(raw_depth_folders[0], depth_folders[0], project(low, up))
    depth_process(raw_depth_folders[1], depth_folders[1], project(low, up))
    depth_process(raw_depth_folders[0], uint8_dir, project2uint8(low, up), mask_dir)
    gaussian_filter(uint8_dir, middle_dir)
    depth_process(middle_dir, low_depth_dir, lambda img: img / 255 * 1200 + 1000)

    crop_size = 260

    # Run evaluation network
    output_dir = f'{test_dir}/refined_depth_map'
    # checkpoint_dir = '../download/hololens/'
    checkpoint_dir = '../download/stop/'
    # checkpoint_dir = '../log/cscd/noBN_L1_sd100_B16/'

    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    # os.chdir('src')
    # with open(f'../{output_time_dir}/{k}', 'w') as file_out:
    #     subprocess.run([
    #         'python', 
    #         'evaluate.py', 
    #         '--dnnet=convResnet', 
    #         '--dtnet=hypercolumn', 
    #         f'--sample_dir=../{output_dir}',
    #         f'--checkpoint_dir={checkpoint_dir}',
    #         f'--csv_path=../{train_csv_path}',
    #         f'--low_thres=1000',
    #         f'--up_thres=2200',
    #         f'--image_size={crop_size}'
    #         ], shell=True, stdout=file_out)
    # os.chdir('..')

    # Show result gif
    for filename in filenames:
        img_show(test_dir, filename)
        show_result(test_dir, filename, crop_size, True)
    result_dir = f'{test_dir}/normal'
    make_gif(result_dir)

    # Calculate mean depth error
    if '_0_5' in k:
        dis = 0
    elif '_1_0' in k:
        dis = 1
    elif '_1_5' in k:
        dis = 2
    
    errors = depth_error(test_dir, filenames, low, up)
    depth_results[dis][0] += errors[0]
    depth_results[dis][1] += errors[1]

    with open(f'{output_time_dir}/{k}', 'r') as fin:
        last = fin.readlines()[-1]
        start_pos = last.find('tt_time: ') + 9
        end_pos = last.find('s; avg')
        time_results[dis][0] += float(last[start_pos:end_pos])
        time_results[dis][1] += len(filenames)


map_wd = {0: '0.5 m', 1: '1.0 m', 2: '1.5 m'}
for i, depth_result in enumerate(depth_results):
    print(f'working distance: {map_wd[i]}')
    print(f'mean_depth_error: {depth_result[0] / depth_result[1]}')
    print(f'processing time: {time_results[i][0] / time_results[i][1]}')
    print(f'memory use: ')


