import os
import h5py
import numpy as np
import openslide
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

slide_folder = 'SLIDE_FOLDER_PATH'
all_data = np.array(pd.read_excel('ALL_DATA_PATH', engine='openpyxl',  header=None))
root_folder = 'ROOT_FOLDER_PATH' 
define_patch_size = 1024 
patch_folder = 'ROOT_FOLDER_PATH' + +'/patches_'+str(define_patch_size)+'/'
save_folder = 'SAVE_FOLDER_PATH'

if(define_patch_size == 2048):
    scale = 8
    save_name = '5x'
elif(define_patch_size == 1024):
    scale = 4
    save_name = '10x'
elif(define_patch_size == 512):
    scale = 2
    save_name = '20x'

if(not os.path.exists(root_folder)):
    os.makedirs(root_folder)

svs2uuid = {}
for i in all_data:
    svs2uuid[i[1].rstrip('\n')] = i[0]

def generate_patch(patch_file_name):
    patch_path = os.path.join(patch_folder, patch_file_name)
    slide_path = os.path.join(slide_folder, svs2uuid[patch_file_name.replace('h5', 'svs')], patch_file_name.replace('h5', 'svs'))

    f = h5py.File(patch_path, 'r')
    coords = f['coords']
    patch_level = coords.attrs['patch_level']
    patch_size = coords.attrs['patch_size']
    slide = openslide.open_slide(slide_path)
    try:
        magnification = int(float(slide.properties['aperio.AppMag']))
    except:
        magnification = 40
    save_path = save_folder + patch_file_name.replace('.h5', '')
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
        if(magnification == 40):
            resized_patch_size = int(patch_size/scale)
        elif(magnification == 20):
            resized_patch_size = int(patch_size/(scale/2))
        for coord in tqdm(coords):
            coord = coord.astype(np.int)
            patch = slide.read_region(coord, int(patch_level), (int(patch_size), int(patch_size))).convert('RGB')
            patch = patch.resize((resized_patch_size, resized_patch_size))
            patch_name = str(coord[0]) + '_' + str(coord[1]) + '.png'
            patch_save_path = os.path.join(save_path, patch_name)
            patch.save(patch_save_path)
    else:
        print(patch_file_name + ': has been processed !')

pool = ThreadPoolExecutor(max_workers=16)
all_file_names = np.array(os.listdir(patch_folder))
for patch_file_name in all_file_names:
    pool.submit(generate_patch, patch_file_name)
pool.shutdown(wait=True)


