from PIL import Image, UnidentifiedImageError
from utils.extras import get_engine
import os
import torch
import numpy as np
import time
from multiprocessing import Pool
from functools import partial
import shutil
import pickle
from collections import defaultdict
# This simply removes unloadable images. Since some of the images can be 

def clean_images_worker(args):
    class_id, root_folder = args
    folder_path = os.path.join(root_folder, class_id)
    file_list = os.listdir(folder_path)
    unloadable_images = []
    for filename in file_list:
        file_path =  os.path.join(folder_path, filename)
        try: 
            img = Image.open(file_path)
            img.convert('RGB')
        except (IOError, OSError, UnidentifiedImageError, Image.DecompressionBombError, ValueError) as e:
            print('Will remove:', file_path)
            unloadable_images.append(filename)
    
    for filename in unloadable_images:
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
        print(f"Removed unloadable image: {filename}")
    
def organize_images_worker(args):
    # Save Images after removing corrupted data
    class_id, root_folder = args
    folder_path = os.path.join(root_folder, class_id)
    file_list = sorted(os.listdir(folder_path), key=sorting_key)
    for i, image in enumerate(file_list):
        im = Image.open(os.path.join(folder_path, image))
        format = 'JPEG'
        op_path = str(i) + '.JPEG'
        rgb_img = im.convert('RGB')
        rgb_img.save(os.path.join(folder_path, op_path), format)
        os.remove(os.path.join(folder_path, image))
    print(f'organized after cleanup - {class_id}')

def clean_images(root_folder: str, num_classes: int = 1000):
    worker_args = [(str(i), root_folder) for i in range(num_classes)]
    num_processes = min(len(worker_args), 32)
    start = time.time()
    with Pool(processes=num_processes) as pool:
        pool.map(clean_images_worker, worker_args)
    
    with Pool(processes=num_processes) as pool:
        pool.map(organize_images_worker, worker_args)
    print('Total time:', time.time() - start)

def sorting_key(filename):
    # Split the filename (assuming '_' as the delimiter)
    parts = filename.split('.')
    
    # Convert the desired part to a number (e.g., the first part)
    return int(parts[0])

