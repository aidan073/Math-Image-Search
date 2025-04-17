from src.experiment.prepare_data import process_mse, process_wikipedia

import os
import csv
import shutil
import random
from typing import Callable

"""
Construct small datasets for testing.
"""

def construct_mini(meta, image_dir:str, output_dir:str, n:int, image_key_idx:int, image_key_to_path:Callable=None):
    os.makedirs(output_dir, exist_ok=False)
    output_img_dir = os.path.join(output_dir, "Images")
    os.mkdir(output_img_dir)
    selected = random.sample(range(len(meta)), k=n)
    meta = [meta[i] for i in selected]
    # save 
    with open(os.path.join(output_dir, "metadata.tsv"), 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(meta)
    for sample in meta:
        key = image_key_to_path(sample[image_key_idx]) if image_key_to_path else sample[image_key_idx]
        src = os.path.join(image_dir, key)
        dst = os.path.join(output_img_dir, key)
        shutil.copy(src, dst)

if __name__ == '__main__':
    MSE_TSV_PATH = "/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE.tsv"
    MSE_IMAGES_PATH = "/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE_images/MathmaticaImages"
    WIKI_TSV_PATH = "/mnt/netstore1_home/aidan.bell/WikipediaMath/TIMath_Wiki.tsv"
    WIKI_IMAGES_PATH = "/mnt/netstore1_home/aidan.bell/WikipediaMath/Wiki_Images"
    mse, _ = process_mse(MSE_TSV_PATH, MSE_IMAGES_PATH, validate_data=False, has_header=True)
    wiki, _ = process_wikipedia(WIKI_TSV_PATH, WIKI_IMAGES_PATH, validate_data=False, has_header=False)
    construct_mini(mse, MSE_IMAGES_PATH, "mini_mse", 100, 0, lambda x: x + '.png')
    construct_mini(wiki, WIKI_IMAGES_PATH, "mini_wiki", 100, 0)
    

    
