from Long_CLIP.model import longclip

import os
import csv
import random
import numpy as np
from tqdm import tqdm
from typing import Union
from PIL import Image, UnidentifiedImageError

def validate_image(image_path, ignore_exception:bool)->bool:
    integral_image = True
    absolute_path = os.path.abspath(image_path)
    if os.path.exists(absolute_path):
        try:
            image = Image.open(absolute_path)
            image.verify()
        except Exception:
            print(f"Image {absolute_path} is corrupted")
            if not ignore_exception:
                raise UnidentifiedImageError()
            integral_image = False
    else:
        print(f"Image {absolute_path} does not exist")
        if not ignore_exception:
            raise FileNotFoundError()
        integral_image = False 

    return integral_image
    
def validate_entry(entry, ignore_exception:bool)->bool:
    return True
  
def integrity(metadata, output_path:str=None, ignore_exception:bool=False, remove_corrupted:bool=False)->Union[bool,list[str]]:
    """
    Verifys the integrity of the metadata file and the images, ensuring that images exist and the images are not corrupted.

    Args:
        metadata: Metadata iterable. Should be list/tuple of format: [[id, title, image_path], ...]
        output_path: .txt file path to save missing/corrupted image ids in.
        ignore_exception (default=False): If set to True, the program execution will continue and return False when a corrupted file is found (instead of raising an exception).
        remove_corrupted (default=False): Corrupted files will be removed from dataset.

    Returns: 
        integral_data: If the entire dataset is integeral or not.
        missing: A list containing all missing ids.
    """
    if output_path and os.path.exists(output_path):
        print(f"Output missing/corrupted images.txt path: {output_path} already exists. Please delete it or provide a different path.")
        raise FileExistsError()
    
    missing = []
    integral_data = True
    for item in tqdm(metadata[:], desc="Verifying integrity of each sample"): # deep copy outer list with shallow copy of samples in case remove_corrupted
        valid_image = validate_image(item[2], ignore_exception)
        valid_entry = validate_entry(item[1], ignore_exception)
        if not (valid_image and valid_entry):
            integral_data = False
            missing.append(item[0])
            if remove_corrupted:
                metadata.remove(item)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(missing)

    return integral_data, missing

def process_generic(metadata_path:str, missing_output_path:str=None, validate_data:bool=False, has_header:bool=True, remove_corrupted:bool=False)->tuple[list[list[str]], list[str]]:
    """
    Process a generic dataset. Must have a .tsv file of the format -> id, text, images_path.

    Args:
        metadata_path: path to .tsv data.
        missing_output_path: .txt file to save ids which are missing or corrupted.
        validate_data: if True, the entire dataset will be checked for corrupted files or other invalidations (slow for large datasets).
        has_header (default=False): Will skip the first row (header) if true.
        remove_corrupted (default=False): Corrupted samples will be removed from the dataset.

    Returns:
        metadata: list/tuple of format: [[id, text, image_path], ...]
        missing: A list containing all missing ids.
    """
    metadata = [] # [[id, text, image_path], ...]
    with open(metadata_path, "r", encoding='utf-8') as f:
        if has_header: 
            next(reader, None)
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            curr_row = [row[0], row[1], row[2]] # id, text, image_path
            metadata.append(curr_row)
    
    # data validation
    missing = []
    if(validate_data):
        _, missing = integrity(metadata, output_path=missing_output_path, ignore_exception=True, remove_corrupted=remove_corrupted)

    return metadata, missing

def process_mse(metadata_path:str, images_path:str, missing_output_path:str=None, validate_data:bool=False, has_header:bool=True, remove_corrupted:bool=False)->tuple[list[list[str]], list[str]]:
    """
    Args:
        metadata_path: path to .tsv MSE data.
        images_path: path to directory containing MSE dataset images.
        missing_output_path: .txt file to save ids which are missing or corrupted.
        validate_data: if True, the entire MSE dataset will be checked for corrupted files or other invalidations (slow for large datasets).
        has_header (default=True): Will skip the first row (header) if true.
        remove_corrupted (default=False): Corrupted samples will be removed from the dataset.

    Returns:
        metadata: list/tuple of format [[id, text, image_path], ...].
        missing: A list containing all missing ids.
    """
    metadata = [] # [[id, text, image_path], ...]
    with open(metadata_path, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        if has_header: 
            next(reader, None)
        for row in reader:
            if(row[1][-2:] == '_0'): # only one image per text (might attempt to modify loss function to support multiple true pairs)
                curr_row = [row[1], row[2], os.path.join(images_path, row[1]+".png")] # id, text, image_path
                metadata.append(curr_row)
    
    # data validation
    missing = []
    if(validate_data):
        _, missing = integrity(metadata, output_path=missing_output_path, ignore_exception=True, remove_corrupted=remove_corrupted)

    return metadata, missing

def process_wikipedia(metadata_path:str, images_path:str, missing_output_path:str=None, validate_data:bool=False, has_header:bool=False, remove_corrupted:bool=False)->tuple[list[list[str]], list[str]]:
    """
    Args:
        metadata_path: path to .tsv Wikipedia data.
        images_path: path to directory containing Wikipedia dataset images.
        missing_output_path: .txt file to save ids which are missing or corrupted.
        validate_data: if True, the entire Wikipedia dataset will be checked for corrupted files or other invalidations (slow for large datasets).
        has_header (default=False): Will skip the first row (header) if true.
        remove_corrupted (default=False): Corrupted samples will be removed from the dataset.

    Returns:
        metadata: list/tuple of format: [[id, text, image_path], ...]
        missing: A list containing all missing ids.
    """
    metadata = [] # [[id, text, image_path], ...]
    with open(metadata_path, "r", encoding='utf-8') as f:
        if has_header: 
            next(reader, None)
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            curr_row = [row[0], row[2], os.path.join(images_path, row[0])] # id, text, image_path
            metadata.append(curr_row)
    
    # data validation
    missing = []
    if(validate_data):
        _, missing = integrity(metadata, output_path=missing_output_path, ignore_exception=True, remove_corrupted=remove_corrupted)

    return metadata, missing

def create_splits(data):
    random.shuffle(data)
    n = len(data)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    return train, test, val





    
