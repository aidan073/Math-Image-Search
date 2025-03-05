from submodules.Long_CLIP.model import longclip

import csv
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

def validate_image(image_path, ignore_exception:bool, output_path:str)->bool:
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
    
    if not integral_image:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(os.path.basename(image_path)[0:-4] + '\n') # basename -png to get the original id
    return integral_image
    
def validate_entry(entry, ignore_exception:bool)->bool:
    return True
  
def integrity(metadata, output_path:str='missing_or_corrupted.txt', ignore_exception:bool=False)->bool:
    """
    Verifys the integrity of the metadata file and the images, ensuring that images exist and the images are not corrupted

    Args:
        metadata: metadata iterable
        output_path: .txt file path to save missing/corrupted image names in
        ignore_exception: if set to True, the program execution will continue and return False when a corrupted file is found (instead of raising an exception)

    Returns: 
        True if file integrity is assured, False (or raises exception) otherwise
    """
    if os.path.exists(output_path):
        print(f"Output missing/corrupted images.txt path: {output_path} already exists. Please delete it or provide a different path.")
        raise FileExistsError()
    
    integral_data = True
    for item in tqdm(metadata, desc="Verifying integrity of each sample"):
        valid_image = validate_image(item[2], ignore_exception, output_path)
        valid_entry = validate_entry(item[1], ignore_exception)
        if not valid_image and valid_entry:
            integral_data = False
    return integral_data

def process_data(metadata_path:str, images_path:str, validate_data:bool=False)->tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    metadata_path: path to .csv MSE dataset
    images_path: path to directory containing MSE images
    validata_data: if True, the entire MSE dataset will be checked for corrupted files or other invalidations (slow for large datasets).
    """
    metadata = []
    with open(metadata_path, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if(row[1][-2:] == '_0'): # only one image per text (might attempt to modify loss function to support multiple true pairs)
                curr_row = [row[1], row[2], os.path.join(images_path, row[1]+".png")] # id, title, image_path
                metadata.append(curr_row)
    
    # data validation
    if(validate_data):
        integrity(metadata, ignore_exception=True)

    # splitting
    np.random.shuffle(metadata)
    train_size = int(0.8 * len(metadata))
    val_size = int(0.1 * len(metadata))

    train_split = metadata[0: train_size]
    val_split = metadata[train_size: val_size]
    test_split = metadata[train_size + val_size:]

    return train_split, val_split, test_split

    


    
        
