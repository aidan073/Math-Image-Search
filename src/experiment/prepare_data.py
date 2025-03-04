from submodules.Long_CLIP.model import longclip

import csv
import os
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MathDataset(Dataset):
    def __init__(self, metadata:str, transform=None):
        self.transform = transform
        self.metadata = metadata

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.metadata[idx][2]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        caption = self.metadata[idx][1]
        text = longclip.tokenize(caption, truncate=True) # Tokenize the caption

        return image, text.squeeze(0) # Remove the extra dimension

def validate_image(image_path, ignore_exception:bool)->bool:
    absolute_path = os.path.abspath(image_path)
    if os.path.exists(absolute_path):
        try:
            image = Image.open(absolute_path)
            image.verify()
        except Exception:
            print(f"Image {absolute_path} is corrupted")
            if not ignore_exception:
                raise UnidentifiedImageError()
            return False
    else:
        print(f"Image {absolute_path} does not exist")
        if not ignore_exception:
            raise FileNotFoundError()
        return False
    return True
    
def validate_entry(entry, ignore_exception:bool)->bool:
    return True
  
def integrity(metadata, ignore_exception=False)->bool:
    """
    Verifys the integrity of the metadata file and the images, ensuring that images exist and the images are not corrupted

    Args:
        ignore_exception: if set to True, the program execution will continue and return False when a corrupted file is found (instead of raising an exception)

    Returns: 
        True if file integrity is assured, False (or raises exception) otherwise
    """
    for item in tqdm(metadata, desc="Verifying integrity of each sample"):
        valid_image = validate_image(item[2], ignore_exception)
        valid_entry = validate_entry(item[1], ignore_exception)
        if not valid_image and valid_entry:
            return False
    return True

def process_data(metadata_path:str, images_path:str, validate_data:bool=False)->tuple[MathDataset, MathDataset, list]:
    """
    metadata_path: path to .csv MSE dataset
    images_path: path to directory containing MSE images
    validata_data: if True, the entire MSE dataset will be checked for corrupted files or other invalidations (could be very slow).
    """
    metadata = []
    with open(metadata_path, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if(row[1][-2:] == '_0'):
                curr_row = [row[1], row[2], os.path.join(images_path, row[1]+".png")] # id, title, image_path
                metadata.append(curr_row)
    
    if(validate_data):
        integrity(metadata, ignore_exception=True)

    
        
