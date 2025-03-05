from submodules.Long_CLIP.model import longclip
from src.experiment.prepare_data import process_data

import os
import torch
import argparse
import numpy as np
from PIL import Image

# will perform the following:
# prepare dataset
# finetune
# evaluate

parser = argparse.ArgumentParser(description="Full experiment pipeline. Data processing, finetuning, and evaluation.")
parser.add_argument('--pipe', '-p', type=str, choices=['data', 'finetune', 'evaluate', 'complete'], help="Which part of pipeline to run. Options are: data, finetune, evaluate, complete")
parser.add_argument('--metadata_path', '-m', type=str, help="Path to metadata tsv file.")
parser.add_argument('--images_path', '-i', type=str, help="Path to images folder.")
parser.add_argument('--save_splits_path', '-s', type=str, help="Folder path to save test/train/val splits.")
args = parser.parse_args()

match args.pipe.lower():
    case 'data':
        if not (args.metadata_path and args.images_path and args.save_splits_path):
            raise argparse.ArgumentError(None, "Missing required arguments for data pipe. Usage: --pipe data -m <metadata_path> -i <images_path> -s <save_splits_path>.")
        train_split, val_split, test_split = process_data(args.metadata_path, args.images_path, validate_data=True)
        if not os.path.exists(args.save_splits_path):
            os.mkdir(args.save_splits_path)
        with open('train_split.npy', 'wb') as f1:
            np.save(f1, train_split)
        with open('val_split.npy', 'wb') as f2:
            np.save(f2, train_split)
        with open('test_split.npy', 'wb') as f3:
            np.save(f3, train_split)

    case 'finetune':
        pass

    case 'evaluate':
        pass
    
    case 'complete':
        if not (args.metadata_path and args.images_path):
            raise argparse.ArgumentError(None, "Missing required arguments for complete pipe. Usage: --pipe complete -m <metadata_path> -i <images_path>")
        train_split, val_split, test_split = process_data(args.metadata_path, args.images_path, validate_data=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

# text = longclip.tokenize(["A man is crossing the street with a red car parked nearby.", "A man is driving a car in an urban scene."]).to(device)
# image = preprocess(Image.open("./img/demo.png")).unsqueeze(0).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image = image_features @ text_features.T
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs) 