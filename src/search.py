from Long_CLIP.model.longclip import tokenize, load

import os
import torch
import argparse
import torchvision
import numpy as np
from ranx import Run
from tqdm import tqdm
from PIL import Image
from typing import Union, Callable

def quick_search_parser():
    parser = argparse.ArgumentParser(description="Run Long-CLIP quick search")
    
    # required
    parser.add_argument("checkpoint_path", type=str, help="Path to Long-CLIP checkpoint file.")
    parser.add_argument("--texts_path", type=str, help="Path to .txt file with texts, one per line.") # this or texts
    parser.add_argument("--texts", type=str, nargs="+", help="Single text or list of texts.") # this or texts_path
    parser.add_argument("--images_path", type=str, help="Path to directory containing image files.")

    # optional / defaults
    parser.add_argument("--search_type", type=str, choices=["t2i", "i2t"], default="t2i", help="Search direction: 't2i' or 'i2t'.")
    parser.add_argument("--output_path", type=str, help="Optional output path to save results.")
    parser.add_argument("--max_batch_size", type=int, default=50, help="Max number of samples per batch.")
    parser.add_argument("--top_n", type=int, default=100, help="Top-N results to return.")

    return parser.parse_args()

def full_search_parser():
    parser = argparse.ArgumentParser(description="Run Long-CLIP fu")
    parser.add_argument('--results_path', '-r', type=str, help="Path to save search results in.")
    parser.add_argument('--c_input_path', '-c', type=str, help="Long-CLIP checkpoint path to finetune.")
    parser.add_argument('--test_split_path', '-t', type=str, help="Path to test split .npy file (generated by prepare_data).")
    parser.add_argument('--missing_or_corrupted', '-x', type=str, default=None, help="Path to missing or corrupted txt file (generated by prepare_data.py).")
    parser.add_argument('--test_split_splits', '-s', type=int, default=1, help="Number of splits to make of the test_split. Defaults to 1, but if the test_split is very large, then splitting it up may be necessary to avoid memory errors.") #TODO just make this max_batch_size. This is stupid.

def quick_search(checkpoint_path:str, search_type:str='t2i', texts_path:str=None, texts:Union[str, list[str]]=None, images_path:str=None, images:Union[Image.Image, list[Image.Image]]=None, output_path:str=None, max_batch_size:int=50, top_n:int=100)->dict[int, list[int]]:
    """
    Perform a quick search using Long-CLIP.

    Args:
        checkpoint_path: Path to Long-CLIP checkpoint file.
        search_type (Default='t2i'): For text to image search, this should be 't2i'. For image to text search, this should be 'i2t'. 
        texts_path (optional): Path to .txt file where each newline is a text. Either this or texts must be provided.
        texts (optional): A single text, or a list of texts. Either this or texts_path must be provided.
        images_path (optional): Path to directory containing the images to use. Either this or images must be provided.
        images (optional): A single image, or a list of images. Either this or images_path must be provided.
        output_path (optional): Path to save Run in.
        max_batch_size (Default=50): Max number of samples to encode at once. Make this lower to lower the memory usage, or higher to speed up the encoding.
        top_n (Default=100): How many results to return for each query.

    Returns:
        Dict with keys = query indices, values = list of retrieved document indices (from most relevant to least). These indices correspond to their index in the input texts/images. eg: {0: [42, 73, 99, ...], ...}
    """
    assert search_type == 't2i' or search_type == 'i2t'
    assert texts_path or texts
    assert images_path or images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Process input data
    if images:
        if isinstance(images, Image.Image): #TODO ensure Image.Image is the correct type
            images = [preprocess(images).to(device)]
    else:
        images = []
        for filename in os.listdir(images_path):
            full_path = os.path.join(images_path, filename)
            images.append(preprocess(Image.open(full_path)).to(device))

    if texts:
        if isinstance(texts, str):
            texts = [tokenize(texts, truncate=True).squeeze(0).to(device)]
    else:
        texts = []
        with open(texts_path, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(tokenize(line.strip(), truncate=True).squeeze(0).to(device))

    if len(texts>max_batch_size):
        _batchify(texts, max_batch_size)
    else:
        texts = [texts]
    if len(images>max_batch_size):
        _batchify(images, max_batch_size)
    else:
        images = [images]
                
    # Run search
    model, preprocess = load(checkpoint_path, device=device)
    model.eval()

    with torch.no_grad(): 
        text_embeddings_list = []
        image_embeddings_list = []
        zipped_input = zip(texts, images)
        for texts, images in tqdm(zipped_input, "Encoding the data.", total=len(zipped_input)):
            text_embeddings_list.append(model.encode_text(texts))
            image_embeddings_list.append(model.encode_image(images))

        text_embeddings = torch.cat(text_embeddings_list, dim=0)
        image_embeddings = torch.cat(image_embeddings_list, dim=0)
    
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True) # TODO: Long-CLIP authors don't norm during inference. Find out if there is a reason.
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    sim = text_embeddings @ image_embeddings.T # This is t2i sim
    results = {}
    if search_type == 'i2t':
        sim = sim.T
    for idx in range(sim.size(dim=0)):
        top_matching_indices = sim[idx, :].argsort(dim=0, descending=True)[:top_n]
        results[idx] = top_matching_indices.tolist()
            
    return results

def longclip_search(checkpoint_path:str, test_split_path:dict, output_path:str=None, missing_or_corrupted:str=None, test_split_splits:int=1)->Run:
    """
    Use Long-CLIP checkpoint for text to image retrieval. This function requires a test split with specific formatting (generated by experiment/prepare_data.py), and will return a Ranx Run which can be used for evaluation.
    If you are not reproducing the experiment, and instead just searching, I recommend using the quick_search() function instead.

    Args:
        checkpoint_path: Path to Long-CLIP checkpoint file.
        test_split_path: Path to test split json file.
        output_path (optional): Save run to this path.
        missing_or_corrupted (optional): Set of str ids that are corrupted, or Path to .txt file in which each row contains the sample id of a corrupted image.
        test_split_splits (optional): Number of splits to make of the test_split. Defaults to 1, but if the test_split is very large, then splitting it up may be necessary to avoid memory errors.

    Returns:
        Ranx Run
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load(checkpoint_path, device=device)
    model.eval()

    ids, texts, images = _load_test(test_split_path, device, tokenize, preprocess, missing_or_corrupted)

    with torch.no_grad():
        if test_split_splits == 1:
            texts_batches = texts.unsqueeze(0)
            images_batches = images.unsqueeze(0)
        else:
            split_size = len(texts) // test_split_splits
            texts_batches = torch.split(texts, split_size)
            images_batches = torch.split(images, split_size)
        
        text_embeddings_list = []
        image_embeddings_list = []
        zipped_input = zip(texts_batches, images_batches)
        for texts, images in tqdm(zipped_input, "Encoding batches", total=len(texts_batches)):
            text_embeddings_list.append(model.encode_text(texts))
            image_embeddings_list.append(model.encode_image(images))

        text_embeddings = torch.cat(text_embeddings_list, dim=0)
        image_embeddings = torch.cat(image_embeddings_list, dim=0)
    
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True) # TODO: Long-CLIP authors don't norm during inference. Find out if there is a reason.
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    t2i_sim = text_embeddings @ image_embeddings.T

    return _construct_run(ids, t2i_sim, "longclip_retrieval", output_path)

def _load_test(test_split_path:str, device:str, tokenize:Callable[[Union[str, list[str]]], torch.Tensor], preprocess:torchvision.transforms.Compose, missing_or_corrupted:Union[set[str], str])->tuple[list, list, list]:
    
    mc_set = set()
    if missing_or_corrupted:
        if isinstance(missing_or_corrupted, set):
            mc_set = missing_or_corrupted
        else:
            with open(missing_or_corrupted, "r", encoding='utf-8') as f:
                mc_set.update(line.strip() for line in f)
    
    test_split = np.load(test_split_path)
    ids = []
    texts = []
    images = []
    for sample in tqdm(test_split, "Loading test samples", total=len(test_split)):
        if sample[0] in mc_set: # If this image is corrupted, skip
            continue
        ids.append(sample[0])
        texts.append(tokenize(sample[1], truncate=True).squeeze(0).to(device))
        images.append(preprocess(Image.open(sample[2])).to(device))

    texts = torch.stack(texts)
    images = torch.stack(images)

    return ids, texts, images

def _construct_run(ids:list[str], similarities:list, run_name:str, output_path:str=None, top_n:int=100):
    run_dict = {}
    for idx in range(similarities.size(dim=0)):
        top_matching_indices = similarities[idx, :].argsort(dim=0, descending=True)[:top_n]
        values = similarities[idx, :][top_matching_indices]
        run_dict[ids[idx]] = {}
        for key_idx, value in zip(top_matching_indices, values):
            run_dict[ids[idx]][ids[key_idx]] = value.item()
    run = Run(run_dict, run_name)
    if output_path:
        run.save(output_path)
    return run

def _batchify(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

if __name__ == "__main__":
    args = parse_args()
    if not (args.results_path and args.c_input_path and args.test_split_path):
        raise argparse.ArgumentError(None, "Missing required arguments for searching. Usage: -r <path to save results in> -c <path to Long-CLIP checkpoint> -t <path to test split>.")

    if os.path.exists(args.results_path):
        response = input(f"Designated output file '{args.results_path}' already exists. Do you want to override it? (y/N): ").strip().lower()
        if response != "y" and response != "yes":
            raise FileExistsError(f"File '{args.results_path}' already exists. Aborting operation.")

    longclip_search(args.c_input_path, args.test_split_path, args.results_path, args.missing_or_corrupted, args.test_split_splits)
    