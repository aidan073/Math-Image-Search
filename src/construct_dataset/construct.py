import src.construct_dataset.vision_filtering as vf
from src.experiment.prepare_data import process_mse, process_wikipedia, process_generic

import os
import csv
import shutil
from tqdm import tqdm

"""
This script was used to create the merged and filtered MSE/WIKI dataset.
"""

def merge(mse_tsv:str, mse_images:str, wiki_tsv:str, wiki_images:str, dataset_output_path:str=None, missing_output_path:str=None, remove_corrupted:bool=False):
    """
    Merge the MSE and Wikipedia datasets.
    """
    mse_data, mse_missing = process_mse(mse_tsv, mse_images, validate_data=True, has_header=True, remove_corrupted=remove_corrupted)
    wiki_data, wiki_missing = process_wikipedia(wiki_tsv, wiki_images, validate_data=True, has_header=False, remove_corrupted=remove_corrupted)
    merged_data = mse_data + wiki_data
    merged_missing = mse_missing + wiki_missing

    if dataset_output_path:
        with open(dataset_output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(merged_data)

    if missing_output_path:
        with open(missing_output_path, 'w', encoding='utf-8') as f:
            f.writelines(merged_missing)

    return merged_data, merged_missing

def finalize(true_math_samples:list[str], true_sim_samples:list[str], output_path:str=None):
    """
    Take the seperate filtered datasets, and combine into one dataset which meets all criteria. If output path provided, gathers those images and saves everything to that path.
    """
    if output_path:
        if os.path.exists(output_path):
            response = input(f"Designated output file '{output_path}' already exists. Do you want to override it? (y/N): ").strip().lower()
            if response == "y" or response == "yes":
                shutil.rmtree(output_path)
                os.mkdir(output_path)
            else:
                raise FileExistsError(f"File '{output_path}' already exists. Aborting operation.")
        else:
            os.mkdir(output_path)
        images_path = os.path.join(output_path, "Images")
        os.mkdir(images_path)

    math_ids = {sample[0] for sample in true_math_samples}
    sim_ids = {sample[0] for sample in true_sim_samples}
    
    common_ids = math_ids & sim_ids
    id_to_sample = {}
    for sample in true_math_samples + true_sim_samples:
        sample_id = sample[0]
        if sample_id in common_ids:
            id_to_sample[sample_id] = sample

    final = list(id_to_sample.values())
    if output_path:
        for sample in tqdm(final, total=len(final), desc=f"Copying images to {images_path}"):
            shutil.copy(sample[2], images_path)
        with open(os.path.join(output_path, "Meta.tsv"), 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(final)
    return final

if __name__ == "__main__":

    # get merged and non-corrupted dataset 
    # MSE_TSV = "/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE.tsv"
    # MSE_IMAGES = "/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE_images/MathmaticaImages"
    # WIKI_TSV = "/mnt/netstore1_home/aidan.bell/WikipediaMath/TIMath_Wiki.tsv"
    # WIKI_IMAGES = "/mnt/netstore1_home/aidan.bell/WikipediaMath/Wiki_Images"
    # dataset, _ = merge(MSE_TSV, MSE_IMAGES, WIKI_TSV, WIKI_IMAGES, "merged.tsv")

    # filter
    MSE_TSV_PATH = "splits/train_split.tsv"
    OUTPUT_PATH = "Math-0.7-Train"
    THRESHOLD = 0.7
    ENV_PATH = ".env"
    HF_TOKEN = None
    MATH_PROMPT = "Text: {text}\n\nDoes the image and text content relate to math? Respond with only the number: 1 if yes, 0 if no. Do not include any explanation or words. Just output 1 or 0."
    # # SIM_PROMPT = "Text: {text}\n\nAre the image and text related? Respond with 1 if yes, or 0 for no. Output only the number and no extra text."
    # SIM_PROMPT = "Text: {text}\n\nAre the image and text related? Respond with only the number: 1 if yes, 0 if no. Do not include any explanation or words. Just output 1 or 0."

    dataset, missing = process_generic(MSE_TSV_PATH, validate_data=True, has_header=False)
    true_math_samples = vf.filter(dataset, MATH_PROMPT, None, OUTPUT_PATH, THRESHOLD, ENV_PATH, HF_TOKEN, save_every=10000, save_every_dir="fallback")
    # true_sim_samples = vf.filter(dataset, SIM_PROMPT, None, OUTPUT_PATH, THRESHOLD, ENV_PATH, HF_TOKEN)
    # final_dataset = finalize(true_math_samples, true_sim_samples, FINAL_OUTPUT_PATH)

    print(f"Original Dataset Size: {len(dataset)}")
    print(f"Final Dataset Size: {len(true_math_samples)}")
    print(f"Missing/Corrupted: {len(missing)}")
    # # print(f"Filtered out by similarity: {len(dataset) - (len(true_sim_samples) + len(missing))}")
    print(f"Filtered out by math relation: {len(dataset) - (len(true_math_samples) + len(missing))}")


    
