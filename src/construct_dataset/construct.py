import src.construct_dataset.vision_filtering as vf
from src.experiment.prepare_data import process_mse, process_wikipedia

import os
import csv

def merge(mse_tsv:str, mse_images:str, wiki_tsv:str, wiki_images:str, dataset_output_path:str=None, missing_output_path:str=None):
    """
    Merge the MSE and Wikipedia datasets.
    """
    mse_data, mse_missing = process_mse(mse_tsv, mse_images, validate_data=True)
    wiki_data, wiki_missing = process_wikipedia(wiki_tsv, wiki_images, validate_data=True)
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
    Take the seperate filtered datasets, and combine into one dataset which meets all criteria.
    """
    if output_path and os.path.exists(output_path):
        response = input(f"Designated output file '{output_path}' already exists. Do you want to override it? (y/N): ").strip().lower()
        if response != "y" and response != "yes":
            raise FileExistsError(f"File '{output_path}' already exists. Aborting operation.")

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
        with open(output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(final)
    return final

if __name__ == "__main__":
    # MSE_TSV_PATH = ""
    # MSE_IMAGES_PATH = ""
    # WIKI_TSV_PATH = ""
    # WIKI_IMAGES_PATH = ""
    # DATASET_OUTPUT_PATH = None
    # MISSING_OUTPUT_PATH = None

    # #TODO Make an argparser for filtering.
    # FINAL_DATA_OUTPUT_PATH = ""
    # MATH_THRESHOLD = 0.5
    # SIM_THRESHOLD = 0.5
    # ENV_PATH = ""
    # HF_TOKEN = None

    MSE_TSV_PATH = "/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE.tsv"
    MSE_IMAGES_PATH = "/mnt/netstore1_home/aidan.bell/MathStackExchange/MSE_images/MathmaticaImages"
    WIKI_TSV_PATH = "/mnt/netstore1_home/aidan.bell/WikipediaMath/TIMath_Wiki.tsv"
    WIKI_IMAGES_PATH = "/mnt/netstore1_home/aidan.bell/WikipediaMath/Wiki_Images"
    DATASET_OUTPUT_PATH = None
    MISSING_OUTPUT_PATH = None

    FINAL_OUTPUT_PATH = "Final-Dataset"
    #TODO Make an argparser for filtering.
    MATH_OUTPUT_PATH = "Merged-Math-Dataset"
    SIM_OUTPUT_PATH = "Merged-Sim-Dataset"
    MATH_THRESHOLD = 0.5
    SIM_THRESHOLD = 0.5
    BATCH_SIZE = 20
    ENV_PATH = ".env"
    HF_TOKEN = None

    dataset, missing = merge(MSE_TSV_PATH, MSE_IMAGES_PATH, WIKI_TSV_PATH, WIKI_IMAGES_PATH, DATASET_OUTPUT_PATH, MISSING_OUTPUT_PATH)
    true_math_samples, true_sim_samples = vf.filter(dataset, missing, MATH_OUTPUT_PATH, SIM_OUTPUT_PATH, MATH_THRESHOLD, SIM_THRESHOLD, BATCH_SIZE, ENV_PATH, HF_TOKEN)
    final_dataset = finalize(true_math_samples, true_sim_samples, FINAL_OUTPUT_PATH)

    print(f"Original Merged Dataset Size: {len(dataset)}")
    print(f"Final Dataset Size: {len(final_dataset)}")
    print(f"Missing/Corrupted: {len(missing)}")
    print(f"Filtered out by similarity: {len(dataset) - (len(true_sim_samples) + len(missing))}")
    print(f"Filtered out by math relation: {len(dataset) - (len(true_math_samples) + len(missing))}")
    
