import src.construct_dataset.vision_filtering as vf
from src.experiment.prepare_data import process_mse, process_wikipedia

import csv

def merge(mse_tsv:str, mse_images:str, wiki_tsv:str, wiki_images:str, dataset_output_path:str=None, missing_output_path:str=None):
    """
    Merge the MSE and Wikipedia datasets.
    """
    mse_data, mse_missing = process_mse(mse_tsv, mse_images, validate_data=False)
    wiki_data, wiki_missing = process_wikipedia(wiki_tsv, wiki_images, validate_data=False)
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

    #TODO Make an argparser for filtering.
    MATH_OUTPUT_PATH = "Merged-Math-Dataset"
    SIM_OUTPUT_PATH = "Merged-Sim-Dataset"
    MATH_THRESHOLD = 0.5
    SIM_THRESHOLD = 0.5
    BATCH_SIZE = 100
    ENV_PATH = ".env"
    HF_TOKEN = None

    dataset, missing = merge(MSE_TSV_PATH, MSE_IMAGES_PATH, WIKI_TSV_PATH, WIKI_IMAGES_PATH, DATASET_OUTPUT_PATH, MISSING_OUTPUT_PATH)
    vf.filter(dataset, missing, MATH_OUTPUT_PATH, SIM_OUTPUT_PATH, MATH_THRESHOLD, SIM_THRESHOLD, BATCH_SIZE, ENV_PATH, HF_TOKEN)
    
