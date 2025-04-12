import csv
import vision_filtering
from src.experiment.prepare_data import process_mse, process_wikipedia

def merge(mse_tsv:str, mse_images:str, wiki_tsv:str, wiki_images:str, dataset_output_path:str=None, missing_output_path:str=None):
    """
    Merge the MSE and Wikipedia datasets.
    """
    mse_data, mse_missing = process_mse(mse_tsv, mse_images, True)
    wiki_data, wiki_missing = process_wikipedia(wiki_tsv, wiki_images, True)
    merged_data = mse_data + wiki_data
    merged_missing = mse_missing + wiki_missing

    if dataset_output_path:
        with open(dataset_output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(merged_data)

    if missing_output_path:
        with open(missing_output_path, 'w', encoding='utf-8') as f:
            f.writelines(merged_missing)

    return merged_missing, merged_data

def finalize(merged_dataset_or_path, merged_missing_or_path, final_output_path):
    """
    Do any processing on the merged dataset. For now, just filtering.
    """
    vision_filtering.filter(merged_data, merged_missing)

    

    
