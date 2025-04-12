import vision_filtering
from src.experiment.prepare_data import process_mse, process_wikipedia

def merge(mse_tsv:str, mse_images:str, wiki_tsv:str, wiki_images:str, output_path:str):
    """
    Merge the MSE and Wikipedia datasets.
    """
    mse_data, mse_missing = process_mse(mse_tsv, mse_images, False) #TODO: Change to True before final run
    wiki_data, wiki_missing = process_wikipedia(wiki_tsv, wiki_images, False)
    merged_missing = mse_missing + wiki_missing
    merged_data = mse_data + wiki_data
    vision_filtering.filter(merged_data, merged_missing)

    

    
