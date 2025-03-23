import json
import torch
import numpy as np
from ranx import Qrels, Run, evaluate
from src import search

def evaluate(checkpoint_path:str, test_split_path:str, metrics:list, return_mean:bool=True, output_path:str=None)->None:
    """
    Function to evaluate the performance of a Long-CLIP checkpoint.

    Args:
        checkpoint_path: Path to Long-CLIP .pt checkpoint that you want to evaluate.
        test_split_path: Path to test split .npy file (generated by prepare_data.py).
        metrics: Ranx metrics to use for evaluation.
        return_mean: If True, returns mean of each metric. If False, returns dict with metric for each query (Default=True).
        output_path: Path .json to save evaluation results in. If None, results will simply be printed.
    """
    run = search.longclip_search(checkpoint_path, test_split_path)

    results = evaluate(qrel, run, metrics, return_mean=return_mean)
    if not return_mean:
        # TODO: CHANGE THIS TO .json FOR CONVENIENCE
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"] + list(run.keys()))
            for metric in metrics:
                writer.writerow([metric, results[metric].tolist()])
    else:
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=4)

def _build_qrel(test_split_path:str, output_path:str=None)->Qrels:
    true_captions = load_test(test_split_path)
    qrel_dict = {}
    for id in true_captions.keys():
        qrel_dict[id] = {id: 1}
    qrel = Qrels(qrel_dict, "t2i_retrieval")
    if output_path:
        qrel.save(output_path)
    return qrel