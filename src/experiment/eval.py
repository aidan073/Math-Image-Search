import json
import numpy as np
from ranx import Qrels, evaluate
from src import search

def evaluate_model(checkpoint_path:str, test_split_path:str, missing_or_corrupted:str, metrics:list, qrel_input_path:str=None, eval_output_path:str=None, qrel_output_path:str=None, return_scores:bool=False, test_split_splits:int=1)->None:
    """
    Function to evaluate the performance of a Long-CLIP checkpoint.

    Args:
        checkpoint_path: Path to Long-CLIP .pt checkpoint that you want to evaluate.
        test_split_path: Path to test split .npy file (generated by prepare_data.py).
        missing_or_corrupted: Path to .txt file containing missing/corrupted samples (generated by prepare_data.py).
        metrics: Ranx metrics to use for evaluation.
        qrel_input_path (optional): Path to Qrel .json file to use for evaluation. If None, a new Qrel will be created.
        eval_output_path (optional): Path .json to save evaluation results in. If None, results will simply be printed.
        qrel_output_path (optional): Path .json to save Qrel in.
        return_scores (Default=False): If False, returns mean of each metric. If True, returns dict with metric for each query.
        test_split_splits (optional): Number of splits to make of the test_split. Defaults to 1, but if the test_split is very large, then splitting it up may be necessary to avoid memory errors.
    """

    # construct missing or corrupted set
    mc_set = set()
    if missing_or_corrupted:
        with open(missing_or_corrupted, "r", encoding='utf-8') as f:
            mc_set.update(line.strip() for line in f)

    # construct run and qrel
    run = search.full_search(checkpoint_path, test_split_path, None, mc_set, test_split_splits)
    if not qrel_input_path:
        qrel = _construct_qrel(test_split_path, mc_set, qrel_output_path)
    else:
        with open(qrel_input_path, "r", encoding='utf-8') as f:
            qrel = json.load(f)

    # evaluate
    results = evaluate(qrel, run, metrics, return_mean=not return_scores)
    if eval_output_path:
        with open(eval_output_path, "w") as f:
            json.dump(results, f)
    else:
        print(results)

def _construct_qrel(test_split_path:str, mc_set:set[str], qrel_output_path:str=None)->Qrels:
    qrel_dict = {}
    metadata = np.load(test_split_path)
    for sample in metadata:
        if sample[0] in mc_set: # skip missing/corrupted
            continue
        qrel_dict[sample[0]] = {sample[0]: 1}
    qrel = Qrels(qrel_dict, "retrieval_qrel")
    if qrel_output_path:
        qrel.save(qrel_output_path)
    return qrel