from src.experiment.prepare_data import process_generic, process_mse, process_wikipedia, create_splits
import src.construct_dataset.vision_filtering as vf
from src.experiment.finetuner import finetune
from src.experiment.eval import evaluate_model

import os
import csv
import torch
import random
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

def get_args():
    parser = argparse.ArgumentParser(description="Full experiment pipeline. Data processing, filtering, finetuning, and evaluation.")
    subparsers = parser.add_subparsers(dest='pipe_type', required=True)

    # data parser
    data_parser = subparsers.add_parser('data', help='Data processing.')
    data_parser.add_argument('--metadata_path', '-m', type=str, help="Path to metadata tsv file.")
    data_parser.add_argument('--images_path', '-i', type=str, help="Path to images folder.")
    data_parser.add_argument('--dataset_type', '-t', type=str, required=True, choices=["generic", "mse", "wiki"], help="Which dataset are you processing? 'generic', 'mse', or 'wiki'. Generic must have .tsv of the form: id, text, image_path. Images directory is not needed for generic.")
    data_parser.add_argument('--has_header', '-h', action="store_true", help="If your dataset has a header, this flag will make sure to skip it to prevent errors.")
    data_parser.add_argument('--splits_path', '-s', type=str, help="Folder path to save test/train/val splits and missing/corrupted .txt in.")

    # filter parser
    filter_parser = subparsers.add_parser('filter', help="Use llama vision to filter data. You may provide both mse and wikipedia, or just one of them.")
    filter_parser.add_argument('--metadata_path', '-m', type=str, help="Path to .tsv metadata file. Each row must be: id, text, image_path.")
    filter_parser.add_argument('--output_path', '-o', type=str, help=".tsv output path to save filtered metadata in.")
    filter_parser.add_argument('--prompt', '-p', type=str, help="Prompt must ask for model to return a '1' or a '0' based on some criteria. It also must include '{text}' where the text should go.")
    filter_parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Value between 0-1, specifying the confidence needed by the model to classify a sample as 1 (based on your prompt).")
    filter_parser.add_argument('--corrupted', '-x', type=str, help="(Required if any corrupted samples) Path to the .txt file that contains missing or corrupted images (the .txt is generated by data pipe).")
    filter_parser.add_argument('--env_path', '-e', type=str, help="Path to .env containing 'hf_token' field, which is a hugging face token with llama3.2 access.")
    filter_parser.add_argument('--hf_token', '-h', type=str, help="Provide this or env_path. Hugging face token with llama3.2 access.")

    # finetune parser
    finetune_parser = subparsers.add_parser('finetune', help='Finetune Long-CLIP.')
    finetune_parser.add_argument('--splits_path', '-s', type=str, help="Folder to load the splits from.")
    finetune_parser.add_argument('--c_input_path', '-c', type=str, help="Path to Long-CLIP checkpoint that you wish to finetune.")
    finetune_parser.add_argument('--c_output_path', '-o', type=str, help="Folder path to save the checkpoints and logs generated by finetune pipe.")
    finetune_parser.add_argument('--distributed', '-d', action="store_true", help="Multi-gpu finetuning. This will use all available GPUs to improve finetuning.")
    finetune_parser.add_argument('--batch_size', '-b', type=int, default=30, help="Finetuning batch_size, defaults to 30.")
    finetune_parser.add_argument('--corrupted', '-x', type=str, help="(Required if any corrupted samples) Path to the .txt file that contains missing or corrupted images (the .txt is generated by data pipe).")

    # evaluate parser
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model performance.')
    evaluate_parser.add_argument('--c_input_path', '-c', type=str, help="Path to Long-CLIP checkpoint that you wish to evaluate.")
    evaluate_parser.add_argument('--test_split_path', '-t', type=str, help="Path to test split .npy file (generated by data pipe).")
    evaluate_parser.add_argument('--return_scores', '-r', action='store_true', help="If False, returns mean of each metric. If True, returns dict with metric for each query.")
    evaluate_parser.add_argument('--corrupted', '-x', type=str, help="(Required if any corrupted samples) Path to the .txt file that contains missing or corrupted images (the .txt is generated by data pipe).")
    evaluate_parser.add_argument('--qrel_input_path', '-qi', type=str, help="(Optional) Qrel .json path to load in. If no path provided, then a new Qrel will be constructed.")
    evaluate_parser.add_argument('--qrel_output_path', '-qo', type=str, help="(Optional) .json path to save Qrel to.")
    evaluate_parser.add_argument('--eval_output_path', '-e', type=str, help="(Optional) .json path to save results in. If no path provided, results will simply be printed.")
    evaluate_parser.add_argument('--batch_size', '-b', type=int, default=100, help="Max number of samples per batch during encoding. Make this larger to speed up encoding, or smaller to prevent out of memory errors.")

    # complete parser (TBD)
    complete_parser = subparsers.add_parser('complete', help='Run complete experiment with pre-determined hyper-parameters.')

    return parser.parse_args()

def main():
    args = get_args()

    match args.pipe_type.lower():
        case 'data':
            if not args.images_path and args.dataset_type != "generic":
                raise argparse.ArgumentError(None, "Missing images_path arg.")
            if not (args.metadata_path and args.splits_path):
                raise argparse.ArgumentError(None, "Missing required arguments for data pipe. Usage: data -m <metadata_path> -s <save_splits_path>.")
            match args.dataset_type.lower():
                case 'generic':
                    dataset, missing = process_generic(args.metadata_path, validate_data=True, has_header=args.has_header)
                case 'mse':
                    dataset, missing = process_mse(args.metadata_path, args.images_path, validate_data=True)
                case 'wiki':
                    dataset, missing = process_mse(args.metadata_path, args.images_path, validate_data=True)

            train, test, val = create_splits(dataset)
            if not os.path.exists(args.splits_path):
                os.mkdir(args.splits_path)
            with open(os.path.join(args.splits_path, 'train_split.tsv'), 'w', encoding='utf-8') as f1:
                writer = csv.writer(f1, delimiter='\t')
                writer.writerows(train)
            with open(os.path.join(args.splits_path, 'val_split.tsv'), 'w', encoding='utf-8') as f2:
                writer = csv.writer(f2, delimiter='\t')
                writer.writerows(val)
            with open(os.path.join(args.splits_path, 'test_split.tsv'), 'w', encoding='utf-8') as f3:
                writer = csv.writer(f3, delimiter='\t')
                writer.writerows(test)
            with open(os.path.join(args.splits_path, 'missing_or_corrupted.txt'), 'w', encoding='utf-8') as f4:
                f4.writelines(missing)
        
        case 'filter':
            if not (args.metadata_path and args.output_path and args.prompt and (args.env_path or args.hf_token)):
                raise argparse.ArgumentError(None, f"Missing required arguments for filter pipe. Usage: filter -m <metadata_path> -o <output_path> -p <prompt> -x <corrupted_path> [-e env_path | -h hf_token]")
            vf.filter(args.metadata_path, args.prompt, args.corrupted, args.output_path, args.threshold, args.env_path, args.hf_token)

        case 'finetune':
            if not (args.splits_path and args.c_input_path and args.c_output_path):
                raise argparse.ArgumentError(None, f"Missing required arguments for finetune pipe. Usage: finetune -s <splits_path> -c <checkpoint_input_path> -o <checkpoint_output_path>")
            if os.path.exists(args.c_output_path):
                raise FileExistsError(f"Designated output folder '{args.c_output_path}' already exists. Please delete it or provide a different output folder name for c_output_path.")
            os.mkdir(args.c_output_path)
            if args.distributed:
                os.environ["MASTER_ADDR"] = "localhost"
                os.environ["MASTER_PORT"] = "12354"
                world_size = torch.cuda.device_count()
                assert world_size >= 2, f"Distributed requires at least 2 GPUs to run, but got {world_size}"
                try:
                    mp.spawn(finetune, args=(args.distributed, args.splits_path, args.corrupted, args.c_input_path, args.c_output_path, 25, args.batch_size, world_size), nprocs=world_size, join=True)
                except:
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    raise
            else:
                finetune(0, args.distributed, args.splits_path, args.corrupted, args.c_input_path, args.c_output_path, batch_size=args.batch_size)
            
        case 'evaluate':
            metrics = ['precision@1', 'mrr'] # Can modify desired metrics here. Reference Ranx library to get list of valid metric names.
            if not (args.c_input_path and args.test_split_path):
                raise argparse.ArgumentError(None, f"Missing required arguments for evaluate pipe. Usage: evaluate -c <Long-CLIP checkpoint path> -t <test_split_path>")
            evaluate_model(args.c_input_path, args.test_split_path, args.corrupted, metrics, args.qrel_input_path, args.eval_output_path, args.qrel_output_path, args.return_scores, args.batch_size)

        case 'complete':
            if not (args.metadata_path and args.images_path):
                raise argparse.ArgumentError(None, "Missing required arguments for complete pipe. Usage: complete -m <metadata_path> -i <images_path>")
            train_split, val_split, test_split = process_mse(args.metadata_path, args.images_path, validate_data=True)

if __name__ == "__main__":
    main() # Wrapped code in main function so that spawned processes don't re-run everything.