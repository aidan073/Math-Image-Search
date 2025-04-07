# Math-Image-Search

## Description
This repo is for studies involving cross-modal math retrieval. Currently, I am testing the capabilities of Long-CLIP within this domain.

## Research Questions

### Q1: Can Long-CLIP be fine-tuned for effective cross-modal math retrieval?

## Installation

This project is developed for Python 3.10.12. To get started:

```
git clone --recurse-submodules https://github.com/aidan073/Math-Image-Search.git
pip install -r requirements.txt
```

### Prerequisites
- Dataset (coming soon)
- Long-CLIP checkpoint: Download `longclip-L.pt` from [HuggingFace](https://huggingface.co/BeichenZhang/LongCLIP-L/tree/main)

## Usage

### Complete Pipeline
Run the entire process (all pipelines) with the same hyper-parameters used for the research:

```
python -m src.pipeline complete [args TBD]
```

### Individual Pipelines

#### 1. Data Processing
Process the dataset:

```
python -m src.pipeline data -m <metadata_path> -i <images_path> -s <save_splits_path>
```

Example:
```
python -m src.pipeline data -m storage/MSE.tsv -i storage/MSE_Images -s splits
```

#### 2. Fine-tuning
Finetune the model on the dataset:

```
python -m src.pipeline finetune -s <splits_path> -x <corrupted_files_path> -c <checkpoint_input_path> -o <checkpoint_output_path> [--distributed -b <batch_size>]
```

Example:
```
python -m src.pipeline finetune --splits_path splits -c longclip-L.pt -x missing_or_corrupted.txt -d -b 70 -o finetuned-model
```

#### 3. Evaluation
Assess model performance:

```
python -m src.pipeline evaluate -c <checkpoint_input_path> -x <corrupted_files_path> -t <test_split_path> [--return_mean -e <results_output_path> -qi <qrel_input_path> -qo <qrel_output_path> -b <batch_size>]
```

Example:
```
python -m src.pipeline evaluate -c longclip-L.pt -x missing_or_corrupted.txt -t splits/test_split.npy -e search_test_1.json -qi qrel_test_1.json -b 1000
```