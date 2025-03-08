# Math-Image-Search
## Description

This repo is for studies involving cross-modal math retrieval. Currently, I am testing the capabilities of Long-CLIP within this domain.

## Research questions

### Q1: Can Long-CLIP be fine-tuned for effective cross-modal math retrieval?

## Installation

Project intended for Python version 3.10.12. Please run the following to get started:

```
git clone --recurse-submodules https://github.com/aidan073/Math-Image-Search.git
pip install -r requirements.txt
```

- Obtain dataset (Will be provided soon)
- Obtain the longclip-L.pt starting checkpoint from: https://huggingface.co/BeichenZhang/LongCLIP-L/tree/main

## Usage

You can run the complete pipeline:
```
python -m src.pipeline --pipe complete (args TBD)
```

Or you can run each pipe individually in the following order:
```
python -m src.pipeline --pipe data -m <metadata_path> -i <images_path> -s <save_splits_path>.
```
```
python -m src.pipeline --pipe finetune -s <splits_path> -x <corrupted_files_path> -c <checkpoint_input_path> -o <checkpoint_output_path> [--distributed]
```
```
python -m src.pipeline --pipe evaluate (args TBD)
```


