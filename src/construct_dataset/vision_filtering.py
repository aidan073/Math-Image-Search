import os
import csv
import math
import torch
from tqdm import tqdm
from PIL import Image
from typing import Union
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

def filter(dataset_or_path:Union[str, list[str]], prompt:str, missing_or_path:Union[str, list[str]], output_path:str, threshold:int=0.5, env_path:str=None, hf_token:str=None, save_every:int=None, save_every_dir:str=None)->list[list[str]]:
    """
    Filter samples from datasets using Llama3.2 vision.
    
    Args:
        dataset_or_path: Path to dataset .tsv file, or dataset of format: [[id, title/text, image_path], ...]
        prompt: Classification prompt for the model. Prompt must ask for a 1 or a 0 label, and must contain a {text} var that will be filled by a formatter. e.g >>> 'Text: {text}\n\nDoes the image and text content relate to math? Respond with 1 if yes, or 0 for no. Output only the number and no extra text.'
        missing_or_path: Path to missing .txt file, or set with missing ids. These will be excluded from the returned datasets.
        output_path: .tsv file to save filtered dataset in.
        threshold (default=0.5): "confidence" required to classify sample as 1. Specifically, confidence = output of softmax(true_token) with denominator containing true_token and false_token.
        env_path (optional*): Path to .env file, which must contain "HF_TOKEN" field with hugging face llama access token as its value. 
        hf_token (optional*): Hugging face llama access token. You must provide this, or env_path.
        save_every (optional): If provided, the output dataset will be saved each time that many samples have been classified.
        save_every_dir (optional): Directory to save data if save_every is provided.

    Returns:
        true_samples: list where each item is a sample.
    """
    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    if output_path:
        if os.path.exists(output_path):
            raise FileExistsError(f"Designated output_path: {output_path} already exists. Please delete it or provide a different output_path.")
        os.makedirs(output_path)

    if save_every_dir:
        if os.path.exists(save_every_dir):
            raise FileExistsError(f"Designated save_every_dir: {save_every_dir} already exists. Please delete it or provide a different save_every_dir.")
        os.makedirs(save_every_dir)

    assert bool(save_every) == bool(save_every_dir)
    assert env_path or hf_token
    if(env_path):
        load_dotenv(env_path)
        hf_token = os.getenv("HF_TOKEN") 
    login(hf_token)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # quantization has a large negative effect on perceived classification accuracy
    # BNB_CONFIG = BitsAndBytesConfig(load_in_8bit=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_ID,
    #     quantization_config=BNB_CONFIG,
    #     device_map="auto",
    #     trust_remote_code=True,
    # )
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    results, no_missing_dataset = _run_filter(model, dataset_or_path, missing_or_path, prompt, threshold, processor, device, save_every, save_every_dir)
    true_samples = [no_missing_dataset[idx] for idx, classification in enumerate(results) if classification == True]

    if output_path:
        with open(os.path.join(output_path, "Meta.tsv"), 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(true_samples)

    return true_samples

def _run_filter(model, dataset_or_path, missing_or_path, prompt, threshold, processor, device, save_every, save_every_dir)->Union[list[str], list[str]]:
    """
    Convert a dataset into input ready batch. Also returns the original dataset with all missing/corrupted samples removed.
    """
    msg = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": None}
        ]}
    ]
    # load missing
    if isinstance(missing_or_path, str):
        with open(missing_or_path, 'r', encoding='utf-8') as f:
            missing = {missing_id.strip() for missing_id in f}
    elif missing_or_path is None:
        missing = set()
    else:
        missing = missing_or_path
    # load dataset
    if isinstance(dataset_or_path, str):
        with open(dataset_or_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t")
            dataset = [sample for sample in reader if sample[0] not in missing]
    else:
        dataset = [sample for sample in dataset_or_path if sample[0] not in missing]

    req_logit_diff = torch.log(torch.tensor(threshold / (1 - threshold))) # required logit difference to meet confidence threshold
    true_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("1")[0])
    false_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("0")[0])
    results = []
    since_last_save = 0
    for sample in tqdm(dataset, total=len(dataset), desc="Classifying Samples"):
        msg[0]["content"][1]["text"] = prompt.format(text=sample[1])
        input_text = processor.apply_chat_template(msg, add_generation_prompt=True)
        input_image = Image.open(sample[2])
        input = processor(input_image, input_text, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt").to(device)
        results.append(_classify_until_answer(model, input, req_logit_diff, true_token_id, false_token_id, topk=1))
        input_image.close()
        since_last_save += 1
        if save_every and since_last_save >= save_every:
            with open(os.path.join(save_every_dir, "Meta.tsv"), 'w', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows([dataset[idx] for idx, classification in enumerate(results) if classification == True])
            since_last_save = 0

    return results, dataset

def _classify_until_answer(model, input, req_logit_diff, id_1, id_0, max_steps=10, topk=5)->bool:
    """
    Classify until the model gives a clear answer or reaches max_steps. If max_steps is reached, false classification is assumed.
    **only works for 1 sample at a time currently**
    """
    device = input["input_ids"].device

    for _ in range(max_steps):
        with torch.no_grad():
            output = model(**input)
            logits = output.logits[:, -1, :]  # shape: (1, vocab_size)

        topk_ids = torch.topk(logits, topk, dim=-1).indices[0].tolist()

        if id_1 in topk_ids or id_0 in topk_ids:
            target_logits = logits[:, [id_1, id_0]]
            difference = target_logits[:, 0] - target_logits[:, 1]
            prediction = difference >= req_logit_diff
            # logit_pred = torch.nn.functional.softmax(target_logits, dim=1)
            # prediction = True if logit_pred[0, 0] >= threshold else False

            return prediction.item()

        # Append the most likely token and update attention masks
        next_token_id = topk_ids[0]
        # print(processor.tokenizer.decode([next_token_id]))
        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        input["input_ids"] = torch.cat([input["input_ids"], next_token_tensor], dim=1)
        next_attention_mask = torch.ones_like(next_token_tensor)
        input["attention_mask"] = torch.cat([input["attention_mask"], next_attention_mask], dim=1)
        next_cross_attention_mask = torch.tensor([[[[1, 1, 0, 0]]]], device=device)
        input["cross_attention_mask"] = torch.cat([input["cross_attention_mask"], next_cross_attention_mask], dim=1)
        "[[[[1,1,0,0]]],[[[1,1,0,0]]],...]"
        
    print(f"Reached classification attempt limit of {max_steps}")
    return False