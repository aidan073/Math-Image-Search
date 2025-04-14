import os
import csv
import math
import torch
from tqdm import tqdm
from PIL import Image
from typing import Union
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import MllamaForConditionalGeneration, AutoProcessor

def filter(dataset_or_path:Union[str, list[str]], missing_or_path:Union[str, list[str]], math_output_path:str=None, sim_output_path:str=None, math_threshold:int=0.5, similarity_threshold:int=0.5, batch_size:int=100, env_path:str=None, hf_token:str=None):
    """
    Filter math samples from datasets using Llama3.2 vision.
    
    Args:
        dataset_or_path: Path to dataset .tsv file, or dataset of format: [[id, title/text, image_path], ...]
        missing_or_path: Path to missing .txt file, or set with missing ids. These will be excluded from the returned datasets.
        math_output_path: .tsv file to save math filtered dataset in.
        sim_output_path: .tsv file to save similarity filtered dataset in.
        math_threshold: "confidence" required to classify sample as math. Specifically, confidence = output of softmax(true_token) with denominator containing true_token and false_token.
        similarity_threshold: "confidence" required to classify a text and image as being related. Specifically, confidence = output of softmax(true_token) with denominator containing true_token and false_token.
        batch_size: Batch size for classifier input.
        env_path: Path to .env file, which must contain "HF_TOKEN" field with hugging face llama access token as its value. 
        hf_token: Hugging face llama access token. You must provide this, or env_path.
    """
    MATH_PROMPT = "Text: {text}\n\nDoes the image and text content relate to math? Respond with 1 if yes, or 0 for no. Output only the number."
    SIM_PROMPT = "Text: {text}\n\nAre the image and text related? Respond with 1 if yes, or 0 for no. Output only the number."
    MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    if math_output_path:
        if os.path.exists(math_output_path):
            raise FileExistsError(f"Designated output_path: {math_output_path} already exists. Please delete it or provide a different output_path.")
        os.makedirs(math_output_path)
    if sim_output_path:
        if os.path.exists(sim_output_path):
            raise FileExistsError(f"Designated output_path: {sim_output_path} already exists. Please delete it or provide a different output_path.")
        os.makedirs(sim_output_path)

    assert env_path or hf_token
    if(env_path):
        load_dotenv(env_path)
        hf_token = os.getenv("HF_TOKEN") 
    login(hf_token)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    math_inputs, _ = _prepare_input(dataset_or_path, missing_or_path, MATH_PROMPT, batch_size, processor, device)
    sim_inputs, no_missing_dataset = _prepare_input(dataset_or_path, missing_or_path, SIM_PROMPT, batch_size, processor, device)

    math_classifications = _get_classifications(math_inputs, math_threshold)
    sim_classifications = _get_classifications(sim_inputs, similarity_threshold)

    true_math_samples = [no_missing_dataset[idx] for idx, classification in enumerate(math_classifications) if classification == True]
    true_sim_samples = [no_missing_dataset[idx] for idx, classification in enumerate(sim_classifications) if classification == True]

    if math_output_path:
        with open(math_output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(true_math_samples)
    if sim_output_path:
        with open(sim_output_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(true_sim_samples)

    return true_math_samples, true_sim_samples

def _prepare_input(dataset_or_path, missing_or_path, prompt, batch_size, processor, device)->Union[list[str], list[str]]:
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
    else:
        missing = missing_or_path
    # load dataset
    if isinstance(dataset_or_path, str):
        with open(dataset_or_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t")
            dataset = [sample for sample in reader if sample[0] not in missing]
    else:
        dataset = [sample for sample in dataset_or_path if sample[0] not in missing]
    # process batches of data
    inputs = []
    for i in tqdm(range(0, len(dataset), batch_size), total=math.ceil(len(dataset) / batch_size), desc="Preparing dataset for model input"):
        texts = []
        images = []
        for sample in dataset[i:i+batch_size]:
            msg[0]["content"][1]["text"] = prompt.format(text=sample[1])
            input_text = processor.apply_chat_template(msg, add_generation_prompt=True)
            texts.append(input_text)
            images.append([Image.open(sample[2])])

        inputs.append(processor(images, texts, add_special_tokens=False, padding=True, truncation=True, return_tensors="pt").to(device))

    return inputs, dataset

def _get_classifications(model, processor, inputs, threshold)->list[bool]:
    """
    Produce list of predictions from input batch.
    """
    tokenizer = processor.tokenizer
    true_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("1")[0])
    false_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("0")[0])
    predictions = []
    for batch in tqdm(inputs, total=len(inputs), desc="Filtering batches"):
        with torch.no_grad():
            output = model(**inputs)
            next_token_logits = output.logits[:, -1, :]
            target_logits = next_token_logits[:, [true_token_id, false_token_id]]
            # calculate true vs false probability
            probs = torch.nn.functional.softmax(target_logits, dim=1)
            temp_predictions = (probs[:, 0] >= threshold).tolist()
            predictions += temp_predictions

    return predictions