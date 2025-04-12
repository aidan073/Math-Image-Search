import os
import torch
from PIL import Image
from huggingface_hub import login
from transformers import MllamaForConditionalGeneration, AutoProcessor

def filter(dataset_path:str, output_path:str, math_threshold:int=0.5, similarity_threshold:int=0.5, env_path:str=None, hf_token:str=None):
    """
    Filter math samples from datasets using Llama3.2 vision.
    
    Args:
        dataset_path:
        output_path: Folder to store filtered dataset in.
        math_threshold: "confidence" required to classify sample as math. Specifically, confidence = output of softmax(true_token) with denominator containing true_token and false_token.
        similarity_threshold: "confidence" required to classify a text and image as being related. Specifically, confidence = output of softmax(true_token) with denominator containing true_token and false_token.
        env_path: Path to .env file, which must contain "HF_TOKEN" field with hugging face llama access token as its value. 
        hf_token: Hugging face llama access token. You must provide this, or env_path.
    """
    assert env_path or hf_token

def __init__(self,model_id:str,prompt:str,env_path=".env",device=None):
    
    self.device=device
    if not device:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Login to hugging face to get access to Llama models
    envs=load_huggingface_environment(env_path)
    login(envs["HF_TOKEN"])
    

        
        
    self.model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    try:
        self.model.to(device)
    except:
        print("Model was not successfully loaded to GPU")
    
    self.prompt=prompt
    self.processor = AutoProcessor.from_pretrained(model_id)
    
    
def caption(self, image_path):
    
    image = Image.open(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self.prompt}
        ]}
    ]
    input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = self.processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(self.model.device)

    output = self.model.generate(**inputs, max_new_tokens=200)
    prompt_len = inputs.input_ids.shape[-1]

    generated_ids = output[:, prompt_len:]

    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return generated_text[0]
    
        
if __name__=="__main__":
    captioner=LlamaCaptioner("meta-llama/Llama-3.2-11B-Vision","Caption this image")
    captioner.caption("misc/test.jpg")