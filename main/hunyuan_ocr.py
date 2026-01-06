from modelscope import AutoProcessor
from modelscope import HunYuanVLForConditionalGeneration
from PIL import Image
import torch
import os
import json
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n<8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]  

    return text

def load_model():
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        "/data/home/yunhao/code/ocr/model/hunyuan",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("/data/home/yunhao/code/ocr/model/hunyuan", use_fast=False)
    return model, processor

def infer(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": (
                        "Free OCR"
                    )},
                ],
            }
        ],
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    inputs = processor(
        text=texts,
        images=image,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=16384, do_sample=False)
    if "input_ids" in inputs:
        input_ids = inputs.input_ids
    else:
        print("inputs: # fallback", inputs)
        input_ids = inputs.inputs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts[0]

if __name__ == "__main__":
    model, processor = load_model()
    data_path = "/data/home/yunhao/code/ocr/fox_data/data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in tqdm(data, desc="Processing images"):
        image_name = item["image"]
        image_path = f"/data/home/yunhao/code/ocr/fox_data/from_text/{image_name}"
        ocr_text = infer(model, processor, image_path)
        item["ocr_text"] = ocr_text
    
    data = sorted(data, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))
    
    with open("/data/home/yunhao/code/ocr/results/other/hunyuan_ocr_from_text.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    data_path = "/data/home/yunhao/code/ocr/fox_data/random.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in tqdm(data, desc="Processing images"):
        image_name = item["image"]
        image_path = f"/data/home/yunhao/code/ocr/fox_data/random/{image_name}"
        ocr_text = infer(model, processor, image_path)
        item["ocr_text"] = ocr_text
    
    data = sorted(data, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))
    
    with open("/data/home/yunhao/code/ocr/results/other/hunyuan_ocr_random.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)