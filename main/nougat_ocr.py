import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from pathlib import Path
from tqdm import tqdm
import os
import json

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def load_images(image_paths):
    images = []
    for image_path in image_paths:
        image = load_image(image_path)
        images.append(image)
    return images

def load_model(device):
    processor = AutoProcessor.from_pretrained("facebook/nougat-base")
    model = AutoModelForVision2Seq.from_pretrained("facebook/nougat-base").to(device)
    model.eval()
    return processor, model

def generate_text(processor, model, image_paths, device):
    results = []
    for image_path in tqdm(image_paths, desc="Generating text"):
        image_name = Path(image_path).name
        inputs = processor(images=load_image(image_path), return_tensors="pt").to(device)
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            max_new_tokens=4096,
            early_stopping=True,
        )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append({"image": image_name, "ocr_text": text})
    return results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor, model = load_model(device)
image_paths = [f"/data/home/yunhao/code/ocr/fox_data/from_text/en_{i}.png" for i in range(1,113)]
# images = load_images(image_paths)
all_results = generate_text(processor, model, image_paths, device)

all_results = sorted(all_results, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))

with open("/data/home/yunhao/code/ocr/results/other/nougat_ocr_from_text.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)