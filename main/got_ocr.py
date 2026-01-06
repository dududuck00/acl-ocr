from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

local_model_path = "/data/home/yunhao/code/ocr/model/got-ocr"

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

# input your test image
# image_file = "../fox_data/random/random_1.png"

# plain texts OCR
# res = model.chat(tokenizer, image_file, ocr_type='ocr')

# format texts OCR:
# res = model.chat(tokenizer, image_file, ocr_type='format')

# fine-grained OCR:
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# multi-crop OCR:
# res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# render the formatted OCR results:
# res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

with open("/data/home/yunhao/code/ocr/fox_data/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

img_dir = "/data/home/yunhao/code/ocr/fox_data/from_text/"
img_names = image_names = [f"en_{i+1}.png" for i in range(112)]

for img_name in tqdm(img_names):
    img_path = os.path.join(img_dir, img_name)
    ocr_text = model.chat(tokenizer, img_path, ocr_type='ocr')
    for item in data:
        if item["image"] == img_name:
            item["ocr_text"] = ocr_text
            break

save_path = "/data/home/yunhao/code/ocr/results/other/got_ocr_from_text.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)