from vllm import LLM, SamplingParams
from PIL import Image
from transformers import AutoProcessor
import json
from tqdm import tqdm

model_path = "/data/home/yunhao/code/ocr/model/hunyuan"
llm = LLM(model=model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0, max_tokens=16384)

img_path = "/path/to/image.jpg"
img = Image.open(img_path)
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": "Free OCR the text in the image."}
    ]}
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
output = llm.generate([inputs], sampling_params)[0]
print(output.outputs[0].text)

data_path = "/data/home/yunhao/code/ocr/fox_data/random.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in tqdm(data, desc="Processing images"):
    image_name = item["image"]
    image_path = f"/data/home/yunhao/code/ocr/fox_data/random/{image_name}"
    img = Image.open(image_path).convert("RGB")
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Free OCR the text in the image."}
                ],
            }
        ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = {"prompt": prompt, "multi_modal_data": {"image": [img]}}
    output = llm.generate([inputs], sampling_params)[0]
    ocr_text = output.outputs[0].text
    item["ocr_text"] = ocr_text

data = sorted(data, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))

with open("/data/home/yunhao/code/ocr/results/other/hunyuan_ocr_random.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)