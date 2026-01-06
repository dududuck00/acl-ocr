import openai
import os
import json
import time
import base64

# model_name = "qwen2.5-vl:72b"
# base_url = "https://uni-api.cstcloud.cn/v1"
# api_key = "08d76ea0a6322a2ab7c49fc2a9cacb75c4457e67b5db4e1499fe3db963e86ac8"


def chat(image_path, client, model_name):
    image = "data:image/png;base64," + base64.b64encode(open(image_path, "rb").read()).decode()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system", 
                "content": [{"type":"text","text": "You are a professional OCR tool. Your task is to transcribe the text in the image exactly as it appears. Do not interpret, summarize, or comment on the text. Even if the text is random, nonsensical, or gibberish, output it exactly. Do not add any conversational filler."}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        }
                    },
                    {
                        "type":"text",
                        "text":"Transcribe all the text in this image exactly as it is. Output ONLY the transcribed text."
                    }
                ]
            }
        ],
        max_tokens=8192,
        temperature=0.0,
    )
    return response.choices[0].message.content

# 用多线程使用模型进行ocrs识别
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_single_image(item, client, model_name):
    image_name = item["image"]
    image_path = os.path.join(imgs_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    try:
        ocr_text = chat(image_path, client, model_name)
        return image_name, ocr_text
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return None

if __name__ == "__main__":
    
    base_url = "https://api.siliconflow.cn/v1"
    api_key = "sk-dclewehmwoihlaisrxlzdsfdnsiohcpdopqfmswpzomndpat"
    client = openai.Client(api_key=api_key, base_url=base_url)

    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"

    data_path = "../fox_data/data.json"
    imgs_dir = "../fox_data/from_text/"
    save_path = "../results/other/qwen2.5_vl_72B_from_text.json"
    
    
    # 读取数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 结果字典
    results_map = {}
    
    # 并行处理
    max_workers = 16  # 根据API限制调整并发数
    print(f"Starting OCR with {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = [executor.submit(process_single_image, item, client, model_name) for item in data]
        
        # 获取结果
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                image_name, ocr_text = result
                results_map[image_name] = ocr_text

    # 合并结果
    for item in data:
        image_name = item["image"]
        if image_name in results_map:
            item["ocr_text"] = results_map[image_name]
    
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"Results saved to {save_path}")
