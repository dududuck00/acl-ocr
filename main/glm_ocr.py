from zai import ZhipuAiClient
import json
import base64
import os
import time
from tqdm import tqdm

def read_image_as_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ocr_image(client, image_path):
    image_base64 = read_image_as_base64(image_path)
    response = client.layout_parsing.create(
        model="glm-ocr",
        file=f"data:image/png;base64,{image_base64}",
    )
    return response.md_results



def main(image_dir, save_path, data_file):
    # 初始化客户端
    client = ZhipuAiClient(api_key="a0134674aee16845354e6b804057d705.l4iiU2hVT3W8blCd")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in tqdm(data):
        image_path = os.path.join(image_dir, item["image"])
        ocr_text = ocr_image(client, image_path)
        item["ocr_text"] = ocr_text
        
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

image_dir = "/data/home/yunhao/code/ocr/fox_data/from_text"

if __name__ == "__main__":
    image_dir = "/data/home/yunhao/code/ocr/fox_data/from_text/"
    save_path = "/data/home/yunhao/code/ocr/results/other/glmocr_ocr_from_text.json"
    data_file = "/data/home/yunhao/code/ocr/fox_data/data.json"
    
    main(image_dir, save_path, data_file)
    
    image_dir = "/data/home/yunhao/code/ocr/fox_data/random/"
    save_path = "/data/home/yunhao/code/ocr/results/other/glmocr_ocr_random.json"
    data_file = "/data/home/yunhao/code/ocr/fox_data/random.json"
    main(image_dir, save_path, data_file)
    


