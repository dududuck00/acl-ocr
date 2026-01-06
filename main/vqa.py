import openai
import os
import json
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# model_name = "qwen2.5-vl:72b"
# base_url = "https://uni-api.cstcloud.cn/v1"
# api_key = "08d76ea0a6322a2ab7c49fc2a9cacb75c4457e67b5db4e1499fe3db963e86ac8"


# model_name = "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
model_name = "Qwen/Qwen3-VL-8B-Instruct"
base_url = "https://api.siliconflow.cn/v1"
api_key = "sk-dclewehmwoihlaisrxlzdsfdnsiohcpdopqfmswpzomndpat"

client = openai.Client(api_key=api_key, base_url=base_url)

data_path = "../fox_data/qa/qa_recheck.json"
imgs_dir = "../fox_data/from_text"


def chat(image_path, question, options):
    image = "data:image/png;base64," + base64.b64encode(open(image_path, "rb").read()).decode()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system", 
                "content": [{"type":"text","text": "You are a helpful assistant for visual question answering. Answer the question based on the image content. Respond only with the option letter (A, B, C, or D)."}]
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
                        "text": f"Question: {question}\nOptions:\n" + "\n".join([f"{opt[0]}. {opt[1:]}" for opt in options]) + "\nAnswer:"
                    }
                ]
            }
        ],
        max_tokens=8192,
        temperature=0.0,
    )
    return response.choices[0].message.content

def process_single_item(item):
    image_name = item["image"]
    image_path = os.path.join(imgs_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    try:
        qa_pairs = item["qa_pairs"]
        for index, qa in enumerate(qa_pairs):
            question = qa["question"]
            options = qa["options"]
            llm_answer = chat(image_path, question, options)
            item["qa_pairs"][index]["LLMAnswer"] = llm_answer.strip()
        return item
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return None

if __name__ == "__main__":
    # 读取数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 结果
    results = []
    
    # 并行处理
    max_workers = 16  # 根据API限制调整并发数
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = [executor.submit(process_single_item, item,) for item in data]
        
        # 获取结果
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)

    # 重新排序结果
    results = sorted(results, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))
    
    # 保存结果
    # save_path = "../results/vqa/qwen2.5_vl_vqa_results.json"
    save_path = "../results/vqa/qwen3_vl_8B_vqa_results.json"
    
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Results saved to {save_path}")
