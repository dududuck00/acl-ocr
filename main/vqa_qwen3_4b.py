from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
from tqdm import tqdm
import json
import torch
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# def load_model():
#     model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
#     )
#     processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
#     return model, processor
 

def infer(model, processor, image_path, question, options):
    image = Image.open(image_path).convert("RGB")
    messages=[
            {
                "role": "system", 
                "content": [{"type":"text","text": "You are a helpful assistant for visual question answering. Answer the question based on the image content. Respond only with the option letter (A, B, C, or D)."}]
            },
            {
                "role": "user",
                "content": [
                    {   
                        "type":"image",
                        "image": image_path 
                    },
                    {
                        "type":"text",
                        "text": f"Question: {question}\nOptions:\n" + "\n".join([f"{opt[0]}. {opt[1:]}" for opt in options]) + "\nAnswer:"
                    }
                ]
            }
        ],
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def process_single_item(item, model, processor, imgs_dir):
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
            llm_answer = infer(model, processor, image_path, question, options)
            item["qa_pairs"][index]["LLMAnswer"] = llm_answer.strip()
        return item
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return None

if __name__ == "__main__":
    model, processor = load_model()
    data_path = "../fox_data/qa/qa_recheck.json"
    imgs_dir = "../fox_data/from_text"
    
    # 读取数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 结果
    results = []
    
    # 并行处理
    # max_workers = 16  # 根据API限制调整并发数
    # print(f"Starting OCR with {max_workers} threads...")
    
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     # 提交任务
    #     futures = [executor.submit(process_single_item, item,) for item in data]
        
    #     # 获取结果
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         result = future.result()
    #         if result is not None:
    #             results.append(result)
    
    # 重新排序结果
    # results = sorted(results, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))
    
    for item in tqdm(data, desc="Processing items"):
        result = process_single_item(item, model, processor, imgs_dir)
        if result is not None:
            results.append(result)
    
    # 保存结果
    save_path = "../results/vqa/qwen3_vl_4B_vqa_results.json"
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Results saved to {save_path}")