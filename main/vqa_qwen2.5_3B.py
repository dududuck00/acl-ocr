from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import json
import os

def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    return model, processor

def infer(model, processor, image_path, question, options):
    messages=[
            {
                "role": "system", 
                "content": [{"type":"text",
                             "text": "You are a helpful assistant for visual question answering. Answer the question based on the image content. Respond only with the option letter (A, B, C, or D)."}]
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
        ]
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
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
    data_path = "/data/home/yunhao/code/ocr/fox_data/qa/qa_recheck.json"
    imgs_dir = "/data/home/yunhao/code/ocr/fox_data/from_text"
    
    # 读取数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 结果
    results = []
    
    for item in tqdm(data, desc="Processing items"):
        result = process_single_item(item, model, processor, imgs_dir)
        if result is not None:
            results.append(result)
    
    # 保存结果
    save_path = "/data/home/yunhao/code/ocr/results/vqa/qwen2.5_vl_3B_vqa_results.json"
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Results saved to {save_path}")