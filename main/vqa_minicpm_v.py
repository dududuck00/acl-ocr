import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import json

torch.manual_seed(100)

enable_thinking=False # If `enable_thinking=True`, the thinking mode is enabled.
stream=True # If `stream=True`, the answer is string

def load_model():
    model = AutoModel.from_pretrained('OpenBMB/MiniCPM-V-4_5', trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained('OpenBMB/MiniCPM-V-4_5', trust_remote_code=True) # or openbmb/MiniCPM-o-2_6
    return model, tokenizer

def infer(model, tokenizer, image_path, question, options):
    image = Image.open(image_path).convert('RGB')
    msgs = [
        {'role': 'user', 'content': [
            image, 
            f"You are a helpful assistant for visual question answering. Answer the question based on the image content. Respond only with the option letter (A, B, C, or D).\nQuestion: {question}\nOptions:\n" + "\n".join([f"{opt[0]}. {opt[1:]}" for opt in options]) + "\nAnswer:"
        ]}
    ]
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        stream=True
    )
    generated_text = ""
    for new_text in answer:
        generated_text += new_text
    return generated_text

def process_single_item(item, model, tokenizer, imgs_dir):
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
            llm_answer = infer(model, tokenizer, image_path, question, options)
            item["qa_pairs"][index]["LLMAnswer"] = llm_answer.strip()
        return item
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
        return None

if __name__ == "__main__":
    model, tokenizer = load_model()
    data_path = "../fox_data/qa/qa_recheck.json"
    imgs_dir = "../fox_data/from_text"
    
    # 读取数据
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 结果
    results = []
    
    for item in tqdm(data, desc="Processing items"):
        result = process_single_item(item, model, tokenizer, imgs_dir)
        if result is not None:
            results.append(result)
    
    # 保存结果
    save_path = "../results/vqa/minicpm_v_vqa_results.json"
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Results saved to {save_path}")