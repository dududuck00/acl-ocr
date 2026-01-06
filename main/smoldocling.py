# Prerequisites:
# pip install torch
# pip install docling_core
# pip install transformers

import json
import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from pathlib import Path
from PIL import Image
import torch.multiprocessing as mp

# 设置多进程启动方法为 'spawn'，以支持 CUDA
mp.set_start_method('spawn', force=True)

import os
import re
from tqdm import tqdm
model_path = "/data/home/yunhao/.cache/huggingface/hub/models--ds4sd--SmolDocling-256M-preview/snapshots/ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8"

def strip_doctags(s: str) -> str:
    # 精准去除 SmolDocling 的 DocTags 标记（结构和位置标记）
    # 结构标记
    struct_tags = r"(doctag|text|title|paragraph|list|item|table|row|cell|figure|caption|code|formula|header|footer|page|column|section)"
    # 位置标记：loc_数字 或 bbox_数字_数字_数字_数字
    loc_tags = r"(loc_\d+|bbox_\d+_\d+_\d+_\d+)"
    # 组合正则：匹配 <tag> 形式
    pattern = rf"<({struct_tags}|{loc_tags})>"
    s = re.sub(pattern, "", s)
    # 合并空白
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_model(gpu_id=0):
    device = f"cuda:{gpu_id}"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to(device)
    return processor, model, device

def load_multiple_models(num_models=10, gpu_id=0):
    device = f"cuda:{gpu_id}"
    processors = []
    models = []
    for i in range(num_models):
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).to(device)
        model.eval()  # 节省显存
        processors.append(processor)
        models.append(model)
        print(f"模型 {i+1} 加载完成，当前显存: {torch.cuda.memory_allocated(device)/1e9:.2f}GB")
    return processors, models, device

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def process_batch(image_paths, gpu_id, num_models=3):
    """处理一批图像，使用指定 GPU，加载多个模型"""
    processors, models, device = load_multiple_models(num_models, gpu_id)
    results = []
    for image_path in tqdm(image_paths, desc=f"Processing on GPU {gpu_id}"):
        model_idx = len(results) % num_models  # 轮流使用不同模型
        processor = processors[model_idx]
        model = models[model_idx]
        try:
            image = load_image(image_path)
            # 手动缩放图像以适应模型（避免 resolution_max_side 错误）
            max_side = 1536  # 根据模型配置调整
            w, h = image.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            # Create input messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Convert this page to docling."}
                    ]
                },
            ]
            # Prepare inputs
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt", do_resize=True)
            inputs = inputs.to(device)

            # Generate outputs
            generated_ids = model.generate(**inputs, max_new_tokens=8192)
            prompt_length = inputs.input_ids.shape[1]
            trimmed_generated_ids = generated_ids[:, prompt_length:]
            doctags = processor.batch_decode(
                trimmed_generated_ids,
                skip_special_tokens=False,
            )[0].lstrip()

            # Populate document
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
            doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
            plain_text = doc.export_to_text()
            image_name = Path(image_path).name
            results.append({
                "image": image_name,
                "ocr_text": plain_text
            })
            # 移除 results[image_path] = plain_text，因为 results 是列表
        except Exception as e:
            image_name = Path(image_path).name
            results.append({
                "image": image_name,
                "ocr_text": f"Error: {str(e)}"
            })
    print(f"Processed {len(results)} images on GPU {gpu_id}")
    return results

def main():
    # 示例图像路径列表（替换为你的实际路径）
    image_paths = []
    image_dir = "/data/home/yunhao/code/ocr/fox_data/from_text/"
    for img_file in os.listdir(image_dir):
        image_paths.append(os.path.join(image_dir, img_file))

    print(f"Total images to process: {len(image_paths)}")
    
    num_models = 32  # 在单 GPU 上加载的模型数量
    gpu_id = 0  # 强制使用 cuda:0
    num_processes = 1  # 单进程（每个进程加载多个模型）

    # 将图像路径分割成批次
    batch_size = len(image_paths) // num_processes + 1
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    batches = batches[:num_processes]

    # 使用 multiprocessing Pool 并行处理（这里单进程，但可扩展）
    with mp.Pool(processes=num_processes) as pool:
        args = [(batch, gpu_id, num_models) for batch in batches]
        results_list = pool.starmap(process_batch, args)

    # 合并结果
    all_results = []
    for res in results_list:
        all_results.extend(res)
    
    # 根据 image 排序结果
    all_results = sorted(all_results, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))
    
    print(f"Total results: {len(all_results)}")
    
    # 输出结果
    # for item in all_results:
    #     print(f"Image: {item['image']}")
    #     print(f"Text: {item['ocr_text'][:200]}...")  # 截断显示
    #     print("-" * 50)
    
    with open("/data/home/yunhao/code/ocr/results/other/smoldocling_from_text.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
