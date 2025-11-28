from PIL import Image
import json
import re
import os
from pathlib import Path
import argparse
import time
import importlib
import sys

# TODO: 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["DEESEEK_MODEL_PATH"] = str(Path("/root/code/research/DeepSeek-OCR/model/deepseek-ocr").resolve())
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def clean_ocr_output(text):
    matches_ref, matches_images, mathes_other = re_match(text)

    for idx, a_match_image in enumerate(matches_images):
        text = text.replace(a_match_image, f'![](images/{idx}.jpg)\n')

    for a_match_other in mathes_other:
        text = text.replace(a_match_other, '')

    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    text = text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
    text = text.replace('<center>', '').replace('</center>', '')

    return text.strip()


def load_model():
    llm = LLM(
        model="/root/code/research/DeepSeek-OCR/model/deepseek-ocr",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        gpu_memory_utilization=0.8,
    )
    return llm


def main(args):
    img_folder = args.input_folder
    output_file = args.output
    img_names = [f"en_{i+1}.png" for i in range(112)]
    image_paths = [os.path.join(img_folder, img_name) for img_name in img_names]
    test_imgs = [Image.open(img_path).convert("RGB") for img_path in image_paths]

    # TODO: debug only
    test_imgs = test_imgs[:64]

    print(f"共加载 {len(test_imgs)} 张图片进行 OCR 识别。")

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    llm = load_model()

    model_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        }
        for img in test_imgs
    ]

    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )

    model_outputs = llm.generate(model_inputs, sampling_param)

    data = load_data("/root/code/research/DeepSeek-OCR/fox_data/data.json")

    data = data[:len(test_imgs)]

    for i, output in enumerate(model_outputs):
        ocr_text = clean_ocr_output(text=output.outputs[0].text)
        data[i]['ocr_text'] = ocr_text

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    del llm


def _configure_mode(mode: str):
    presets = {
        "tiny": {"base": 512, "image": 512, "crop": False},
        "small": {"base": 640, "image": 640, "crop": False},
        "raw": {"base": 1024, "image": 640, "crop": False},
    }
    return presets.get(mode, presets["raw"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Input image file path")
    parser.add_argument("--output", type=str, required=False, help="Output text file path")
    parser.add_argument("--mode", type=str, default="tiny", choices=["tiny", "small", "raw"], help="Model mode to use")

    args = parser.parse_args()

    if args.output is None:
        data_folder_name = args.input_folder.strip('/').split('/')[-1]
        args.output = f"/root/code/research/DeepSeek-OCR/results/fox/{data_folder_name}_{args.mode}.json"

    from vllm.transformers_utils.processors import deepseek_ocr as processor_cfg

    mode_cfg = _configure_mode(args.mode)
    processor_cfg.BASE_SIZE = mode_cfg["base"]
    processor_cfg.IMAGE_SIZE = mode_cfg["image"]
    processor_cfg.CROP_MODE = mode_cfg["crop"]

    module_name = "vllm.model_executor.models.deepseek_ocr"
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)

    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    print(f"开始处理数据集: {args.input_folder}, 输出文件: {args.output}, 模式: {args.mode}")
    start_time = time.localtime()
    print("开始时间:", time.strftime("%Y-%m-%d %H:%M:%S", start_time))

    main(args)

    end_time = time.localtime()
    print("结束时间:", time.strftime("%Y-%m-%d %H:%M:%S", end_time))
    print("总耗时:{}分{}秒".format(
        int(time.mktime(end_time) - time.mktime(start_time)) // 60,
        int(time.mktime(end_time) - time.mktime(start_time)) % 60,
    ))