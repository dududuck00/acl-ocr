import torch,json,sys,os,time,re,transformers
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
ocr_model_path = "../model/deepseek-ocr"

# 全局锁用于模型推理（确保线程安全）
# model_lock = threading.Lock()
# results_lock = threading.Lock()


tokenizer = AutoTokenizer.from_pretrained(ocr_model_path, _attn_implementation='flash_attention_2', trust_remote_code=True)
model = AutoModel.from_pretrained(
    ocr_model_path, trust_remote_code=True, use_safetensors=True
)
# 全局加载一次模型
model = model.eval().cuda("cuda:3").to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
# image_file = 'your_image.jpg'
# output_path = 'your/output/dir'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 原论文中只评估了 tiny 和 small 模型 分别对应的image_size base_size = 512 640
def ocr_image(tokenizer, model, image_file, output_path, image_size=512,base_size=512):
    with torch.inference_mode():
        res = model.infer(
            tokenizer=tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            # crop_mode=True,
            crop_mode=False,
            # save_results=True,    # 这个设置会将结果保存到output_path目录下
            save_results=False,
            eval_mode=True,         # 评估模式，不保存结果，将结果返回
            test_compress=True,
        )
        return res

def re_match(text):
    """
    提取 grounding 标记
    返回:
        matches: 所有匹配项 (完整标记, 文本内容, 坐标)
        mathes_image: 图片相关的标记
        mathes_other: 文本相关的标记
    """
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

def clean_ocr_output_official(text):
    """
    官方的清理方法：
    1. 提取所有标记
    2. 替换图片标记为 markdown 图片格式
    3. 删除所有文本标记
    """
    matches_ref, matches_images, mathes_other = re_match(text)
    
    # 替换图片标记
    for idx, a_match_image in enumerate(matches_images):
        text = text.replace(a_match_image, f'![](images/{idx}.jpg)\n')
    
    # 删除所有文本标记
    for idx, a_match_other in enumerate(mathes_other):
        text = text.replace(a_match_other, '')
    
    # 额外清理
    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    text = text.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
    text = text.replace('<center>', '').replace('</center>', '')
    
    return text.strip()

def process_single_image(image_path, output_path,images_dir,mode):
    """
    处理单张图片，返回清理后的 OCR 文本
    """
    if mode == "tiny":
        IMAGE_SIZE = 512
        BASE_SIZE = 512
    elif mode == "small":
        IMAGE_SIZE = 640
        BASE_SIZE = 640
    image_path = os.path.join(images_dir, image_path)
    res = ocr_image(
        tokenizer,
        model,
        image_path,
        output_path=output_path,
        image_size=IMAGE_SIZE,
        base_size=BASE_SIZE,
        )
    clean_text = clean_ocr_output_official(res)
    return clean_text


def main(data_path=None, output_path = "../output", save_path=None, images_dir=None, mode="tiny"):
    ocr_results = []
    image_names = [img_name for img_name in os.listdir(images_dir)]
    print(f"开始处理 {len(image_names)} 张图片...")
    
    for image_name in tqdm(image_names):
        clean_text = process_single_image(image_name, output_path, images_dir,mode)
        ocr_results.append({
            "image": image_name,
            "ocr_text": clean_text
        })
    # 按照image name重新排序
    ocr_results = sorted(
        ocr_results,
        key=lambda x: int(x["image"].split("_")[-1].split(".")[0])
    )
    
    # 将结果合并到原始数据中
    data = load_data(data_path)
    for item in data:
        image_name = item["image"]
        for ocr_item in ocr_results:
            if ocr_item["image"] == image_name:
                item["ocr_text"] = ocr_item["ocr_text"]
                break
    
    # 将最终结果保存到文件中
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    
    print(f"\n结果已保存到: {save_path}")

if __name__ == "__main__":
    data_path = "../fox_data/data.json"
    output_path = "../output"
    images_dir = "../fox_data/distort"
    modes = ["tiny", "small"]
    for mode in modes:
        if mode == "tiny":
            save_path = "../fox/distort_mode_tiny.json"
        elif mode == "small":
            save_path = "../fox/distort_mode_small.json"
        print(f"处理数据文件夹：{images_dir} , 保存路径：{save_path} 分辨率模式：{mode}")
        print(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        main(data_path, output_path, save_path, images_dir,mode)
        print(f"结束处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"总共用时: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    