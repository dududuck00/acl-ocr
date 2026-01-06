import sys,os,time,re,json,torch,transformers
import torch.multiprocessing as mp
import math
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from PIL import Image
import argparse

ocr_model_path = "../model/deepseek-ocr"
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6,7'


def load_model(ocr_model_path,gpu_id=0):
     tokenizer = AutoTokenizer.from_pretrained(
         ocr_model_path, _attn_implementation='flash_attention_2', trust_remote_code=True
     )
     model = AutoModel.from_pretrained(
         ocr_model_path, trust_remote_code=True, use_safetensors=True
     )
     # 全局加载一次模型
     model = model.eval().cuda(f"cuda:{gpu_id}").to(torch.bfloat16)
     return tokenizer, model

prompt = "<image>\nFree OCR. "

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 原论文中只评估了 tiny 和 small 模型 分别对应的image_size base_size = 512 640
def ocr_image(tokenizer, model, image_file, output_path, image_size=512,base_size=512, mode="tiny"):
    if mode == "raw":   # 如果是 raw 模式，则不进行压缩
        test_compress = False
    else:
        test_compress = True
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
        test_compress=test_compress,
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

def clean_ocr_output(text):
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

def process_single_image(tokenizer, model, image_name, output_path, imgs_dir, mode):
    """
    处理单张图片，返回清理后的 OCR 文本
    """
    process_func = ocr_image
    if mode == "tiny":
        IMAGE_SIZE = 512
        BASE_SIZE = 512
    elif mode == "small":
        IMAGE_SIZE = 640
        BASE_SIZE = 640
    elif mode == "raw":
        IMAGE_SIZE = 1024
        BASE_SIZE = 1024
    elif mode == "base":
        IMAGE_SIZE = 1024
        BASE_SIZE = 1024
    elif mode == "large":
        IMAGE_SIZE = 1280
        BASE_SIZE = 1280
        
    image_path = os.path.join(imgs_dir, image_name)
    res = process_func(
        tokenizer,
        model,
        image_path,
        output_path=output_path,
        image_size=IMAGE_SIZE,
        base_size=BASE_SIZE,
        )
    clean_text = clean_ocr_output(res)
    return clean_text

def worker_process(gpu_id, image_names, output_path, imgs_dir, mode, return_dict):
    try:
        tokenizer, model = load_model(ocr_model_path, gpu_id=gpu_id)
        local_results = []
        for image_name in tqdm(image_names, desc=f"GPU {gpu_id}"):
            try:
                clean_text = process_single_image(tokenizer, model, image_name, output_path, imgs_dir, mode)
                local_results.append({
                    "image": image_name,
                    "ocr_text": clean_text
                })
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
        return_dict[gpu_id] = local_results
    except Exception as e:
        print(f"Worker process error on GPU {gpu_id}: {e}")

def check_gpu_memory():
    """
    检查所有 GPU 的可用显存，返回可用显存大于 20GB 的 GPU 列表
    """
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
        available_mem = total_mem - (allocated_memory + reserved_memory)
        print(f"GPU {i}: Total Memory: {total_mem:.2f} GB, Allocated Memory: {allocated_memory:.2f} GB, Reserved Memory: {reserved_memory:.2f} GB, Available Memory: {available_mem:.2f} GB")
        if available_mem > 20:
            available_gpus.append(i)
    return available_gpus


def ocr(data_path=None, output_path = "../output", save_path=None, imgs_dir=None, mode="tiny"):
    ocr_results = []
    # 是否并行处理
    parallel_process = False
    if "random" in imgs_dir:
        image_names = [f"random_{i+1}.png" for i in range(112)]
    else:
        # image_names = [f"en_{i+1}.png" for i in range(112)]
        image_names = os.listdir(imgs_dir)
        image_names = sorted(image_names, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # image_paths = [os.path.join(images_dir, img_name) for img_name in image_names]
    print(f"开始处理 {len(image_names)} 张图片...")
    
    # 检查可用显卡
    available_gpus = check_gpu_memory()
    if len(available_gpus) >= 2:
        # 加载多个模型并行处理数据
        parallel_process = True
    if parallel_process:
        print(f"检测到多张GPU，启用并行处理模式...")
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        
        # Split images
        chunk_size = math.ceil(len(image_names) / len(available_gpus))
        
        for i, gpu_id in enumerate(available_gpus):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(image_names))
            subset = image_names[start_idx:end_idx]
            
            if not subset:
                continue

            p = mp.Process(target=worker_process, args=(gpu_id, subset, output_path, imgs_dir, mode, return_dict))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
        for gpu_id in available_gpus:
            if gpu_id in return_dict:
                ocr_results.extend(return_dict[gpu_id])
    if not parallel_process:
        gpu_id = check_gpu_memory()[0] if len(check_gpu_memory()) > 0 else 0
        tokenizer, model = load_model(ocr_model_path, gpu_id=gpu_id)
        for image_name in tqdm(image_names):
            clean_text = process_single_image(tokenizer, model, image_name, output_path, imgs_dir, mode)
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
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--data_path", type=str, default="../fox_data/data.json", help="输入数据文件路径")
    # arg_parser.add_argument("--output_path", type=str, default="../output", help="输出结果文件夹路径")
    # arg_parser.add_argument("--save_path", type=str, default="../results/ocr_results.json", help="保存结果文件路径")
    # arg_parser.add_argument("--images_dir", type=str, default="../fox_data/distort", help="图片文件夹路径")
    # arg_parser.add_argument("--mode", type=str, default="tiny", help="分辨率模式 tiny/small/raw")
    # args = arg_parser.parse_args()
    
    # data_path = "../fox_data/compress.json"
    # output_path = args.output_path
    # images_dir = "../fox_data/compress"
    # modes = ["tiny", "small"]
    # for mode in modes:
    #     if mode == "tiny":
    #         save_path = "../results/compress/compress_tiny_ocr_results.json"
    #     elif mode == "small":
    #         save_path = "../results/compress/compress_small_ocr_results.json"
    #     print(f"处理数据文件夹：{images_dir} , 保存路径：{save_path} 分辨率模式：{mode}")
    #     print(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    #     ocr(
    #         data_path=data_path,
    #         output_path=output_path,
    #         save_path=save_path,
    #         imgs_dir=images_dir,
    #         mode=mode
    #     )
    #     print(f"结束处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    #     print(f"总共用时: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    #     time.sleep(10)
    
    stories = os.listdir("../fox_data/story_txt/")
    for story in stories:
        arg_parser = argparse.ArgumentParser()
        story_name = story.replace(".txt", "")
        data_path = f"../fox_data/story_data/{story_name}_data.json"
        imges_dir = f"../fox_data/story_images/{story_name}/"
        output_path = "../output"
        # modes = ["tiny", "small"]
        modes = ["large"]
        for mode in modes:
            if mode == "tiny":
                save_path = f"../results/compress/{story_name}_ocr_tiny_results.json"
            elif mode == "small":
                save_path = f"../results/compress/{story_name}_ocr_small_results.json"
            elif mode == "base":
                save_path = f"../results/compress/{story_name}_ocr_base_results.json"
            elif mode == "large":
                save_path = f"../results/compress/{story_name}_ocr_large_results.json"
            print(f"处理数据文件夹：{imges_dir} , 保存路径：{save_path} 分辨率模式：{mode}")
            print(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            ocr(
                data_path=data_path,
                output_path=output_path,
                save_path=save_path,
                imgs_dir=imges_dir,
                mode=mode
            )
            print(f"结束处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            print(f"总共用时: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
            time.sleep(10)
    
    
    # data_path = "../fox_data/replace_swap_5.json"
    # output_path = args.output_path
    # images_dir = "../fox_data/replace_swap_5"
    # modes = ["tiny", "small", "raw"]
    # for mode in modes:
    #     if mode == "tiny":
    #         save_path = "../results/replace/swap_5/swap_5_tiny.json"
    #     elif mode == "small":
    #         save_path = "../results/replace/swap_5/swap_5_small.json"
    #     elif mode == "raw":
    #         save_path = "../results/replace/swap_5/swap_5_raw.json"
    #     print(f"处理数据文件夹：{images_dir} , 保存路径：{save_path} 分辨率模式：{mode}")
    #     print(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    #     ocr(
    #         data_path=data_path,
    #         output_path=output_path,
    #         save_path=save_path,
    #         imgs_dir=images_dir,
    #         mode=mode
    #     )
    #     print(f"结束处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    #     print(f"总共用时: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    #     time.sleep(10)
    
    # data_path = "../fox_data/replace_shuffle_5.json"
    # output_path = args.output_path
    # images_dir = "../fox_data/replace_shuffle_5"
    # modes = ["tiny", "small", "raw"]
    # for mode in modes:
    #     if mode == "tiny":
    #         save_path = "../results/replace/shuffle_5/shuffle_5_tiny.json"
    #         pass
    #     elif mode == "small":
    #         save_path = "../results/replace/shuffle_5/shuffle_5_small.json"
    #     elif mode == "raw":
    #         save_path = "../results/replace/shuffle_5/shuffle_5_raw.json"
    #     print(f"处理数据文件夹：{images_dir} , 保存路径：{save_path} 分辨率模式：{mode}")
    #     print(f"开始处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    #     ocr(
    #         data_path=data_path,
    #         output_path=output_path,
    #         save_path=save_path,
    #         imgs_dir=images_dir,
    #         mode=mode
    #     )
    #     print(f"结束处理时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    #     print(f"总共用时: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    #     time.sleep(10)
    