import os
import json
import openai
import base64
from tqdm import tqdm
from openai import OpenAI

# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

# PaddleOCR-VL 1.5
from paddleocr import PaddleOCRVL

# 使用vllm部署
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM 不校验 key，随便写
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ocr_image(image_path):
    image_base64 = encode_image(image_path)
    # 调用多模态接口
    response = client.chat.completions.create(
        model="PaddlePaddle/PaddleOCR-VL",  # 如果你启动时没改 served-model-name
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "OCR:",
                    },
                ],
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def main(image_dir, save_path, data_file):
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for item in tqdm(data):
        image_path = os.path.join(image_dir, item["image"])
        ocr_text = ocr_image(image_path)
        item["ocr_text"] = ocr_text

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    image_dir = "/data/home/yunhao/code/ocr/fox_data/from_text/"
    save_path = "/data/home/yunhao/code/ocr/results/other/paddle_ocr_vl_from_text.json"
    data_file = "/data/home/yunhao/code/ocr/fox_data/data.json"
    main(image_dir, save_path, data_file)
    
    image_dir = "/data/home/yunhao/code/ocr/fox_data/random/"
    save_path = "/data/home/yunhao/code/ocr/results/other/paddle_ocr_vl_random.json"
    data_file = "/data/home/yunhao/code/ocr/fox_data/random.json"
    main(image_dir, save_path, data_file)
    
    
# pipeline = PaddleOCRVL()

# output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png")
# for res in output:
#     res.print()
#     res.save_to_json(save_path="output")
#     res.save_to_markdown(save_path="output")



# def ocr_image(ocr_model, image_path):
#     result = ocr_model.predict(input=image_path,)
    
    
#     return ocr_text

# def main(image_dir, save_path, data_file):
    
#     # image_dir = "/data/home/yunhao/code/ocr/fox_data/test/"
    
#     with open(data_file, "r", encoding="utf-8") as f:
#         data = json.load(f)
        
#     # 初始化 PaddleOCR 实例
#     ocr_model = PaddleOCR(
#         use_doc_orientation_classify=False,
#         use_doc_unwarping=False,
#         use_textline_orientation=False,
#         device='gpu:0',
#         use_tensorrt=True,
#         cpu_threads=64,
#         rec_batch_num=32,
#         )
    
#     for item in tqdm(data):
#         image_path = os.path.join(image_dir, item["image"])
#         ocr_text = ocr_image(ocr_model, image_path)
#         # print(f"Image: {item['image']}, OCR Text: {ocr_text}")
#         item["ocr_text"] = ocr_text
    
#     # 按照 image_name 排序结果
#     # results = sorted(results, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))

#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

# if __name__ == "__main__":
#     # image_dir = "/data/home/yunhao/code/ocr/fox_data/from_text/"
#     # save_path = "/data/home/yunhao/code/ocr/results/other/paddle_ocr_from_text.json"
#     # data_file = "../fox_data/data.json"
#     # main(image_dir, save_path, data_file)
    
#     image_dir = "/data/home/yunhao/code/ocr/fox_data/random/"
#     save_path = "/data/home/yunhao/code/ocr/results/other/paddle_ocr_random.json"
#     data_file = "../fox_data/random.json"
#     main(image_dir, save_path, data_file)
    
    