from paddleocr import PaddleOCR
import os
import json
from tqdm import tqdm


# PaddleOCR-V5

def ocr_image(ocr_model, image_path):
    result = ocr_model.predict(input=image_path,)
    
    rec_texts = result[0]["rec_texts"]
    ocr_text = "\n".join(rec_texts)
    
    # for res in result:
    #     print(res["rec_texts"])
    
    return ocr_text

def main(image_dir, save_path, data_file):
    
    # image_dir = "/data/home/yunhao/code/ocr/fox_data/test/"
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 初始化 PaddleOCR 实例
    ocr_model = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device='gpu:0',
        use_tensorrt=True,
        cpu_threads=64,
        rec_batch_num=32,
        )
    
    for item in tqdm(data):
        image_path = os.path.join(image_dir, item["image"])
        ocr_text = ocr_image(ocr_model, image_path)
        # print(f"Image: {item['image']}, OCR Text: {ocr_text}")
        item["ocr_text"] = ocr_text
    
    # 按照 image_name 排序结果
    # results = sorted(results, key=lambda x: int(x["image"].split("_")[-1].split(".")[0]))

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # image_dir = "/data/home/yunhao/code/ocr/fox_data/from_text/"
    # save_path = "/data/home/yunhao/code/ocr/results/other/paddle_ocr_from_text.json"
    # data_file = "../fox_data/data.json"
    # main(image_dir, save_path, data_file)
    
    image_dir = "/data/home/yunhao/code/ocr/fox_data/random/"
    save_path = "/data/home/yunhao/code/ocr/results/other/paddle_ocr_random.json"
    data_file = "../fox_data/random.json"
    main(image_dir, save_path, data_file)
    
    