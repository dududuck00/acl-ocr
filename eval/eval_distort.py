import json
import argparse
import nltk
from nltk.metrics import precision, recall, f_measure
import numpy as np
import jieba
import re
from nltk.translate import meteor_score
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from tqdm import tqdm

# ✅ 在主进程启动时预加载 WordNet，避免多线程竞争
# from nltk.corpus import wordnet
# _ = wordnet.ensure_loaded()  # 强制加载


def contain_chinese_string(text):
    """
    使用正则表达式检查字符串中是否包含中文字符
    """
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))

def cal_per_metrics(image, pred, gt):
    """
    比较预测文本和真实文本，计算各种评估指标
    计算指标包括：BLEU、METEOR、F-measure、Precision、Recall、编辑距离
    适用于中英文混合文本的评估
    传入参数：
    - pred: 预测文本字符串
    - gt: 真实文本字符串
    返回值：
    - metrics: 包含各项评估指标的字典
    """

    metrics = {}
    metrics["image"] = image

    # 根据文本内容选择分词方式
    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    # 计算各项指标
    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    
    # print(f"Image: {image}, BLEU: {metrics['bleu']:.4f}, METEOR: {metrics['meteor']:.4f}, F-measure: {metrics['f_measure']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, Edit Distance: {metrics['edit_dist']:.4f}")
    
    return metrics

def eval(predicts, max_workers=8):
    """
    对预测结果文件进行评估，计算整体的评估指标
    批量评估OCR预测结果并计算平均指标
    预测结果格式为：
    [
        {
            "label": "预测文本",
            "answer": "真实标签"
        },
        ...
    ]
    输出整体评估指标的平均值
    """
    print(f"开始评估 {len(predicts)} 个样本，使用 {max_workers} 个线程...")
    eval_results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，并保存 future 到 image 的映射
        future_to_pred = {
            executor.submit(cal_per_metrics, pred["image"], pred["ocr_text"], pred["distorted_text"]): pred
            for pred in predicts
        }
        # 使用 as_completed 配合 tqdm 显示进度
        for future in tqdm(as_completed(future_to_pred), total=len(predicts), desc="评估进度", unit="样本"):
            try:
                result = future.result()
                eval_results.append(result)
            except Exception as e:
                pred = future_to_pred[future]
                print(f"\n处理 {pred['image']} 时出错: {e}")
    
    # 按照img名称重新排序结果
    eval_results = sorted(
        eval_results,
        key=lambda x: int(x["image"].split("_")[-1].split(".")[0])
    )
    
    
    mean_dict = {}
    mean_dict["eval question num"] = len(eval_results)
    
    # ✅ 只初始化数值类型的指标
    for k, v in eval_results[0].items():
        if k != "image":
            mean_dict[k] = 0.0

    # ✅ 累加时排除 "image" 字段
    for each in eval_results:
        for k, v in each.items():
            if k != "image":
                mean_dict[k] += v
    
    # 计算平均值
    for k in list(mean_dict.keys()):
        if k == "eval question num":
            continue
        mean_dict[k] /= len(eval_results)
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    for k, v in mean_dict.items():
        if k == "eval question num":
            print(f"{k}: {int(v)}")
        else:
            print(f"{k}: {v:.4f}")
    print("="*60)
        
    return eval_results , mean_dict

def main(args):    
    with open(args.predict_file, "r", encoding="utf-8") as f:
        predict_data = json.load(f)
    
    # TODO debug only
    # predict_data = predict_data[:8] # 测试代码, 运行时删除此行
        
    results, mean_dict = eval(predict_data, max_workers=args.max_workers)
    # 将image相同的评估结果保存到原始预测结果中
    for item in predict_data:
        image = item["image"]
        for res in results:
            if res["image"] == image:
                item.update(res)
                break
    
    # 额外添加整体评估结果
    predict_data.append({"overall_metrics": mean_dict})

    # 保存结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(predict_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR Predictions")
    parser.add_argument("--predict_file", type=str, required=True, help="Path to the OCR predictions JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results JSON file")
    parser.add_argument("--max_workers", type=int, default=64, help="Number of parallel workers for evaluation")
    
    args = parser.parse_args()
    main(args)
    
# python eval/eval.py --predict_file ./results/fox/en_png_tiny.json --output_file ./results/eval/en_png_tiny_eval.json