#!/usr/bin/env python3
"""
PaddleOCR-VL 官方API批量调用脚本（优化版：支持并行提交和轮询）
需要先安装: pip install requests tqdm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue
import threading

try:
    import requests
except ImportError:
    print("请先安装 requests: pip install requests")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
SUPPORTED_MODEL = "PaddleOCR-VL-1.6"


class PaddleOCRAPI:
    """PaddleOCR-VL API 调用封装（并行版）"""

    def __init__(self, token, model=SUPPORTED_MODEL, poll_interval=3, max_concurrent=5):
        self.token = token
        self.model = model
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.headers = {"Authorization": f"bearer {token}"}
        self._lock = Lock()
        self._submitted_count = 0
        self._completed_count = 0

    def submit_job(self, file_path, optional_payload=None, max_retries=5):
        """提交一个OCR任务，支持429重试"""
        if optional_payload is None:
            optional_payload = {}

        for attempt in range(max_retries):
            try:
                if file_path.startswith("http"):
                    payload = {
                        "fileUrl": file_path,
                        "model": self.model,
                        "optionalPayload": optional_payload
                    }
                    response = requests.post(
                        JOB_URL, json=payload,
                        headers={"Authorization": f"bearer {self.token}", "Content-Type": "application/json"},
                        timeout=60
                    )
                else:
                    if not os.path.exists(file_path):
                        return None, f"File not found: {file_path}"

                    data = {"model": self.model, "optionalPayload": json.dumps(optional_payload)}
                    with open(file_path, "rb") as f:
                        response = requests.post(JOB_URL, headers=self.headers, data=data, files={"file": f}, timeout=60)

                # 检测429频率限制
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 5  # 递增等待: 5, 10, 15, 20, 25秒
                    print(f"  [频率限制] 等待 {wait_time}秒后重试 ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    return None, f"HTTP {response.status_code}: {response.text[:200]}"

                job_id = response.json()["data"]["jobId"]
                with self._lock:
                    self._submitted_count += 1
                return job_id, None

            except requests.exceptions.Timeout:
                print(f"  [超时] 重试 ({attempt+1}/{max_retries})")
                time.sleep(3)
            except Exception as e:
                return None, f"Error: {e}"

        return None, f"提交失败，已重试 {max_retries} 次"

    def poll_job(self, job_id):
        """轮询单个任务状态"""
        headers = {"Authorization": f"bearer {self.token}"}
        while True:
            response = requests.get(f"{JOB_URL}/{job_id}", headers=headers)
            if response.status_code != 200:
                return None, f"Polling failed: HTTP {response.status_code}"

            data = response.json()["data"]
            state = data["state"]

            if state == "done":
                try:
                    json_url = data["resultUrl"]["jsonUrl"]
                    with self._lock:
                        self._completed_count += 1
                    return json_url, None
                except KeyError:
                    return None, "No result URL"
            elif state == "failed":
                with self._lock:
                    self._completed_count += 1
                return None, f"Job failed: {data.get('errorMsg', 'Unknown')}"
            else:
                # pending or running
                yield state

            time.sleep(self.poll_interval)

    def download_result(self, json_url):
        """下载JSON结果"""
        response = requests.get(json_url)
        if response.status_code != 200:
            return None, f"Download failed: HTTP {response.status_code}"

        text = response.text.strip()
        if not text:
            return [], None

        results = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                result = parsed.get("result", parsed)
                results.append(result)
            except json.JSONDecodeError:
                continue

        return results if results else None, None

    def ocr_single(self, file_path, optional_payload=None):
        """完整的OCR流程"""
        job_id, error = self.submit_job(file_path, optional_payload)
        if error:
            return None, error

        # 轮询
        try:
            for state in self.poll_job(job_id):
                pass  # 等待完成
        except Exception as e:
            return None, str(e)

        # 获取结果URL
        json_url, error = self.poll_job(job_id).__next__() if False else None, None
        # 直接重新轮询获取URL
        headers = {"Authorization": f"bearer {self.token}"}
        while True:
            response = requests.get(f"{JOB_URL}/{job_id}", headers=headers)
            data = response.json()["data"]
            if data["state"] == "done":
                json_url = data["resultUrl"]["jsonUrl"]
                break
            elif data["state"] == "failed":
                return None, data.get("errorMsg", "Unknown")
            time.sleep(self.poll_interval)

        results, error = self.download_result(json_url)
        if error:
            return None, error

        return results, None

    def _extract_text(self, results):
        """从OCR结果中提取纯文本"""
        texts = []
        for result in results:
            for layout_result in result.get("layoutParsingResults", []):
                md = layout_result.get("markdown", {})
                if isinstance(md, dict):
                    text = md.get("text", "")
                    if text:
                        texts.append(text)
                    for key in ["content", "html", "latex"]:
                        if key in md:
                            texts.append(str(md[key]))
        return "\n\n".join(texts)

    def process_batch_optimized(self, items, optional_payload=None, max_workers=3, poll_interval=3, submit_interval=0.5):
        """
        优化版批量处理：先提交所有任务，再并行轮询
        items: [(file_path, item_data), ...]
        """
        print(f"提交 {len(items)} 个任务 (并发={max_workers}, 间隔={submit_interval}s)...")

        # 第一步：并行提交所有任务（带延迟避免429）
        submitted_jobs = []  # [(job_id, file_path, item_data), ...]

        def submit_with_delay(path, item):
            time.sleep(submit_interval)  # 提交间隔
            return self.submit_job(path, optional_payload), path, item

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(submit_with_delay, path, item): (path, item)
                for path, item in items
            }

            for future in as_completed(futures):
                path, item = futures[future]
                try:
                    (job_id, error), _, _ = future.result()
                    if error:
                        print(f"  提交失败 {path}: {error}")
                    else:
                        submitted_jobs.append((job_id, path, item))
                except Exception as e:
                    print(f"  提交异常 {path}: {e}")

        print(f"成功提交 {len(submitted_jobs)} 个任务，开始轮询...")

        if not submitted_jobs:
            return []

        # 第二步：并行轮询所有任务
        results = []
        completed_count = 0
        total = len(submitted_jobs)

        def poll_single(job_id, path, item):
            headers = {"Authorization": f"bearer {self.token}"}
            while True:
                try:
                    response = requests.get(f"{JOB_URL}/{job_id}", headers=headers, timeout=30)
                    data = response.json()["data"]
                    state = data["state"]

                    if state == "done":
                        json_url = data["resultUrl"]["jsonUrl"]
                        ocr_results, error = self.download_result(json_url)
                        if error:
                            return {**item, "model_name": self.model, "error": error, "ocr_text": ""}
                        text = self._extract_text(ocr_results) if ocr_results else ""
                        return {**item, "model_name": self.model, "ocr_text": text}
                    elif state == "failed":
                        return {
                            **item,
                            "model_name": self.model,
                            "error": data.get("errorMsg", "Unknown"),
                            "ocr_text": "",
                        }

                    time.sleep(poll_interval)
                except Exception as e:
                    return {**item, "model_name": self.model, "error": str(e), "ocr_text": ""}

        # 使用线程池并行轮询
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(poll_single, job_id, path, item): (job_id, path)
                for job_id, path, item in submitted_jobs
            }

            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理中"):
                job_id, path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "ocr_text": ""})

        return results


def build_parser():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL官方API批量处理（优化版）")
    parser.add_argument("--token", default=os.environ.get("PADDLEOCR_TOKEN", ""))
    parser.add_argument("--model", default=SUPPORTED_MODEL, choices=[SUPPORTED_MODEL])
    parser.add_argument("--data-root", default="fox_data/deepseek_mode_images")
    parser.add_argument("--datasets", nargs="+", default=["from_text"])
    parser.add_argument("--modes", nargs="+", default=["tiny", "small", "base"])
    parser.add_argument("--output-dir", default="results/other/paddleocr_vl_1.6_api_deepseek_modes")
    parser.add_argument("--model-label", default="paddleocr_vl_1.6_api")
    parser.add_argument("--max-workers", type=int, default=3, help="并发数（建议3-5）")
    parser.add_argument("--poll-interval", type=int, default=3, help="轮询间隔(秒)")
    parser.add_argument("--submit-interval", type=float, default=0.5, help="提交任务间隔(秒)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    if not args.token:
        print("错误: 请提供 --token 或设置环境变量 PADDLEOCR_TOKEN")
        sys.exit(1)

    api = PaddleOCRAPI(args.token, args.model, args.poll_interval, args.max_workers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }

    for dataset in args.datasets:
        for mode in args.modes:
            data_dir = Path(args.data_root) / dataset / mode
            data_file = data_dir / "data.json"
            images_dir = data_dir / "images"
            output_file = output_dir / f"{args.model_label}_{dataset}_{mode}.json"

            if not data_file.exists():
                print(f"跳过: {data_file} 不存在")
                continue

            with open(data_file) as f:
                data = json.load(f)

            if args.limit:
                data = data[:args.limit]

            # 检查已完成的
            completed = set()
            if args.resume and output_file.exists():
                with open(output_file) as f:
                    existing = json.load(f)
                    completed = {item.get("image") or item.get("processed_image")
                                for item in existing if item.get("ocr_text") and not item.get("error")}
                print(f"{dataset}/{mode}: 已有 {len(completed)} 条")

            todo = [d for d in data if (d.get("image") or d.get("processed_image")) not in completed]
            if not todo:
                print(f"{dataset}/{mode}: 全部完成")
                continue

            print(f"\n处理 {dataset}/{mode}: {len(todo)}/{len(data)} 张")

            # 收集图片路径
            items = []
            for item in todo:
                img_name = item.get("image") or item.get("processed_image")
                img_path = images_dir / img_name
                if img_path.exists():
                    items.append((str(img_path), item))

            if not items:
                continue

            # 优化版批量处理
            start_time = time.time()
            results = api.process_batch_optimized(
                items, optional_payload,
                max_workers=args.max_workers,
                poll_interval=args.poll_interval,
                submit_interval=args.submit_interval
            )
            elapsed = time.time() - start_time

            # 保存结果
            existing = []
            if args.resume and output_file.exists():
                with open(output_file) as f:
                    existing = json.load(f)

            merged = existing + results
            seen = {}
            for item in merged:
                key = item.get("processed_image") or item.get("image")
                seen[key] = item
            final_results = list(seen.values())

            with open(output_file, "w") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)

            errors = sum(1 for r in results if r.get("error"))
            print(f"  完成: {len(results)-errors}/{len(results)}, 耗时: {elapsed:.0f}秒 ({elapsed/len(results):.1f}秒/张)")


if __name__ == "__main__":
    main()
