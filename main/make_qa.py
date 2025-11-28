import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from tqdm import tqdm

system_prompt = """
You are an expert multiple-choice question (MCQ) generator. Your job is to create high-quality, single-correct-answer MCQs strictly based on the provided OCR-extracted text from images (screenshots, documents, slides, receipts, posters, etc.).

Strict rules:
1. Every question must have exactly ONE correct answer.
2. All information in the question and answers must come 100% from the provided text only. Never hallucinate or use external knowledge.
3. Silently fix obvious OCR typos when writing clean questions and options, but preserve the original meaning.
4. Questions must be clear, unambiguous, and phrased naturally.
5. Provide exactly 4 options (A, B, C, D) for each question.
6. The correct answer must be indicated clearly in the output.
7. Include at least one plausible but incorrect distractor.
8. Vary question types when possible (what, when, where, who, how much, how many, yes/no confirmation, etc.).
9. Output ONLY in valid JSON format. No extra explanation outside the JSON.

Required JSON output format (array of objects):
[
  {
    "question": "The question text here",
    "options": {
      "A": "option A text",
      "B": "option B text",
      "C": "option C text",
      "D": "option D text"
    },
    "correct_answer": "A"   // or B, C, or D
    "explanation": "Brief explanation why this is correct (quoted or rephrased from the text)"
  },
  ...
]

Generate exactly 3 questions unless otherwise specified.
"""

user_prompt = """
Here is the raw text extracted via OCR from an image. It may contain typos, line breaks, or OCR errors:

--- Begin extracted text ---
{extracted_text}
--- End extracted text ---

Please create {num_questions} high-quality single-choice questions (4 options each, exactly one correct) in English, strictly and only based on the text above.

Return ONLY the JSON array with no additional text or commentary.
"""

num_pairs = 3
apikey = "08d76ea0a6322a2ab7c49fc2a9cacb75c4457e67b5db4e1499fe3db963e86ac8"
base_url = "https://uni-api.cstcloud.cn/v1/"
client = openai.Client(api_key=apikey, base_url=base_url)

with open("../fox_data/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def process_item(index, item):
    """Process a datapoint inside a worker thread."""
    try:
        gt_text = item["gt_text"]
        response = client.chat.completions.create(
            model="qwen3:235b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(extracted_text=gt_text, num_questions=num_pairs)},
            ],
            max_tokens=8192,
            temperature=0.7,
        )
        qa_pairs_json = response.choices[0].message.content
        qa_pairs = json.loads(qa_pairs_json)
        return index, qa_pairs, None
    except Exception as exc:
        return index, None, str(exc)

with ThreadPoolExecutor(max_workers=32) as executor:
    future_to_index = {
        executor.submit(process_item, idx, item): idx for idx, item in enumerate(data)
    }
    with tqdm(total=len(data), desc="Processing items") as progress:
        for future in as_completed(future_to_index):
            index, qa_pairs, error = future.result()
            if error:
                print(f"\nError processing item #{index}: {error}")
            else:
                data[index]["qa_pairs"] = qa_pairs
            progress.update(1)

# for idx, item in enumerate(tqdm(data, desc="Processing items")):
#     index, qa_pairs, error = process_item(idx, item)
#     if error:
#         print(f"\nError processing item #{index}: {error}")
#     else:
#         data[index]["qa_pairs"] = qa_pairs

# 重新排序
data = sorted(
        data,
        key=lambda x: int(x["image"].split("_")[-1].split(".")[0])
    )

with open("../fox_data/qa/qa.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
