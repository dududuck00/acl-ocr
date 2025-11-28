import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from tqdm import tqdm
from textwrap import dedent

system_prompt = """
You are a strict, impartial single-choice question validator. Your only job is to check whether a given single-choice question is 100% correct and reasonable when compared to the original source text.

Validation criteria (all must be satisfied for "Yes"):
1. The question must be clearly answerable from the original text only.
2. The marked "correct_answer" must be factually correct and the only fully correct option.
3. None of the distractors (wrong options) can be correct or partially correct.
4. All information in question and options must match the original text.
5. No external knowledge is introduced.
6. Distractors must be plausible but unambiguously wrong.

Output rules:
- If the single-choice question is perfectly correct and reasonable → output ONLY "Yes" in the following JSON format:
{
  "verdict": "Yes"
}
- If there is ANY problem → output in the following JSON format ONLY:

{
  "verdict": "No",
  "issue": "Clear and concise description of what is wrong (in English)",
  "correct_answer_should_be": "A"   // or B, C, D, or "none" if all options are wrong
  "explanation": "Brief explanation why the original 'correct_answer' is wrong and why the suggested answer is correct (quoted or rephrased from the text)"
}

Do not output anything else. Never add apologies or extra text.
"""

num_pairs = 3
apikey = "08d76ea0a6322a2ab7c49fc2a9cacb75c4457e67b5db4e1499fe3db963e86ac8"
base_url = "https://uni-api.cstcloud.cn/v1/"
client = openai.Client(api_key=apikey, base_url=base_url)

with open("../fox_data/qa/qa_checked.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = "gpt-oss-120b"
output_file = "../fox_data/qa/check/gptoss_recheck.json"

# model = "qwen3:235b"
# output_file = "../fox_data/qa/check/qwen3_recheck.json"

# model = "deepseek-v3:671b"
# output_file = "../fox_data/qa/check/dpskv3_recheck.json"

# item = data[0]
# process_item(0, item, model)

error_items = []

def process_item(index, item, model):
    """Process a datapoint inside a worker thread."""
    original_extracted_text = item["gt_text"]
    num_pairs = len(item["qa_pairs"])
    questions_json_array = json.dumps(item["qa_pairs"])
    user_prompt = dedent("""
Source text:
--- Begin source text ---\n"""
+ original_extracted_text + "\n"
"""--- End source text ---

Now here are 3 single-choice questions to validate. For each one, check if it is 100% correct and reasonable according to the rules.

Questions:\n""" +
questions_json_array + "\n"
+ """
# Example of the expected format for the array above:
[
  {
    "question": "What is the release date of the product?",
    "options": {"A": "March 15, 2024", "B": "April 2024", "C": "March 2025", "D": "January 15, 2024"},
    "correct_answer": "A",
    "explanation": "The text says Release Date: March 15, 2024"
  },
]

For each question, output exactly in JSON format as specified in the system instructions, and your final verdict for all questions should also be in a JSON array like this:
[
  {
    "verdict": "Yes"
  },
  {
    "verdict": "No",
    "issue": "The correct answer is not supported by the text.",
    "correct_answer_should_be": "B",
    "explanation": "The text states that the release date is April 2024, not March 15, 2024."
  },
  ...
]
""")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=8192,
            temperature=0.7,
        )
        check_results_json = response.choices[0].message.content
        check_results = json.loads(check_results_json)
        return index, check_results, None
    except Exception as exc:
        print("Exception processing item {}:".format(item.get("image", str(index))))
        error_items.append((index, item))
        return index, None, str(exc)

def process_batch(items):
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_index = {
            executor.submit(process_item, idx, item, model): idx for idx, item in enumerate(items)
        }
        with tqdm(total=len(items), desc="Processing items") as progress:
            for future in as_completed(future_to_index):
                index, check_results, error = future.result()
                if error:
                    print(f"\n第一次处理出错 item #{index}: {error}")
                else:
                    items[index]["check_results"] = check_results
                progress.update(1)
    return items

data = process_batch(data)

if error_items:
    print(f"\nRetrying {len(error_items)} failed items...")
    retry_items = list(error_items)
    error_items.clear()
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_index = {
            executor.submit(process_item, idx, item, model): idx for idx, item in retry_items
        }
        with tqdm(total=len(retry_items), desc="Reprocessing failed items") as progress:
            for future in as_completed(future_to_index):
                index, check_results, error = future.result()
                if error:
                    print(f"\n第二次处理出现异常 #{index}: {error}")
                else:
                    data[index]["check_results"] = check_results
                progress.update(1)
                
data = sorted(
  data,
  key=lambda x: int(x["image"].split("_")[-1].split(".")[0])
)      


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
