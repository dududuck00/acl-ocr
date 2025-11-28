# python eval/eval.py --predict_file ./results/fox/en_png_tiny.json --output_file ./results/eval/en_png_tiny_eval.json
# python eval/eval.py --predict_file ./results/fox/en_png_small.json --output_file ./results/eval/en_png_small_eval.json
# python eval/eval.py --predict_file ./results/fox/distort_tiny.json --output_file ./results/eval/distort_tiny_eval.json
# python eval/eval.py --predict_file ./results/fox/distort_small.json --output_file ./results/eval/distort_small_eval.json
# python eval/eval.py --predict_file ./results/fox/from_text_tiny.json --output_file ./results/eval/from_text_tiny_eval.json
# python eval/eval.py --predict_file ./results/fox/from_text_small.json --output_file ./results/eval/from_text_small_eval.json

# 评估原始分辨率
python eval/eval.py --predict_file ./results/fox/en_png_raw.json --output_file ./results/eval/en_png_raw_eval.json
python eval/eval.py --predict_file ./results/fox/distort_raw.json --output_file ./results/eval/distort_raw_eval.json
python eval/eval.py --predict_file ./results/fox/from_text_raw.json --output_file ./results/eval/from_text_raw_eval.json

python eval/eval.py --predict_file ./results/test_10_raw.py --output_file ./results/test_10_raw_eval.py 
python eval/eval.py --predict_file ./results/test_10_tiny.py --output_file ./results/test_10_tiny_eval.py 
python eval/eval.py --predict_file ./results/test_10_small.py --output_file ./results/test_10_small_eval.py 