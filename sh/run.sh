# 这个脚本的作用是批量运行 vllm_batch_script.py 脚本，处理不同文件夹下的图像数据进行ocr识别，使用不同的模型模式（tiny 和 small）。
# 分别是原始图片、扭曲图片和从文本生成的图片。
# python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/en_png --mode tiny
# python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/en_png --mode small
# python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/distort/ --mode tiny
# python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/distort/ --mode small
# python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/from_text/ --mode tiny
# python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/from_text/ --mode small

# 原始分辨率，也就是不传入image_size 和 base_size 的版本
python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/en_png --mode raw
python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/distort/ --mode raw
python ./main/vllm_batch_script.py --input_folder /root/code/research/DeepSeek-OCR/fox_data/from_text/ --mode raw