from transformers import AutoProcessor, AutoModelForImageTextToText

p = r"C://Users//zhangrx59//.cache//huggingface//hub//Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(p)
model = AutoModelForImageTextToText.from_pretrained(p)
print("OK")
