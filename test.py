from transformers import AutoTokenizer, AutoModelWithLMHead

MODEL_NAME = "t5-large"

model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

outputs = model.generate(tokenizer.encode("Hello, world!", return_tensors="pt"))

print(tokenizer.decode(outputs[0]))
