from transformers import MBart50Tokenizer

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)

print(tokenizer.tokenize("Hello world!"))

# Output:
# ['▁Hello', '▁world', '!']