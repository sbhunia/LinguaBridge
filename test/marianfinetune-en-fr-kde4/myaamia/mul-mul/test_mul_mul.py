
from transformers import MarianMTModel, MarianTokenizer

# Model trained on multiple source languages → multiple target languages
# Example: multilingual to Arabic (arb)
model_name = "Helsinki-NLP/opus-mt-mul-mul"  # Tatoeba Challenge model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Prepend the desired output language code (3-letter ISO 639-3)
src_texts = ["arb>> Hello, how are you today?"]

# Tokenize and translate
inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**inputs)

# Decode and print result
translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(translated_texts[0])
