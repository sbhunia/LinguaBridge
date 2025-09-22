from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_checkpoint = "/home/yogesh/LinguaBridge/test/marianfinetune-en-fr-kde4/marian-finetuned-opusbooks-en-to-fr/checkpoint-8340"
print(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, local_files_only=True)


translator = pipeline("translation", model=model, tokenizer=tokenizer)

# Test translation
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
