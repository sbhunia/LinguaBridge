from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_checkpoint = "/home/chaudhy/LinguaBridge/test/marianfinetune-en-fr-kde4/myaamia/20-muti-updated/marian_finetuned_myaamia/checkpoint-2100"
print(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, local_files_only=True)

translator = pipeline("translation", model=model, tokenizer=tokenizer)

# Test translation
result = translator(">>mya<<Last summer")
print(result[0]['translation_text'])
