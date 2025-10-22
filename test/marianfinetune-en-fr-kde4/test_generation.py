from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_checkpoint = "/home/yogesh/LinguaBridge/test/marianfinetune-en-fr-kde4/myaamia/40-epocs-multi-eval/marian_finetuned_myaamia/checkpoint-33440"

print(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, local_files_only=True)


translator = pipeline("translation", model=model, tokenizer=tokenizer)

# Test translation
result = translator("kiišikwi iihkipakinki")
print(result[0]['translation_text'])
