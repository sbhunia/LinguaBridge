from transformers import T5ForConditionalGeneration

hf_model = T5ForConditionalGeneration.from_pretrained("/home/yogesh/LinguaBridge/test/t5-small-60m/t5-fine-tuned/checkpoint-8340")
for name, p in hf_model.named_parameters():
    print(name, p.shape)
