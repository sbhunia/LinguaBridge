# from transformers import AutoTokenizer

# print("Starting..")
# tokenizer = AutoTokenizer.from_pretrained("/home/yogesh/LinguaBridge/test/marianfinetune-en-fr-kde4/myaamia/en-mul/40-epocs/marian_finetuned_myaamia/checkpoint-33440",use_fast=True)

# # Save as a single tokenizer.json file
# tokenizer.save_pretrained("/home/yogesh/LinguaBridge/test/marianfinetune-en-fr-kde4/myaamia/en-mul/40-epocs/marian_finetuned_myaamia/tokenizerFast")

# print("Done.")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/home/yogesh/LinguaBridge/test/t5-small-60m/tokenizer", use_fast=True)

tok.save_pretrained("/home/yogesh/LinguaBridge/test/t5-small-60m/tokenizerFast2")
print("Done.")