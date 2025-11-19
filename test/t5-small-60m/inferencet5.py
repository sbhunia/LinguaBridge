import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
from transformers import AutoModelForSeq2SeqLM, MT5Tokenizer

model_path = "/home/chaudhy/LinguaBridge/test/t5-small-60m/myaamia/test3/mt5_finetuned_myaamia/final_model"
tokenizer = MT5Tokenizer.from_pretrained(
    model_path
    )
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto"
    )

sentences = [
    "Are you thirsty?",
    "I think it will rain.",
    "I feel good, am happy",
    "Are you sore?",
    "He talks a lot",
    "I am a little bit sick",
    "You are too sick",
    "I am always working",
    "I will be light weight",
    "I slept well"
]

print("--- Starting Translation Loop ---")

for sentence in sentences:
    # 1. Prepare the text with the prefix
    text_to_translate = f"translate English to Myaamia: {sentence}"
    
    # 2. Tokenize
    input_ids = tokenizer(text_to_translate, return_tensors="pt").to(model.device)

    # 3. Generate (Removed static cache for safety in loops)
    output = model.generate(**input_ids)

    # 4. Decode and Print
    decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Input: {sentence}")
    print(f"Output: {decoded_text}")
    print("-" * 20)

#     --- Starting Translation Loop ---
# Input: Are you thirsty?
# Output: aahkwaahkiyani-nko
# --------------------
# Input: I think it will rain.
# Output: iihkiinkweehsiiwaani
# --------------------
# Input: I feel good, am happy
# Output: eelwaaphkiaani
# --------------------
# Input: Are you sore?
# Output: eewiihkiyani-nko
# --------------------
# Input: He talks a lot
# Output: aalhsoohkiici
# --------------------
# Input: I am a little bit sick
# Output: eehšiihki eehšiihki eehšiih
# --------------------
# Input: You are too sick
# Output: eewiihkweeyani
# --------------------
# Input: I am always working
# Output: eehšiihki eehšiihki eehšiih
# --------------------
# Input: I will be light weight
# Output: eehšiihkiinkweehšinaani
# --------------------
# Input: I slept well
# Output: eehšiihkiwiaani
# --------------------