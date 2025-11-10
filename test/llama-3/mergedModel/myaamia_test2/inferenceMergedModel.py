import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. Configuration ---

# ⬇️ *** NOW, JUST POINT TO YOUR NEW MERGED MODEL *** ⬇️
MERGED_MODEL_PATH = "llama-3.2-1b-en-mia-translation-merged"

# Set which GPU to use for inference
INFERENCE_GPU = "5"

# --- 2. Set Up Device ---
os.environ["CUDA_VISIBLE_DEVICES"] = INFERENCE_GPU
DEVICE = "cuda"

# --- 3. Load Your Merged Model ---
# No more PeftModel! This is just a standard transformers model.
print(f"Loading merged model from: {MERGED_MODEL_PATH}")

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map={"": DEVICE}
)
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
model = model.eval()

# --- 4. Prepare and Run Inference (Same as before) ---
system_prompt = "You are an expert translator. Translate the given English text into Myaamia."
english_text = "Yes, it is sunny"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"translate english to myaamia: {english_text}"},
]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

print("\n--- Prompt ---")
print(prompt)

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=75,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

input_length = inputs.input_ids.shape[1]
new_tokens = output_tokens[:, input_length:]
prediction = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

print("\n--- Prediction ---")
print(prediction.strip())