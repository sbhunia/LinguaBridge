import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

# --- 1. Configuration ---

# ⬇️ *** SET THIS TO YOUR CHECKPOINT PATH *** ⬇️
# This is the path to the checkpoint you want to test
# (e.g., "llama-3.2-1b-en-mia-translation/checkpoint-100")
ADAPTER_PATH = "/home/chaudhy/LinguaBridge/test/llama-3/Gen-AI/llama-3.2-1b-fino1-finetuned/checkpoint-619"

# This is the base model you used for training
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Set which GPU to use for inference
INFERENCE_GPU = "5"

# --- 2. Set Up Device ---
os.environ["CUDA_VISIBLE_DEVICES"] = INFERENCE_GPU
DEVICE = "cuda"

# --- 3. Load Base Model ---
print(f"Loading base model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"": DEVICE}  # Load the whole model onto our target GPU
)

# --- 4. Load Adapter and Tokenizer ---
print(f"Loading adapter and tokenizer from: {ADAPTER_PATH}")
# Load the PEFT adapter on top of the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# Load the tokenizer from the adapter path
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Set model to evaluation mode
model = model.eval()

# --- 5. Prepare the Prompt ---
# This prompt must match the format you used in training
system_prompt = "You are a finance expert."

# ⬇️ *** TEST WITH YOUR OWN SENTENCE HERE *** ⬇️
english_text = "Yes, it is sunny"

# Create the list of messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"what are debentures?"},
]

# Apply the chat template for inference
# add_generation_prompt=True adds the <|start_header_id|>assistant<|end_header_id|>
# tokens, which tells the model to start generating.
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# --- 6. Run Inference ---
print("\n--- Prompt ---")
print(prompt)

# Tokenize the prompt
inputs = tokenizer(
    prompt, 
    return_tensors="pt", 
    padding=True, 
    truncation=True
).to(DEVICE)

# Generate the translation
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=75,  # Max length of a Myaamia translation
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode the output
# We need to decode *only* the newly generated tokens
input_length = inputs.input_ids.shape[1]
new_tokens = output_tokens[:, input_length:]
prediction = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

print("\n--- Prediction ---")
print(prediction.strip())