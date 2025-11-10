import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 1. Configuration ---

# The base model ID (the "Instruct" version you trained on)
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Path to your saved adapter (the output of your training)
# ⬇️ *** SET THIS TO YOUR ADAPTER PATH *** ⬇️
# (e.g., "llama-3.2-1b-en-mia-translation/checkpoint-538")
ADAPTER_PATH = "/home/chaudhy/LinguaBridge/test/llama-3/myaamia/test2/llama-3.2-1b-en-mia-translation/checkpoint-538"

# Path for the new, merged model
# ⬇️ *** SET THIS TO YOUR FINAL OUTPUT PATH *** ⬇️
MERGED_OUTPUT_PATH = "llama-3.2-1b-en-mia-translation-merged"

# --- 2. Load Base Model and Tokenizer ---
print(f"Loading base model from: {MODEL_ID}")
# Load the tokenizer from the *base model* to get the chat template
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,  # Use the same dtype as training
    device_map="auto",
)

# --- 3. Load and Merge the PEFT Adapter ---
print(f"Loading adapter from: {ADAPTER_PATH}")
# Load the PEFT adapter
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging adapter into the base model...")
# ⬇️ --- THIS IS THE CRITICAL STEP --- ⬇️
# This combines the adapter weights with the base model weights
model = model.merge_and_unload()
print("Merge complete.")

# --- 4. Save the Merged Model ---
print(f"Saving merged model and tokenizer to: {MERGED_OUTPUT_PATH}")
model.save_pretrained(MERGED_OUTPUT_PATH)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

print("✅ Merged model saved successfully!")