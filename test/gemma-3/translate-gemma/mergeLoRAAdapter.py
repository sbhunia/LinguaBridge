import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -------------------------
# PATHS
# -------------------------
base_model_name = "google/translategemma-4b-it"
adapter_path = "./gemma_finetuned_myaamia"   # where your LoRA adapter is saved
output_path = "./gemma_myaamia_full_model"   # where merged model will be saved

# -------------------------
# LOAD TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# -------------------------
# LOAD BASE MODEL
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# -------------------------
# LOAD LORA ADAPTER
# -------------------------
model = PeftModel.from_pretrained(model, adapter_path)

# -------------------------
# MERGE LORA INTO BASE MODEL
# -------------------------
model = model.merge_and_unload()

# -------------------------
# SAVE MERGED MODEL
# -------------------------
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Merged model saved to:", output_path)