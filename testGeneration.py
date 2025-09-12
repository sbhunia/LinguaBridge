from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Load tokenizer (runs on CPU)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Load model (can run on CUDA if available)
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # ← THIS puts the model on the GPU if available

# Encode text using tokenizer (CPU)
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)  # ← move input tensors to GPU

# Generate translation or output (GPU if available)
output_tokens = model.generate(**inputs)

# Decode output (CPU)
translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(translated_text)
