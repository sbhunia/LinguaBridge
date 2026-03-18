import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# -------------------------
# CONFIG
# -------------------------
base_model_name = "google/translategemma-4b-it"
lora_adapter_path = "./gemma_finetuned_myaamia"

# -------------------------
# LOAD TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"  # Important for inference

# -------------------------
# LOAD BASE MODEL
# -------------------------
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch_dtype,
    device_map="auto"
)

# -------------------------
# LOAD LORA ADAPTER
# -------------------------
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval()

# -------------------------
# BUILD PIPELINE
# -------------------------
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# -------------------------
# PROMPT TEMPLATE
# -------------------------
def build_prompt(english_text):
    return (
        "<start_of_turn>user\n"
        "You are a professional English (en) to Myaamia (mia) translator. "
        "Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Myaamia grammar, vocabulary, and cultural sensitivities. "
        "Produce only the Myaamia translation, without any additional explanations or commentary. "
        "Please translate the following English text into Myaamia:\n\n"
        f"{english_text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

# -------------------------
# TRANSLATION FUNCTION
# -------------------------
def translate_en_to_mia(text, max_new_tokens=128):
    prompt = build_prompt(text)

    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,        # deterministic
        temperature=0.0,
        top_p=1.0,
        return_full_text=False
    )

    generated = outputs[0]["generated_text"]

    # Clean at end_of_turn if model outputs it
    translation = generated.split("<end_of_turn>")[0].strip()

    return translation


# -------------------------
# INTERACTIVE CLI
# -------------------------
if __name__ == "__main__":
    print("\nEnglish → Myaamia Translator\n")
    while True:
        text = input("Enter English sentence (or 'exit'): ")
        if text.lower() == "exit":
            break

        translation = translate_en_to_mia(text)
        print(f"\nMyaamia: {translation}\n")