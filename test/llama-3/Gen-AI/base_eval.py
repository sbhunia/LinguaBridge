import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
# from peft import PeftModel  <-- No longer needed
from tqdm import tqdm

# --- 1. CONFIGURATION ---
base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
# adapter_path = "./llama-3.2-1b-fino1-finetuned" # <-- REMOVED
dataset_name = "TheFinAI/Fino1_Reasoning_Path_FinQA"

# --- 2. LOAD BASE MODEL IN 16-BIT ---

print(f"Loading BASE model ({base_model_id}) in 16-bit (bfloat16)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)

# --- NO adapter loading ---
# model = PeftModel.from_pretrained(model, adapter_path) # <-- REMOVED
model.eval() # Set model to evaluation mode

# --- 3. LOAD DATASET AND METRIC ---
print("Loading dataset and metric...")
dataset = load_dataset(dataset_name, split="train")
dataset_splits = dataset.train_test_split(test_size=0.1)
test_data = dataset_splits["test"]

rouge = evaluate.load("rouge")

# --- 4. HELPER FUNCTIONS TO FORMAT PROMPTS ---

def create_eval_prompt(example):
    """Formats the evaluation prompt WITHOUT the answer."""
    system_prompt = (
        "You are a helpful and meticulous financial analyst. "
        "Answer the question based *only* on the provided context. "
        "Show your reasoning step-by-step to arrive at the final answer."
    )
    user_prompt = example.get('Open-ended Verifiable Question', '')
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def get_label(example):
    """Gets the ground-truth label for comparison."""
    reasoning = example.get('Complex_CoT', '')
    answer = example.get('Ground-True Answer', '')
    
    return (
        f"Reasoning:\n{reasoning}\n\n"
        f"Answer:\n{answer}"
    )

# --- 5. RUN EVALUATION ---
predictions = []
references = []

print(f"Starting evaluation on {len(test_data)} samples...")

# Use torch.no_grad() for massive memory savings
with torch.no_grad():
    # Iterate one by one (batch size of 1)
    for example in tqdm(test_data):
        prompt_text = create_eval_prompt(example)
        label_text = get_label(example)
        
        # Skip any corrupt data
        if not prompt_text or not label_text:
            continue
            
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        # Generate the response
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Adjust as needed
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the output and strip the prompt
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # The output contains the prompt, so we remove it
        # to get only the generated text.
        generated_text = decoded_output[len(prompt_text):].strip()

        predictions.append(generated_text)
        references.append(label_text)

# --- 6. COMPUTE METRICS ---
print("Computing ROUGE metrics...")
result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

print("\n--- BASE MODEL Evaluation Results ---")
print(f"Rouge-1: {result['rouge1'] * 100:.2f}%")
print(f"Rouge-2: {result['rouge2'] * 100:.2f}%")
print(f"Rouge-L: {result['rougeL'] * 100:.2f}%")
print("-----------------------------------")