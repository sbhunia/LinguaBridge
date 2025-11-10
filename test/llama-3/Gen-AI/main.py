import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# 1. Configuration
model_id = "meta-llama/Llama-3.2-1B-Instruct"
dataset_name = "TheFinAI/Fino1_Reasoning_Path_FinQA"
output_dir = "llama-3.2-1b-fino1-finetuned"

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 3. Load Model
# Use 'dtype' instead of 'torch_dtype' to fix the deprecation warning
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,  # Fix: Use 'dtype'
    device_map="auto"
)

# 4. Load Dataset
dataset = load_dataset(dataset_name, split="train")
dataset_splits = dataset.train_test_split(test_size=0.1)

train_data = dataset_splits["train"]
test_data = dataset_splits["test"]

# 5. ⬇️ CORRECTED Formatting Function
# This function now uses the *actual* column names from your sample
def format_chat_prompt(example):
    """
    Formats the dataset example and returns a single, formatted string
    by applying the tokenizer's chat template.
    """
    system_prompt = (
        "You are a helpful and meticulous financial analyst. "
        "Answer the question based *only* on the provided context. "
        "Show your reasoning step-by-step to arrive at the final answer."
    )
    
    user_prompt = example.get('Open-ended Verifiable Question', '')
    reasoning = example.get('Complex_CoT', '')
    answer = example.get('Ground-True Answer', '')

    # Prevent training on empty/corrupt examples
    if not user_prompt or not reasoning or not answer:
        return ""  # Return an empty string to be skipped

    # Create the list of messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": (
            f"Reasoning:\n{reasoning}\n\n"
            f"Answer:\n{answer}"
        )}
    ]
    
    # ⬇️ --- THIS IS THE CRITICAL FIX --- ⬇️
    # Apply the chat template to convert the list of messages into a single string
    # We pass this string back to the old SFTTrainer.
    try:
        formatted_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return formatted_string
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return ""
# 6. Configure PEFT (LoRA) (Unchanged)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 7. Define Evaluation Metrics (Unchanged)
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    pred_logits = predictions[0] if isinstance(predictions, tuple) else predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    
    label_ids = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    decoded_preds = [pred for pred, label in zip(decoded_preds, decoded_labels) if label]
    decoded_labels = [label for label in decoded_labels if label]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"],
    }

# 8. Configure Training Arguments (Unchanged)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    # per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    logging_steps=25,
    # eval_strategy="steps",
    # eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    report_to="none",
    bf16=True,
    save_total_limit=2,
    # load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 9. ⬇️ CORRECTED SFTTrainer Initialization
# This uses the older API arguments (processing_class, no max_seq_length)
# that your system's tracebacks require.
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # Fix 1: Use processing_class
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=peft_config,
    formatting_func=format_chat_prompt,
)

# 10. Start Training
print("Starting training...")
trainer.train()

# 11. Save the final adapter model
print("Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete. Model and tokenizer saved to:", output_dir)