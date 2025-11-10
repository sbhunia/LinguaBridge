import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer

# 1. Configuration
model_id = "meta-llama/Llama-3.2-1B-Instruct"
dataset_name = "TheFinAI/Fino1_Reasoning_Path_FinQA"
output_dir = "llama-3.2-1b-fino1-finetuned"

# 2. Load Tokenizer
# Llama 3.2 models do not have a default padding token.
# We set it to the end-of-sentence token for training.
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 3. Load Model
# We load in bfloat16 for a good balance of precision and memory efficiency.
# No 4-bit quantization is needed for a 1B model on most modern GPUs.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficient training
    device_map="auto"            # Automatically use available GPU
)

# 4. Load Dataset
# This dataset only has a 'train' split, so we'll load it and then
# create our own 90/10 train/validation split.
dataset = load_dataset(dataset_name, split="train")
dataset_splits = dataset.train_test_split(test_size=0.1)

train_data = dataset_splits["train"]
test_data = dataset_splits["test"]

# 5. Helper Function to Format the Dataset
# This is the most important part. We must format the dataset's columns
# into the Llama 3.2 chat template.

def serialize_table(table_dict):
    """Converts the table dictionary into a readable markdown string."""
    try:
        header = " | ".join(table_dict['header'])
        separator = " | ".join(["---"] * len(table_dict['header']))
        rows = "\n".join([" | ".join(row) for row in table_dict['rows']])
        return f"{header}\n{separator}\n{rows}"
    except Exception:
        # Fallback for any unexpected table format
        return str(table_dict)

def format_chat_prompt(example):
    """
    Formats the dataset example into a list of messages
    for the Llama 3.2 chat template.
    """
    system_prompt = (
        "You are a helpful and meticulous financial analyst. "
        "Answer the question based *only* on the provided text and table. "
        "Show your reasoning step-by-step to arrive at the final answer."
    )
    
    # Serialize the table and create the user's query
    table_str = serialize_table(example['table'])
    user_prompt = (
        f"Context Text:\n{example['text']}\n\n"
        f"Context Table:\n{table_str}\n\n"
        f"Question:\n{example['question']}"
    )
    
    # Format the model's desired response
    assistant_response = (
        f"Reasoning:\n{example['reasoning_path']}\n\n"
        f"Answer:\n{example['answer']}"
    )
    
    # Return the message structure
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response},
    ]

# 6. Configure PEFT (LoRA)
# We apply LoRA to speed up training and reduce memory, even on a 1B model.
peft_config = LoraConfig(
    r=16,                # LoRA rank (higher rank = more parameters)
    lora_alpha=32,       # Alpha for scaling
    target_modules=[     # Target modules to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 7. Configure Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,   # Lower if you run out of VRAM
    gradient_accumulation_steps=4,   # Effective batch size = 2 * 4 = 8
    learning_rate=2e-4,
    num_train_epochs=1,              # 1-3 epochs is usually fine for PEFT
    lr_scheduler_type="cosine",
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    report_to="none",                # Set to "wandb" or "tensorboard" if you use them
    fp16=False,                      # Not needed, as we use bf16
    bf16=True,                       # Use bfloat16 precision
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 8. Initialize the SFTTrainer
# This trainer automatically handles:
# - Applying the chat template (via formatting_func)
# - Tokenizing the prompts
# - Packing sequences for efficiency
# - Masking labels (so loss is only calculated on the assistant's response)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=peft_config,
    formatting_func=format_chat_prompt, # This is the key!
    max_seq_length=2048,               # This dataset has long context
    dataset_kwargs={"num_proc": 4}     # Use multiple cores for formatting
)

# 9. Start Training
print("Starting training...")
trainer.train()

# 10. Save the final adapter model
print("Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete. Model and tokenizer saved to:", output_dir)