from datasets import load_dataset

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")
print(dataset["train"][0])  # Inspect the first sample

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-7b")
def preprocess_data(example):
    return tokenizer(
        example["text"],  # Assumes your dataset has a 'text' column
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(preprocess_data, batched=True)
print(tokenized_dataset["train"][0])  # Check tokenized output

# def format_data(example):
#     # Concatenate text for language modeling
#     example["input_ids"] = tokenizer.encode(example["text"], truncation=True, max_length=512)
#     example["labels"] = example["input_ids"].copy()
#     return example

# formatted_dataset = tokenized_dataset.map(format_data, remove_columns=["text"])
# print(formatted_dataset["train"][0])

train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_data = train_test_split["train"]
test_data = train_test_split["test"]
print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# tokenized_dataset = dataset.map(preprocess_data, batched=True, batch_size=1000)

from transformers import AutoModelForCausalLM

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-7b",
    device_map="auto",  # Automatically map layers across GPUs if available
    load_in_4bit=True   # Enables memory-efficient training
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./llama3_finetuned",  # Where to save the model
    evaluation_strategy="steps",     # Evaluate during training
    save_strategy="steps",           # Save checkpoints
    learning_rate=2e-5,              # A good starting point for fine-tuning
    per_device_train_batch_size=4,   # Adjust based on GPU memory
    gradient_accumulation_steps=8,   # Simulates a larger batch size
    num_train_epochs=3,              # Experiment with more epochs for small datasets
    logging_steps=100,               # Log training progress
    save_steps=500,                  # Save model every 500 steps
    fp16=True,                       # Mixed precision for faster training
    save_total_limit=3,
    push_to_hub=False                # Skip pushing to Hugging Face Hub for now
)

from transformers import Trainer

trainer = Trainer(
    model=model,                          # Pretrained LLaMA 3 model
    args=training_args,                   # Training configurations
    train_dataset=tokenized_dataset["train"],  # Your tokenized training data
    eval_dataset=tokenized_dataset["test"]    # Your tokenized validation data
)

# Start training
trainer.train()

model.save_pretrained("./llama3_finetuned")
tokenizer.save_pretrained("./llama3_finetuned")
