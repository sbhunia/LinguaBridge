import os
# -------------------------
# ENV SETUP
# -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.environ["WANDB_PROJECT"] = "translate-gemma-myaamia-lora"

import logging
import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer

# -------------------------
# CONFIG
# -------------------------
model_name = "google/translategemma-4b-it"
save_dir = "./gemma_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./gemma_eval_results.tsv"

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

logging.info("Starting Gemma fine-tuning script with LoRA.")

# -------------------------
# LOAD DATA
# -------------------------
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        df.columns = [c.lower() for c in df.columns]
        
        if "mia" not in df.columns or "en" not in df.columns:
            logging.warning(f"Skipping {path}: Missing columns.")
            return None
            
        df = df.dropna(subset=["mia", "en"])
        df["mia"] = df["mia"].astype(str)
        df["en"] = df["en"].astype(str)
        logging.info(f"Loaded {len(df)} rows from {path}")
        return Dataset.from_pandas(df)
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

# Update paths relative to your script
datasets = [
    load_tsv("../../../data/basic_forms.tsv"),
    load_tsv("../../../data/command_forms.tsv"),
    load_tsv("../../../data/sentences.tsv"),
    load_tsv("../../../data/story_book.tsv"),
]

datasets = [ds for ds in datasets if ds is not None]

if not datasets:
    logging.error("No data was successfully loaded. Exiting.")
    exit()

full_dataset = concatenate_datasets(datasets)
full_dataset = full_dataset.shuffle(seed=42)
full_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
logging.info(f"Train size: {len(full_dataset['train'])}, Test size: {len(full_dataset['test'])}")

# -------------------------
# FORMATTING FOR CAUSAL LM
# -------------------------
# Gemma needs text formatted as a single instruction block
# -------------------------
# FORMATTING FOR CAUSAL LM
# -------------------------
def formatting_prompts_func(example):
    # Base prompt template
    system_prompt = (
        "<start_of_turn>user\n"
        "You are a professional Myaamia (mia) to English (en) translator. Produce only the English translation, without any additional explanations or commentary. Please translate the following Myaamia text into English:\n\n"
    )

    # If SFTTrainer passes a batch (lists)
    if isinstance(example['mia'], list):
        output_texts = []
        for i in range(len(example['mia'])):
            prompt = (
                f"{system_prompt}{example['mia'][i]}<end_of_turn>\n"
                f"<start_of_turn>model\n{example['en'][i]}<end_of_turn>"
            )
            output_texts.append(prompt)
        return output_texts
        
    # If SFTTrainer passes a single instance (strings)
    else:
        return (
            f"{system_prompt}{example['mia']}<end_of_turn>\n"
            f"<start_of_turn>model\n{example['en']}<end_of_turn>"
        )
# -------------------------
# TOKENIZER
# -------------------------
logging.info(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right" # Important for training causal LMs

# Gemma does not have a pad token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# MODEL & LORA SETUP
# -------------------------
logging.info(f"Loading base model: {model_name}")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto" # Automatically places model on available GPU(s)
)

model.config.use_cache = False 
model.gradient_checkpointing_enable()

# LoRA Configuration for Gemma
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------
# TRAINER (SFTTrainer)
# -------------------------
training_args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=40,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    report_to="wandb",
    fp16=not use_bf16,
    bf16=use_bf16,
    optim="adamw_torch" 
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=full_dataset["train"],
    eval_dataset=full_dataset["test"],
    args=training_args,
    formatting_func=formatting_prompts_func,
)

# -------------------------
# GENERATION EVALUATION (BLEU/chrF++)
# -------------------------
sacrebleu = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

def evaluate_translation(model, tokenizer, test_dataset):
    logging.info("Starting evaluation for SacreBLEU and chrF++...")
    
    tokenizer.padding_side = "left"
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    preds = []
    refs = []
    
    with open(eval_file, "w") as f:
        f.write("myaamia\treference\tprediction\n")

    for item in test_dataset:
        # Construct the evaluation prompt without the target answer
        prompt = (
            "<start_of_turn>user\n"
            "You are a professional Myaamia (mia) to English (en) translator. Produce only the Myaamia translation, without any additional explanations or commentary. Please translate the following Myaamia text into English:\n\n"
            f"{item['mia']}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        
        # We only want the model to generate the translation, not repeat the prompt
        outputs = pipe(
            prompt, 
            max_new_tokens=128, 
            do_sample=False, 
            truncation=True,
            return_full_text=False # This prevents the pipeline from outputting the prompt
        )
        
        generated_text = outputs[0]["generated_text"]
        
        # Clean up the output by splitting at the end_of_turn token if it exists
        translation = generated_text.split("<end_of_turn>")[0].strip()
            
        preds.append(translation)
        refs.append([item["en"]])
        
        with open(eval_file, "a") as f:
            f.write(f"{item['mia']}\t{item['en']}\t{translation}\n")

    bleu_result = sacrebleu.compute(predictions=preds, references=refs)
    chrf_result = chrf_metric.compute(predictions=preds, references=refs, word_order=2)
    
    logging.info(f"Final SacreBLEU: {bleu_result['score']:.4f}")
    logging.info(f"Final chrF++: {chrf_result['score']:.4f}")
    
    with open(eval_file, "a") as f:
        f.write(f"\nFINAL METRICS:\nSacreBLEU:\t{bleu_result['score']}\nchrF++:\t{chrf_result['score']}\n")
# -------------------------
# EXECUTION
# -------------------------
if __name__ == "__main__":
    logging.info("Starting training...")
    trainer.train()

    logging.info(f"Saving LoRA adapter to {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Run custom evaluation on the test set after training
    evaluate_translation(trainer.model, tokenizer, full_dataset["test"])

    logging.info("Process complete.")

#     {'eval_loss': 0.35729506611824036, 'eval_runtime': 40.8688, 'eval_samples_per_second': 46.735, 'eval_steps_per_second': 5.848, 'eval_entropy': 0.11016088537092489, 'eval_num_tokens': 67506840.0, 'eval_mean_token_accuracy': 0.9556261064118421, 'epoch': 40.0}                                       
# {'train_runtime': 57855.5077, 'train_samples_per_second': 11.882, 'train_steps_per_second': 0.743, 'train_loss': 0.12291671465025392, 'epoch': 40.0}