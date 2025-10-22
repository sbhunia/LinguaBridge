# https://huggingface.co/learn/llm-course/chapter7/4?fw=pt

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
import numpy as np
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import time
import torch
import evaluate

start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/el-fr?views%5B%5D=el_fr
# 127k rows en-fr
raw_datasets = load_dataset("Helsinki-NLP/opus_books", "en-fr")
split_datasets = raw_datasets["train"].train_test_split(train_size=0.7, seed=20)
split_datasets["validation"] = split_datasets.pop("test")

model_checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

max_length = 128

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model) #used for padding not only input ids but labels too

#version
# sacrebleu-2.5.1
metric = evaluate.load("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    if preds.ndim == 3:  # sometimes you get logits instead of ids
        preds = np.argmax(preds, axis=-1)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# documentation
# https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer#transformers.TrainingArguments
args = Seq2SeqTrainingArguments(
    f"marian-finetuned-opusbooks-en-to-fr-graph",
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=5,
    report_to="tensorboard"
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("Before:")
print(trainer.evaluate())

trainer.train()
print("After:")
print(trainer.evaluate())
end = time.time()     # record end time

print(f"Time taken: {(end - start)/60:.2f} minutes")

print(
    trainer.push_to_hub(tags="translation", commit_message="Training complete")
)

print("Ended")

