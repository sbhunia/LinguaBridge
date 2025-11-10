import os
import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import nltk
import bert_score
from statistics import mean
# https://huggingface.co/transformers/v3.5.1/model_doc/marian.html#mariantokenizer
# -------------------------
# CONFIG
# -------------------------
model_name = "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul"   # ✅ replace with right pair if exists
save_dir = "./marian_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./eval_results.tsv"
nltk.download("wordnet", quiet=True)

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------
# LOAD DATA
# -------------------------
def load_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["mia", "en"])  # Remove rows with missing translations
    df["mia"] = df["mia"].astype(str)
    df["en"] = df["en"].astype(str)
    return Dataset.from_pandas(df)


datasets = [
    load_tsv("../../../../data/basic_forms.tsv"),
    load_tsv("../../../../data/command_forms.tsv"),
    load_tsv("../../../../data/sentences.tsv"),
    load_tsv("../../../../data/story_book.tsv"),
]

full_dataset = concatenate_datasets(datasets)
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)  # 70% train, 30% val

# -------------------------
# MODEL
# -------------------------
model = MarianMTModel.from_pretrained(model_name)

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Old code had a source and target direction issue with the model selected. Expected: english as souce and myaamia/french(pretrained model) as target
# def preprocess(batch):
#     model_inputs = tokenizer(batch["mya"], max_length=128, truncation=True)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(batch["en"], max_length=128, truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# tokenized_dataset = full_dataset.map(preprocess, batched=True) 

tgt_token = ">>mia<<" # Marian style target language token based on ISO 639-3 language codes

# tokenizer.add_tokens([tgt_token])
# ➤ Adds the new token (e.g., >>MIA<<) to the tokenizer’s vocabulary.

# Before this, the tokenizer doesn’t recognize >>MIA<< as a single token — it would split it into smaller pieces or unknown tokens.

# After this, it becomes a valid token that can appear in tokenized input or output.

# model.resize_token_embeddings(len(tokenizer))
# ➤ Expands the model’s embedding layer to match the new vocabulary size.

# MarianMT (and most Transformer models) has an embedding matrix where each token has a vector.

# When you add a new token, this call tells the model to add a new row to that matrix so it can learn an embedding for your new token during fine-tuning.

if tgt_token not in tokenizer.get_vocab():
    tokenizer.add_tokens([tgt_token])
    model.resize_token_embeddings(len(tokenizer))

def preprocess(batch):
    # prepend the language token to source sentences
    src_texts = [f"{tgt_token} {text}" for text in batch["en"]]
    tgt_texts = batch["mia"]

     # Tokenize source (English with language tag)
    model_inputs = tokenizer(
        src_texts,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["mia"], max_length=128, truncation=True, padding="max_length")


    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# def preprocess(batch):
#     # Source: English
#     model_inputs = tokenizer(batch["en"], max_length=128, truncation=True, padding="max_length")
    
#     # Target: Myaamia
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer([prefix + t for t in batch["mya"]], 
#                            max_length=128, truncation=True, padding="max_length")
    
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

tokenized_dataset = full_dataset.map(preprocess, batched=True)


# -------------------------
# METRIC (sacreBLEU)
# -------------------------
metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_score = bleu_result["score"]

    # Compute METEOR using NLTK
    meteor_scores = [
        nltk.translate.meteor_score.single_meteor_score(ref[0].split(), hyp.split())
        for hyp, ref in zip(decoded_preds, decoded_labels)
    ]
    meteor_score = mean(meteor_scores)

    # Compute BERTScore
    P, R, F1 = bert_score.score(decoded_preds, [ref[0] for ref in decoded_labels],
                                lang="fr", rescale_with_baseline=True)
    bert_f1 = float(F1.mean())

    # Combine results
    result = {
        "bleu": round(bleu_score, 4),
        "meteor": round(meteor_score, 4),
        "bert_score": round(bert_f1, 4)
    }

    # Log to file
    with open(eval_file, "a") as f:
        f.write(str(result) + "\n")

    return result

# -------------------------
# TRAINING
# -------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    eval_strategy="epoch",               
    save_strategy="epoch",
    learning_rate=2e-5,
    warmup_ratio=0.06,           
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=5,
    fp16=True,
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------
# TRAIN
# -------------------------
trainer.train()


# @article{tiedemann2023democratizing,
#   title={Democratizing neural machine translation with {OPUS-MT}},
#   author={Tiedemann, J{\"o}rg and Aulamo, Mikko and Bakshandaeva, Daria and Boggia, Michele and Gr{\"o}nroos, Stig-Arne and Nieminen, Tommi and Raganato\
# , Alessandro and Scherrer, Yves and Vazquez, Raul and Virpioja, Sami},
#   journal={Language Resources and Evaluation},
#   number={58},
#   pages={713--755},
#   year={2023},
#   publisher={Springer Nature},
#   issn={1574-0218},
#   doi={10.1007/s10579-023-09704-w}
# }

# @InProceedings{TiedemannThottingal:EAMT2020,
#   author = {J{\"o}rg Tiedemann and Santhosh Thottingal},
#   title = {{OPUS-MT} — {B}uilding open translation services for the {W}orld},
#   booktitle = {Proceedings of the 22nd Annual Conferenec of the European Association for Machine Translation (EAMT)},
#   year = {2020},
#   address = {Lisbon, Portugal}
#  }