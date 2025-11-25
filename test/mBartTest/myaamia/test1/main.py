import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
# ➤ CHANGED: Import MBart model and tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import nltk
import bert_score
from statistics import mean
import wandb
from transformers import GenerationConfig

# -------------------------
# CONFIG
# -------------------------
# ➤ CHANGED: Model name is now mBART-50
model_name = "facebook/mbart-large-50"
save_dir = "./mbart_finetuned_myaamia"
log_file = "./mbart_training.log"
eval_file = "./mbart_eval_results.tsv"
nltk.download("wordnet", quiet=True)

# mBART-50 requires language codes.
# English is 'en_XX'. Myaamia is a low-resource language and does not have a standard code in the 50 set.
# We will use the 'en_XX' language code for *both* English and Myaamia, or a custom token,
# but the most stable approach for a language outside the 50 is often to use a **pivot language code**
# or add a new token (like you did with Marian), which is complex for mBART.
# For simplicity, and since it is often the pivot, we will use 'en_XX' as the target code
# and rely on the model learning the new tokens, **BUT WE MUST ADD A NEW TOKEN**.

# For Myaamia (MIA), we MUST add a new language token to the vocabulary and use it.
MIA_LANG_CODE = "mia_MIA"
EN_LANG_CODE = "en_XX"

os.environ["WANDB_PROJECT"] = "myaamia-translation-mt5"

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting mBART fine-tuning script.")

# -------------------------
# LOAD DATA (Identical to T5 script)
# -------------------------
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
        df["mia"] = df["mia"].astype(str)
        df["en"] = df["en"].astype(str)
        logging.info(f"Loaded {len(df)} rows from {path}")
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        logging.error(f"Data file not found at: {path}")
        return None
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

datasets = [
    load_tsv("../../../../data/basic_forms.tsv"),
    load_tsv("../../../../data/command_forms.tsv"),
    load_tsv("../../../../data/sentences.tsv"),
    load_tsv("../../../../data/story_book.tsv"),
]

datasets = [ds for ds in datasets if ds is not None]

if not datasets:
    logging.error("No data was successfully loaded. Exiting.")
    exit()

full_dataset = concatenate_datasets(datasets)
logging.info(f"Total combined dataset size: {len(full_dataset)}")
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)
logging.info(f"Train size: {len(full_dataset['train'])}, Test size: {len(full_dataset['test'])}")

# -------------------------
# MODEL & TOKENIZER
# -------------------------
# ➤ CHANGED: Using MBartForConditionalGeneration
logging.info(f"Loading model: {model_name}")
model = MBartForConditionalGeneration.from_pretrained(model_name)

# ➤ CHANGED: Using MBart50TokenizerFast
logging.info(f"Loading tokenizer: {model_name}")
# Must set a dummy src_lang for initialization
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang=EN_LANG_CODE)

# ➤ CRITICAL: Add the new language token for Myaamia
if MIA_LANG_CODE not in tokenizer.lang_code_to_id:
    # Add the special token to the tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": [MIA_LANG_CODE]})
    tokenizer.lang_code_to_id[MIA_LANG_CODE] = tokenizer.convert_tokens_to_ids(MIA_LANG_CODE)
    # Resize the model embeddings to account for the new token
    model.resize_token_embeddings(len(tokenizer))
    logging.info(f"Added new language token '{MIA_LANG_CODE}' and resized embeddings.")


# -------------------------
# PREPROCESSING (CRITICAL CHANGES FOR mBART-50)
# -------------------------
# mBART-50 format: [SRC_LANG_CODE] X [EOS] for source, [TGT_LANG_CODE] Y [EOS] for target labels
def preprocess(batch):
    # Set the source language for the tokenizer
    tokenizer.src_lang = EN_LANG_CODE
    
    # Source: English. The tokenizer will automatically prepend the EN_LANG_CODE token.
    src_texts = batch["en"]
    
    # Target: Myaamia
    tgt_texts = batch["mia"]

    model_inputs = tokenizer(src_texts, max_length=256, truncation=True, padding="max_length")

    # Set the target language for the labels
    tokenizer.tgt_lang = MIA_LANG_CODE
    
    # Tokenize targets. The tokenizer will automatically prepend the MIA_LANG_CODE token.
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, max_length=128, truncation=True, padding="max_length")

    # Shift the labels one position to the right to create the decoder_input_ids
    # For training, we need 'labels' to have the target language code as the first token.
    model_inputs["labels"] = labels["input_ids"]
    
    # The decoder_start_token_id is handled automatically by the Seq2SeqTrainer for mBART
    # but we must pass the forced_bos_token_id to the generate call during evaluation.
    return model_inputs

logging.info("Tokenizing dataset...")
tokenized_dataset = full_dataset.map(preprocess, batched=True)
logging.info("Tokenization complete.")

# -------------------------
# METRIC (Identical to T5 script, but modified generate settings)
# -------------------------
metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
chrf_metric = evaluate.load("chrf")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    # Get the ID of the new Myaamia language token
    forced_bos_token_id = tokenizer.lang_code_to_id[MIA_LANG_CODE]

    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # Replace -100 (ignore index) with pad token for decoding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and references
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # sacreBLEU
    bleu_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_score = bleu_result["score"]

    # METEOR (using NLTK implementation)
    meteor_scores = [
        nltk.translate.meteor_score.single_meteor_score(ref[0].split(), hyp.split())
        for hyp, ref in zip(decoded_preds, decoded_labels)
    ]
    meteor_score = mean(meteor_scores)

    # BERTScore
    try:
        # For a low-resource language, rely on the multilingual model (no lang code)
        P, R, F1 = bert_score.score(decoded_preds, [ref[0] for ref in decoded_labels],
                                    rescale_with_baseline=True)
        bert_f1 = float(F1.mean())
    except Exception as e:
        logging.warning(f"BERTScore calculation failed: {e}")
        bert_f1 = 0.0

    # chrF++
    try:
        chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        chrf_score = chrf_result["score"]
    except Exception as e:
        logging.warning(f"chrF++ calculation failed: {e}")
        chrf_score = 0.0

    # Combine results
    result = {
        "bleu": round(bleu_score, 4),
        "meteor": round(meteor_score, 4),
        "bert_score": round(bert_f1, 4),
        "chrf++": round(chrf_score, 4),
    }

    try:
        with open(eval_file, "a") as f:
            f.write(f"Epoch result: {result}\n")
    except Exception as e:
        logging.warning(f"Could not write to eval_file: {e}")

    return result

# -------------------------
# TRAINING
# -------------------------
# Get the ID of the new Myaamia language token
forced_bos_token_id = tokenizer.lang_code_to_id[MIA_LANG_CODE]

# Create the GenerationConfig object
# MBart uses the decoder_start_token_id for the BOS token of the target language
generation_config = GenerationConfig.from_pretrained(model_name)
generation_config.decoder_start_token_id = forced_bos_token_id
generation_config.max_length = 128 # Set a max length for generation
# https://huggingface.co/docs/transformers/v4.57.3/en/main_classes/text_generation#transformers.GenerationConfig
training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    warmup_ratio=0.06,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    logging_dir="./logs_mbart",
    logging_strategy="steps",
    logging_steps=5,
    fp16=True, # mBART-50 is a large model, fp16 is often preferred
    report_to="wandb",
    run_name="mbart-myaamia-finetune-test1",
    generation_config=generation_config
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
logging.info("Starting training...")
trainer.train()
logging.info("Training complete.")

# Save the final model
trainer.save_model(os.path.join(save_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(save_dir, "final_model"))
logging.info(f"Final model saved to {os.path.join(save_dir, 'final_model')}")

# Finish the wandb run
wandb.finish()