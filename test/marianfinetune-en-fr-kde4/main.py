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

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

# teting tokenizer
# en_sentence = split_datasets["train"][1]["translation"]["en"]
# fr_sentence = split_datasets["train"][1]["translation"]["fr"]

# inputs = tokenizer(en_sentence, text_target=fr_sentence)
# print(inputs)
# Copied
# {'input_ids': [405, 2, 606, 41418, 101, 192, 4751, 14552, 101, 4646, 2, 189, 202, 5048, 254, 18, 4, 33353, 3452, 235, 123, 13514, 18, 877, 3022, 101, 1185, 444, 4, 4862, 85, 173, 24447, 2, 61, 6595, 69, 103, 4752, 85, 36555, 6127, 22, 6774, 1308, 2, 288, 173, 8620, 2, 18, 4, 34689, 2, 218, 47711, 244, 30124, 2346, 2, 22792, 7, 98, 14656, 37, 58, 561, 1010, 15, 37151, 46, 3811, 2, 10, 15, 577, 139, 14698, 1686, 18, 1631, 2, 26, 9528, 1356, 139, 14698, 1098, 14971, 18, 7517, 4015, 16462, 13079, 7, 9612, 50, 7, 85, 47, 79, 264, 256, 650, 1514, 18777, 51, 3897, 2593, 4, 26263, 9, 2, 27771, 10, 6309, 12, 2655, 4, 6821, 5, 18142, 133, 2, 1353, 42, 472, 12, 103, 1081, 1773, 2, 10, 606, 1165, 17606, 212, 151, 7, 103, 855, 2, 85, 32, 890, 12, 1063, 538, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': [419, 294, 3584, 371, 4751, 359, 5415, 4646, 2, 44, 16, 9415, 395, 39, 31842, 3334, 174, 116, 40, 4152, 1504, 17, 316, 1713, 14, 1, 40193, 68, 1, 107, 7773, 2, 8281, 137, 1, 82, 125, 5616, 44, 15, 29660, 395, 19, 2557, 1992, 6774, 1308, 1766, 1, 107, 44727, 31, 19, 35281, 129, 9762, 81, 4277, 31490, 30124, 9, 2, 32571, 5, 182, 16, 13995, 78, 153, 47, 51, 558, 41327, 5, 3811, 11, 40, 139, 14698, 1686, 5, 2208, 2, 4838, 40, 139, 14698, 1098, 23, 18394, 3697, 2044, 5, 2208, 78, 5, 915, 986, 1, 211, 2175, 7180, 1604, 1460, 23, 9840, 9, 34961, 3897, 2593, 16, 527, 6266, 1096, 2, 27090, 11, 6371, 5, 505, 1689, 6034, 617, 19, 14233, 5, 1648, 1744, 78, 363, 447, 39, 4516, 472, 5, 125, 19387, 2, 11, 294, 27998, 139, 658, 5, 125, 1801, 1167, 2, 44, 43, 924, 3, 416, 0]} 
# input IDs are associated with the English sentence, while the IDs associated with the French one are stored in the labels field.


max_length = 256

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
    f"marian-finetuned-opusbooks-en-to-fr",
    do_eval=False,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    push_to_hub=True
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
print(trainer.evaluate(max_length=max_length))

trainer.train()
print("After:")
print(trainer.evaluate(max_length=max_length))
end = time.time()     # record end time

print(f"Time taken: {(end - start)/60:.2f} minutes")

print(
    trainer.push_to_hub(tags="translation", commit_message="Training complete")
)

print("Ended")

# Output:
# (myaamiatranslate) yogesh@bhunias-gpu:~/test/marianfinetune-en-fr-kde4$ python main.py
# /home/yogesh/miniconda3/envs/myaamiatranslate/lib/python3.13/site-packages/transformers/models/marian/tokenization_marian.py:175: UserWarning: Recommended: pip install sacremoses.
#   warnings.warn("Recommended: pip install sacremoses.")
# /home/yogesh/test/marianfinetune-en-fr-kde4/main.py:93: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
#   trainer = Seq2SeqTrainer(
# Before:
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 596/596 [09:28<00:00,  1.05it/s]
# {'eval_loss': 2.3046212196350098, 'eval_model_preparation_time': 0.0027, 'eval_bleu': 21.56748969583471, 'eval_runtime': 569.3743, 'eval_samples_per_second': 66.961, 'eval_steps_per_second': 1.047}
# The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None}.
# {'loss': 2.0025, 'grad_norm': 2.789850950241089, 'learning_rate': 1.8803357314148683e-05, 'epoch': 0.18}                                                     
# {'loss': 1.8387, 'grad_norm': 3.082397699356079, 'learning_rate': 1.760431654676259e-05, 'epoch': 0.36}                                                      
# {'loss': 1.8056, 'grad_norm': 3.162931442260742, 'learning_rate': 1.64052757793765e-05, 'epoch': 0.54}                                                       
# {'loss': 1.7473, 'grad_norm': 2.9437859058380127, 'learning_rate': 1.520623501199041e-05, 'epoch': 0.72}                                                     
# {'loss': 1.7236, 'grad_norm': 3.2606217861175537, 'learning_rate': 1.4007194244604318e-05, 'epoch': 0.9}                                                     
#  33%|██████████████████████████████████████▋                                                                             | 2780/8340 [01:49<03:37, 25.60it/s]/home/yogesh/miniconda3/envs/myaamiatranslate/lib/python3.13/site-packages/transformers/modeling_utils.py:4034: UserWarning: Moving the following attributes in the config to the generation config: {'max_length': 512, 'num_beams': 4, 'bad_words_ids': [[59513]]}. You are seeing this warning because you've set generation parameters in the model config, as opposed to in the generation config.
#   warnings.warn(
# {'loss': 1.6888, 'grad_norm': 3.1859474182128906, 'learning_rate': 1.2808153477218225e-05, 'epoch': 1.08}                                                    
# {'loss': 1.6321, 'grad_norm': 2.9554243087768555, 'learning_rate': 1.1609112709832136e-05, 'epoch': 1.26}                                                    
# {'loss': 1.6084, 'grad_norm': 2.7089009284973145, 'learning_rate': 1.0410071942446045e-05, 'epoch': 1.44}                                                    
# {'loss': 1.6036, 'grad_norm': 2.7929227352142334, 'learning_rate': 9.211031175059952e-06, 'epoch': 1.62}                                                     
# {'loss': 1.5844, 'grad_norm': 2.7569596767425537, 'learning_rate': 8.011990407673862e-06, 'epoch': 1.8}                                                      
# {'loss': 1.5841, 'grad_norm': 2.8149566650390625, 'learning_rate': 6.81294964028777e-06, 'epoch': 1.98}                                                      
# {'loss': 1.5443, 'grad_norm': 3.128256320953369, 'learning_rate': 5.613908872901679e-06, 'epoch': 2.16}                                                      
# {'loss': 1.5346, 'grad_norm': 2.7403597831726074, 'learning_rate': 4.414868105515588e-06, 'epoch': 2.34}                                                     
# {'loss': 1.5286, 'grad_norm': 2.4553604125976562, 'learning_rate': 3.215827338129497e-06, 'epoch': 2.52}                                                     
# {'loss': 1.5246, 'grad_norm': 2.7141470909118652, 'learning_rate': 2.0167865707434053e-06, 'epoch': 2.7}                                                     
# {'loss': 1.5389, 'grad_norm': 2.808008909225464, 'learning_rate': 8.177458033573142e-07, 'epoch': 2.88}                                                      
# {'train_runtime': 331.124, 'train_samples_per_second': 805.973, 'train_steps_per_second': 25.187, 'train_loss': 1.6505343210782937, 'epoch': 3.0}            
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8340/8340 [05:31<00:00, 25.19it/s]
# After:
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 596/596 [09:38<00:00,  1.03it/s]
# {'eval_loss': 1.5279783010482788, 'eval_model_preparation_time': 0.0027, 'eval_bleu': 28.540603034063956, 'eval_runtime': 578.6826, 'eval_samples_per_second': 65.884, 'eval_steps_per_second': 1.03, 'epoch': 3.0}
# Time taken: 24.86 minutes
# Processing Files (4 / 4)                : 100%|█████████████████████████████████████████████████████████████████████████████████|  300MB /  300MB,  204MB/s  
# New Data Upload                         : |                                                                                     |  0.00B /  0.00B,  0.00B/s  
#   ...pusbooks-en-to-fr/training_args.bin: 100%|█████████████████████████████████████████████████████████████████████████████████| 5.97kB / 5.97kB            
#   ...tuned-opusbooks-en-to-fr/source.spm: 100%|█████████████████████████████████████████████████████████████████████████████████|  778kB /  778kB            
#   ...tuned-opusbooks-en-to-fr/target.spm: 100%|█████████████████████████████████████████████████████████████████████████████████|  802kB /  802kB            
#   ...pusbooks-en-to-fr/model.safetensors: 100%|█████████████████████████████████████████████████████████████████████████████████|  299MB /  299MB            
# https://huggingface.co/chaudhy/marian-finetuned-opusbooks-en-to-fr/tree/main/
# Ended