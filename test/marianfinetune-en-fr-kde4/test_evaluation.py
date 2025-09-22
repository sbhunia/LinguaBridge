# https://github.com/EliasK93/transformer-models-for-domain-specific-machine-translation/tree/master

from statistics import mean
import bert_score
import pandas as pd
import nltk
import evaluate
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download("wordnet")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load metrics
bleu_metric = evaluate.load("sacrebleu")

def calc_meteor_avg(predictions_: list[str], references_: list[list[str]]) -> float:
    scores = [nltk.translate.meteor_score.single_meteor_score(ref[0].split(), hyp.split())
              for hyp, ref in zip(predictions_, references_)]
    return mean(scores)

def calc_bleu(predictions_: list[str], references_: list[list[str]]) -> float:
    return bleu_metric.compute(predictions=predictions_, references=references_)["score"] / 100

def calc_bert_score(predictions_: list[str], references_: list[list[str]], lang: str) -> float:
    return float(bert_score.score(predictions_, references_, lang=lang, rescale_with_baseline=True)[2].mean())

def generate_translations(model, tokenizer, sources, batch_size=32, max_length=256):
    model.eval()
    translations = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    return translations


if __name__ == '__main__':
    # Load the raw dataset
    raw_datasets = load_dataset("Helsinki-NLP/opus_books", "en-fr")
    split_datasets = raw_datasets["train"].train_test_split(train_size=0.7, seed=20)
    split_datasets["validation"] = split_datasets.pop("test")
    validation_data = split_datasets["validation"]

    # Extract source and reference texts
    sources = [item["translation"]["en"] for item in validation_data]
    references = [[item["translation"]["fr"]] for item in validation_data]  # wrapped in extra list as expected

    # Load local model and tokenizer
    local_checkpoint_path = "/home/yogesh/LinguaBridge/test/marianfinetune-en-fr-kde4/marian-finetuned-opusbooks-en-to-fr/checkpoint-8340"
    tokenizer = AutoTokenizer.from_pretrained(local_checkpoint_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_checkpoint_path, local_files_only=True).to(device)

    # Generate translations
    print("Generating translations...")
    predictions = generate_translations(model, tokenizer, sources)

    # Compute metrics
    print("Evaluating...")
    scores = {
        "sacrebleu": calc_bleu(predictions, references),
        "meteor": calc_meteor_avg(predictions, references),
        "bert_score": calc_bert_score(predictions, references, lang="fr")
    }

    # Display results
    df = pd.DataFrame([scores], index=["marian-finetuned-opusbooks-en-to-fr"]).round(3)
    print(df.to_string())
    # df.to_excel("evaluation_results_finetuned_model.xlsx")


#     Output:
#                                          sacrebleu  meteor  bert_score
# marian-finetuned-opusbooks-en-to-fr      0.278   0.457       0.586
