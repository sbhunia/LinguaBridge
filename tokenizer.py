from transformers import MBart50Tokenizer
from datasets import load_dataset

tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
print(tokenizer.lang_code_to_id.keys())

# Load your data (for both directions)
dataset = load_dataset(
    "csv",
    data_files="./data/basic_forms.tsv",
    delimiter="\t"  # Important for TSV
)

# Tokenize your data for both directions
def tokenize_function(examples):
    # Add appropriate language tokens for each direction
    tokenizer.src_lang = "mya_US" 
    tokenizer.tgt_lang = "en_XX"

    source = examples["source_sentence"]
    target = examples["target_sentence"]
    
    model_inputs = tokenizer(source, text_target=target, padding="max_length", truncation=True)
    return model_inputs

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print(type(tokenized_datasets))
