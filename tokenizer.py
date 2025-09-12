from transformers import MBart50TokenizerFast
from datasets import load_dataset

# Load pre-trained MarianMT model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Load your data (for both directions)
dataset = load_dataset(
    "csv",
    data_files="./dataset/basic_forms.tsv",
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
print(tokenized_datasets)
