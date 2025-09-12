from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files="./data/basic_forms.tsv",
    delimiter="\t"  # Important for TSV
)

print(dataset['train' ][0])

# Command:
# python dataset.py

# Output:
# Generating train split: 12891 examples [00:00, 740140.90 examples/s]
# {'mya': 'aahkohkimaki', 'en': 'I command him / her'}
