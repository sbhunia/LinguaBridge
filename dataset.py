from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files="./dataset/basic_forms.tsv",
    delimiter="\t"  # Important for TSV
)

print(dataset)