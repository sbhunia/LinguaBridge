# Myaamia Translator

This project focuses on building a **Myaamia ↔ English translator** using fine-tuned model. The dataset includes parallel sentences for training translation models between Myaamia and English.

## Dataset Overview

The dataset is provided in **TSV format** and contains the following files:

| File Name           | Number of Entries | Description             |
| ------------------- | ----------------- | ----------------------- |
| `sentences.tsv`     | 1,935             | Parallel sentence pairs |
| `command_forms.tsv` | 3,819             | Command forms           |
| `basic_forms.tsv`   | 12,892            | Basic forms             |
| `story_book.tsv`    | 455               | Storybook sentences     |

## Usage

1. Load the TSV files for fine-tuning your translation model.
2. Each TSV file contains parallel sentences, with Myaamia and English in separate columns.

## Notes

* The dataset is primarily intended for research and development of Myaamia-English translation systems.
* Do not public the data wthout prior permission from the Myaamia center.
