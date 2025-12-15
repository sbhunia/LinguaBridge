import os
import pandas as pd
import logging
import evaluate
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
# Point this to the DIRECTORY containing your .pte file
EXECUTORCH_MODEL_DIR = "/home/yogesh/LinguaBridge/test/llama-3/optimum-executorch/llama2_export/"

# --- 2. Load Model & Tokenizer ---
logging.info(f"Loading ExecuTorch Llama model from: {EXECUTORCH_MODEL_DIR}")

try:
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/yogesh/LinguaBridge/test/llama-3/myaamia/llama-3.2-1b-en-mia-translation-merged")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = ExecuTorchModelForCausalLM.from_pretrained(EXECUTORCH_MODEL_DIR)
    
except Exception as e:
    raise RuntimeError(f"Failed to load model. Error: {e}")

# --- 3. Data Loading ---
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
        df["mia"] = df["mia"].astype(str).str.strip()
        df["en"] = df["en"].astype(str).str.strip()
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        logging.error(f"Data file not found at: {path}")
        return None

data_files = [
    "../../../../data/basic_forms.tsv",
    "../../../../data/command_forms.tsv",
    "../../../../data/sentences.tsv",
    "../../../../data/story_book.tsv",
]

datasets = [load_tsv(path) for path in data_files]
valid_datasets = [ds for ds in datasets if ds is not None]

if not valid_datasets:
    raise ValueError("No valid datasets loaded.")

full_dataset = concatenate_datasets(valid_datasets)
full_dataset = full_dataset.shuffle(seed=42)
dataset_splits = full_dataset.train_test_split(test_size=0.1)
test_data = dataset_splits["test"]

# --- 4. Inference Function (Aligned with your Definition) ---
def run_llama_inference(text_input):
    system_prompt = "You are an expert translator. Translate the given English text into Myaamia."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"translate english to myaamia: {text_input}"}
    ]
    
    # 1. Tokenize to PyTorch Tensor first
    input_ids_tensor = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )
    
    # 2. Convert Tensor to List[int] (Required by your model version)
    prompt_tokens = input_ids_tensor[0].tolist()

    try:
        # 3. Generate using the EXACT signature you found
        # (prompt_tokens: List[int], echo: bool, max_seq_len: int)
        
        # Calculate max_seq_len (Prompt Length + 128 New Tokens)
        max_len = len(prompt_tokens) + 128
        
        generated_tokens = model.generate(
            prompt_tokens=prompt_tokens,
            echo=False,         # Don't return the input prompt in the output
            max_seq_len=max_len # Hard limit for buffer size
        )

        # 4. Decode
        # Since echo=False, generated_tokens should contain ONLY the new text
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text.strip()

    except Exception as e:
        logging.error(f"Error evaluating input '{text_input}': {e}")
        return ""

# --- 5. Evaluation Loop ---
logging.info("Starting evaluation...")
all_predictions = []
all_references = []

# Debug print
print("\n--- TEST RUN (First Item) ---")
first_ex = test_data[0]
print(f"Input: {first_ex['en']}")
pred = run_llama_inference(first_ex['en'])
print(f"Prediction: '{pred}'")
print("-" * 30 + "\n")

for example in tqdm(test_data, desc="Evaluating"):
    prediction = run_llama_inference(example['en'])
    all_predictions.append(prediction)
    all_references.append([example['mia']])

# --- 6. Metrics ---
sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results = sacrebleu.compute(predictions=all_predictions, references=all_references)

chrf = evaluate.load("chrf") 
chrf_results = chrf.compute(predictions=all_predictions, references=all_references, word_order=2, beta=2.0)

# Save
model_name = os.path.basename(os.path.normpath(EXECUTORCH_MODEL_DIR))
output_filename = f"eval_results_executorch_llama_{model_name}.txt"

result_text = (
    "\n" + "="*30 + "\n"
    f"RESULTS FOR: {model_name}\n"
    + "="*30 + "\n"
    f"SacreBLEU:     {sacrebleu_results['score']:.4f}\n"
    f"ChrF++:        {chrf_results['score']:.4f}\n"
    + "="*30 + "\n"
)

print(result_text)
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(result_text)