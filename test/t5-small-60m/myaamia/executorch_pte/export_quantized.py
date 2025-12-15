import logging
from optimum.exporters.executorch import main_export

# 1. Configuration
# Make sure this points to your MERGED model (the folder containing config.json and pytorch_model.bin)
MODEL_PATH = "/home/yogesh/LinguaBridge/test/t5-small-60m/myaamia/test5-mt5-large-lora-2/mt5_large_myaamia_final_merged"
OUTPUT_DIR = "mt5_large_myaamia_int8_fixed2"

print(f"Loading model from: {MODEL_PATH}")
print(f"Exporting to:       {OUTPUT_DIR}")

# 2. Run Export
# We pass the quantization parameters as keyword arguments (kwargs), 
# which main_export will forward to the xnnpack recipe.
try:
    main_export(
        model_name_or_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        task="text2text-generation",
        recipe="xnnpack",
        # --- QUANTIZATION ARGUMENTS ---
        qlinear="8da",    # 8-bit dynamic activations, 4-bit weights
        qembedding="8w",    # 8-bit embeddings
        # ------------------------------
    )
    print(f"\nSUCCESS! Check the file size in: {OUTPUT_DIR}")

except Exception as e:
    print(f"\nERROR during export: {e}")