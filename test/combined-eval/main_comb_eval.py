import pandas as pd
import glob
import os
from tensorboard.plugins.hparams import api as hp
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter  # A more straightforward way to write events

# Assuming your CSV files are in a directory named 'csv_exports'
csv_files = glob.glob(os.path.join('csv_exports', '*.csv'))

if not csv_files:
    print("No CSV files found in the 'csv_exports' directory.")
else:
    # --- 1. Combine the data into a single DataFrame ---
    dfs = []
    for file in csv_files:
        print("Files:",file)
        df = pd.read_csv(file)
        dfs.append(df)

    # Use 'concat' to stack the DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # --- 2. Aggregate the data by step (e.g., calculate the mean) ---
    # Group by the 'Step' and calculate the mean of the 'Value'
    aggregated_df = combined_df.groupby('Step')['Value'].mean().reset_index()
    aggregated_df.rename(columns={'Value': 'Combined__Score'}, inplace=True)

    # --- 3. Write a new TensorBoard event file ---
    # Create a new directory for the combined log
    log_dir = 'combined_log/aggregate_run'
    os.makedirs(log_dir, exist_ok=True)

    # Use SummaryWriter to log the aggregated values
    writer = SummaryWriter(log_dir)
    for index, row in aggregated_df.iterrows():
        step = int(row['Step'])
        combined_bleu = row['Combined__Score']
        writer.add_scalar('Combined/mean', combined_bleu, step)
    writer.close()

    print(f"Aggregated TensorBoard log created at: {log_dir}")
    print("You can now run TensorBoard on the parent directory to view the result.")
    print("For example: tensorboard --logdir combined_log")

