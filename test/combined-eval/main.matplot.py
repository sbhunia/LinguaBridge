import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # Use 'TkAgg' or another GUI backend
import matplotlib.pyplot as plt


# --- 1. Load the data from each CSV file ---
df1 = pd.read_csv('./csv_exports/bleu.csv')
df2 = pd.read_csv('./csv_exports/meteor.csv')

# --- 2. Create the plot ---
plt.figure(figsize=(10, 6))

# Plot each DataFrame's data. You can customize colors and labels.
plt.plot(df1['Step'], df1['Value'], label='Run 1', marker='o')
plt.plot(df2['Step'], df2['Value'], label='Run 2', marker='x')

# --- 3. Customize the plot ---
plt.title('BLEU Score over Training Steps')
plt.xlabel('Training Step')
plt.ylabel('BLEU Score')
plt.grid(True)
plt.legend()

# --- 4. Display the plot ---
plt.show()
