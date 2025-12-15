import pandas as pd
import matplotlib.pyplot as plt
import itertools

# 1. Read the data
df = pd.read_csv('sacrebleu_epocs - Sheet1.csv')
df = df.dropna(axis=1, how='all')

# 2. Setup the plot
plt.figure(figsize=(10, 6))

# Style iterators
linestyles = itertools.cycle(['-', '--', '-.', ':'])
markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'p', '*'])
colors = itertools.cycle(plt.cm.tab10.colors) 

epochs = df.index + 1

# 3. Plot
for col in df.columns:
    plt.plot(
        epochs, 
        df[col], 
        label=col,
        linestyle=next(linestyles),
        marker=next(markers),
        color=next(colors),
        markevery=5,
        linewidth=2
    )

plt.ylabel('SacreBLEU Score')
plt.xlabel('Epochs')

# --- MODIFIED LEGEND ---
plt.legend(
    bbox_to_anchor=(0., 1.02, 1., .102), 
    loc='lower left',
    ncol=2,            # 1 column = 1 label per line
    mode="expand",     # Expands to full width
    borderaxespad=0.
)
# -----------------------

plt.grid(True)
plt.tight_layout()

# 4. Save as PDF
plt.savefig('sacrebleu_chart.pdf', format='pdf', bbox_inches='tight')
plt.show()