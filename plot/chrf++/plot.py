import pandas as pd
import matplotlib.pyplot as plt
import itertools

# 1. Read the data
file_name = 'chrf++_epocs - Sheet1.csv'
df = pd.read_csv(file_name)
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

plt.ylabel('chrF++ Score')
plt.xlabel('Epochs')

# --- MODIFIED LEGEND SECTION ---
# ncol=1: Forces every label onto a new line (1 column).
# mode="expand": Forces the legend box to span the full width of the chart.
# bbox_to_anchor: Places the legend above the chart.
plt.legend(
    bbox_to_anchor=(0., 1.02, 1., .102), 
    loc='lower left',
    ncol=2, 
    mode="expand", 
    borderaxespad=0.
)
# -------------------------------

plt.grid(True)

# Tight layout usually helps, but with a large external legend, 
# you might need to rely on the savefig 'bbox_inches' parameter below.
plt.tight_layout()

# 4. Save as PDF
# bbox_inches='tight' is crucial here to ensure the tall legend isn't cut off
plt.savefig('chrF++_chart.pdf', format='pdf', bbox_inches='tight')
plt.show()