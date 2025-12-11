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
markers = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>', 'p', '*'])
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
plt.title('chrF++ Score vs Epochs')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# 4. Save as PDF
# 'bbox_inches="tight"' ensures the legend isn't cut off
plt.savefig('chrF++_chart.pdf', format='pdf', bbox_inches='tight')
plt.show()