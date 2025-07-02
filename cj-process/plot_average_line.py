import matplotlib.pyplot as plt
import numpy as np

# Sample data: lists of different lengths
data = [
    [1, 3, 5, 7, 9],
    [2, 4, 6, 8],
    [1, 2, 3, 4, 5, 6],
    [3, 6, 9],
    [2, 5, 7, 10, 12, 15]
]

# Find the max length of any list
max_len = max(len(lst) for lst in data)

# Create a dictionary to store index-wise values
values_at_index = {}

# Populate dictionary with values at each index
for lst in data:
    for i, value in enumerate(lst):
        if i not in values_at_index:
            values_at_index[i] = []
        values_at_index[i].append(value)

# Compute average and number of values per index
x_vals = sorted(values_at_index.keys())
y_avg = [np.mean(values_at_index[x]) for x in x_vals]
line_widths = [len(values_at_index[x]) for x in x_vals]  # Width proportional to number of lists

# Normalize line widths for visualization
min_width, max_width = 1, 6  # Min/max line thickness
normalized_widths = np.interp(line_widths, (min(line_widths), max(line_widths)), (min_width, max_width))

# Plot individual data lines
for lst in data:
    plt.plot(range(len(lst)), lst, linestyle='dotted', marker='o', alpha=0.6)

# Plot averaged line with varying thickness
for i in range(len(x_vals) - 1):
    plt.plot(
        [x_vals[i], x_vals[i + 1]],  # X range
        [y_avg[i], y_avg[i + 1]],  # Y range
        color='black',
        linewidth=line_widths[i]
    )

plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Individual Data Lines and Averaged Line with Thickness Corresponding to Data Count")
plt.savefig(f'output.png', dpi=300)
