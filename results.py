import json
import matplotlib.pyplot as plt
import re

with open('results/results.json', 'r') as file:
    results = json.load(file)

# Extract strategy name from the first key
strategy_name = re.match(r'([A-Za-z]+)_round_\d+', list(results.keys())[0]).group(1)

# Prepare data
rounds = []
losses = []
accuracies = []

for key in sorted(results.keys()):
    round_num = int(key.split('_')[-1])
    rounds.append(round_num)
    losses.append(results[key]['loss'])
    accuracies.append(results[key]['cen_accuracy'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle(f'Training Results for {strategy_name}', fontsize=16)

# Plot loss
ax1.plot(rounds, losses, 'b-o')
ax1.set_xlabel('Round')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs. Round')
ax1.grid(True)

# Plot accuracy
ax2.plot(rounds, accuracies, 'r-o')
ax2.set_xlabel('Round')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs. Round')
ax2.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig(f'results/{strategy_name}_results.png')
plt.show()