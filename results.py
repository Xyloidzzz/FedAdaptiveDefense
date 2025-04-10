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

# Create loss plot
plt.figure(1, figsize=(10, 6))
plt.plot(rounds, losses, 'b-o')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.title(f'Loss vs. Round for {strategy_name}')
plt.grid(True)
plt.savefig(f'results/{strategy_name}_loss.png')

# Create accuracy plot
plt.figure(2, figsize=(10, 6))
plt.plot(rounds, accuracies, 'r-o')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs. Round for {strategy_name}')
plt.grid(True)
plt.savefig(f'results/{strategy_name}_accuracy.png')

# Show all plots
plt.show()