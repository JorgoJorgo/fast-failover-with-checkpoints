import matplotlib.pyplot as plt

# Data
average_resilience = {
    'One Tree': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.96, 0.8799999999999999, 0.6, 0.6, 0.5599999999999999], 
    'One Tree Checkpoint': [1.0, 1.0, 1.0, 1.0, 1.0, 0.96, 0.96, 0.76, 0.6799999999999999, 0.48, 0.48, 0.48], 
    'SquareOne': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9199999999999999, 0.8, 0.36, 0.36, 0.36]
}

average_hops = {
    'One Tree': [5.4, 5.0, 9.0, 9.0, 9.2, 9.4, 7.8, 7.8, 12.0, 14.6, 13.8, 13.0], 
    'One Tree Checkpoint': [6.8, 6.8, 8.8, 8.8, 8.8, 9.2, 8.4, 7.2, 9.6, 15.0, 13.5, 13.5],
    'SquareOne': [5.6, 6.0, 9.2, 9.2, 9.2, 8.4, 7.6, 7.4, 10.6, 6.75, 6.75, 5.75]
}

failure_rates = [i for i in range(1, len(average_resilience['One Tree']) + 1)]

# Define colors
colors = {
    'One Tree': 'blue',
    'One Tree Checkpoint': 'green',
    'SquareOne': 'red'
}

# Plotting Average Resilience
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for algorithm, values in average_resilience.items():
    plt.plot(failure_rates, values, marker='o', label=algorithm, color=colors[algorithm])
plt.xlabel('Failure Rate')
plt.ylabel('Resilience')
plt.title('Resilience of Algorithms for Random Generated Topologies (n=50, k=5)')
plt.legend()

# Plotting Average Hops
plt.subplot(1, 2, 2)
for algorithm, values in average_hops.items():
    plt.plot(failure_rates, values, marker='o', label=algorithm, color=colors[algorithm])
plt.xlabel('Failure Rate')
plt.ylabel('Hops')
plt.title('Hops of Algorithms for Random Generated Topologies (n=50, k=5)')
plt.legend()

plt.tight_layout()
plt.show()
