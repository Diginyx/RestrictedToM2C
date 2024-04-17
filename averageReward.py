import matplotlib.pyplot as plt
import numpy as np

# Path to your log file
filepath = ''
file = open(filepath, 'r')
lines = file.readlines()
# List to store mean rewards
mean_rewards = []

# Read the file
count = 1
for line in lines:
    # Find the start of the mean reward section
    start = line.find('mean reward')
    if start != -1:
        start += 12
        # Find the end which is the start of std reward section
        end = line.find(',', start)
        # Extract and convert the mean reward to float
        mean_reward = float(line[start:end].strip())
        # Append to the list
        mean_rewards.append(mean_reward)
        count += 1

# Plotting the mean rewards
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, count), mean_rewards, color='b')
plt.title('Mean Reward Over Time')
plt.xlabel('Training Step')
plt.ylabel('Mean Reward')
plt.grid(True)
plt.show()