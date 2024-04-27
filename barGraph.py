import matplotlib.pyplot as plt

# Data for plotting
categories = ['2WayComm', 'Baseline']
values = [-1.4296169132409213, -1.2343694493945527]

# Creating the bar graph
plt.figure(figsize=(8, 6))
bars = plt.bar(categories, values, color=['blue', 'green'])

# Adding title and labels
plt.title('Mean reward over 100 episodes')
plt.ylabel('Reward')

plt.savefig('reward_graph2WayComm.png')

# Show the plot
plt.show()
