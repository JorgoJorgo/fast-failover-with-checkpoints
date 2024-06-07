import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv('real_topos_cp.csv')

# Resilience (Success rate) as a percentage
df['success'] = df['success'] * 100

# Group by graph size and algorithm for resilience
grouped_df_resilience = df.groupby(['size', 'algorithm'])['success'].mean().reset_index()

# Group by graph size and algorithm for hops
grouped_df_hops = df.groupby(['size', 'algorithm'])['hops'].mean().reset_index()

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Resilience plot
sns.barplot(ax=axes[0], x='size', y='success', hue='algorithm', data=grouped_df_resilience, palette='muted')
axes[0].set_title('Resilience of Algorithms for Topologies from Topology Zoo')
axes[0].set_xlabel('Graph Size (n)')
axes[0].set_ylabel('Resilience (%)')
axes[0].legend(title='Algorithm')
axes[0].grid(True)

# Hops plot
sns.barplot(ax=axes[1], x='size', y='hops', hue='algorithm', data=grouped_df_hops, palette='muted')
axes[1].set_title('Hops of Algorithms for Topologies from Topology Zoo')
axes[1].set_xlabel('Graph Size (n)')
axes[1].set_ylabel('Hops')
axes[1].legend(title='Algorithm')
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
