import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))

x = np.linspace(-12, 0, 2000)

# Distribution 1
a1 = 3.0
scale1 = 2.0
y1_dummy = skewnorm.pdf(x, a1, loc=0, scale=scale1)
peak1_0 = x[np.argmax(y1_dummy)]
loc1 = -6 - peak1_0
y1 = skewnorm.pdf(x, a1, loc=loc1, scale=scale1)
y1 = (y1 / y1.max()) * 0.7

# Distribution 2
a2 = 6.0
scale2 = 1.2
y2_dummy = skewnorm.pdf(x, a2, loc=0, scale=scale2)
peak2_0 = x[np.argmax(y2_dummy)]
loc2 = -8 - peak2_0
y2 = skewnorm.pdf(x, a2, loc=loc2, scale=scale2)
y2 = (y2 / y2.max()) * 0.9

ax.plot(x, y1, label='Distribution 1', color='#1f77b4', linewidth=2.5)
ax.plot(x, y2, label='Distribution 2', color='#d62728', linewidth=2.5)

ax.fill_between(x, y1, alpha=0.15, color='#1f77b4')
ax.fill_between(x, y2, alpha=0.15, color='#d62728')

# Axes limits
ax.set_xlim(0, -12)
ax.set_ylim(0, 1)

# Labels
ax.set_xlabel('Binding Free Energy', fontsize=12)
ax.set_ylabel('Density', fontsize=12)

# Vertical line at -10
ax.axvline(-10, color='black', linestyle='--', linewidth=1.5)
ax.text(-10.2, 0.5, 'Lowest energy docking pose', rotation=90, 
        verticalalignment='center', horizontalalignment='right', fontsize=12)

ax.legend(loc='upper right', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('/media/iwe20/DataSSD/pdbbind_dock/figures/energy_distributions.png', dpi=300)
print("Plot saved to /media/iwe20/DataSSD/pdbbind_dock/figures/energy_distributions.png")
