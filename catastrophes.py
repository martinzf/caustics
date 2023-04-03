import numpy as np
import matplotlib.pyplot as plt

grid = plt.GridSpec(2, 6, wspace=0.4, hspace=0.3)
ax1 = plt.subplot(grid[0, 0:2])
ax2 = plt.subplot(grid[0, 2:4])
ax3 = plt.subplot(grid[0, 4:6])
ax4 = plt.subplot(grid[1, 0:3])
ax5 = plt.subplot(grid[1, 3:6])
axes = [ax1, ax2, ax3, ax4, ax5]

ax1.plot(0, 0, '.')

x = np.linspace(-1, 1, 100)
y = np.abs(x) ** (2/3)
ax2.plot(x, y)

t = np.linspace(-1.75, 1.75, 100)
x = 2 * t * (5 - 2 * t ** 2)
y = t ** 2 * (3 * t ** 2  - 5)
ax3.plot(x, y)

t = np.linspace(0, 2 * np.pi, 100) 
t += 1
x = 2 * np.cos(t) + np.cos(2 * t)
y = 2 * np.sin(t) - np.sin(2 * t) 
xy = np.array([x, y])
theta = np.pi / 2
c, s = np.cos(theta), np.sin(theta)
M = np.array([[c, -s], [s, c]])
r = M @ xy
ax4.plot(r[0], r[1])

x1 = np.linspace(- 1.2, 1.2, 100)
y1 = - .6 * x1 ** 2
x2 = np.linspace(-1, 1, 100)
y2 = - np.abs(x2) ** (2/3)
ax5.plot(x1, y1, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
ax5.plot(x2, y2, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])

titles = ['Fold', 'Cusp', 'Swallowtail', 'Elliptic Umbilic Section', 'Hyperbolic Umbilic Section']
codim = [1, 2, 3, 3, 3]
for i in range(5):
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(titles[i])
    axes[i].text(.1, .85, f'codim={codim[i]}', bbox=dict(facecolor='lightgrey'), transform=axes[i].transAxes)

plt.show()
