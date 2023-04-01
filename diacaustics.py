import numpy as np
from matplotlib import pyplot as plt

def brute_force(f, i, mu, xlims, ylims, N):
    # Plotting y(x) boundary
    xmin, xmax = xlims
    ymin, ymax = ylims
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, f(x))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # Calculating refraction
    x = np.linspace(xmin, xmax, N)
    dydx = np.gradient(f(x), x)
    n = - np.array([- dydx, np.ones(N)]) / np.sqrt(dydx ** 2 + 1)
    i = np.reshape(i, (2, 1))
    d = mu * i - (mu * i.T @ n + np.sqrt(1 - mu ** 2 * (1 - (i.T @ n) ** 2))) * n
    for k in range(N):
        # Plotting incident rays
        plt.plot([x[k], x[k] - 20 * i[0, 0]], [f(x[k]), f(x[k]) - 20 * i[1, 0]], linewidth = 0.5, color='r')
        # Plotting refracted rays
        plt.plot([x[k], x[k] + 20 * d[0, k]], [f(x[k]), f(x[k]) + 20 * d[1, k]], linewidth = 0.5, color='r')
    plt.gca().set_aspect('equal')

f = lambda x: - np.sqrt(1 - x ** 2)
i = np.array([[0], [1]])
mu = 1 / 1.3325
xlims = [-1, 1]
ylims = [-1, 3.5]
n = 40
brute_force(f, i, mu, xlims, ylims, n)
plt.xlabel('X')
plt.ylabel('Y')

t = np.linspace(np.pi, 2 * np.pi, 100)
c, s = np.cos(t), np.sin(t)
x = mu ** 2 * np.cos(t) ** 3
y = (- mu ** 3 * c ** 4 + (mu * c) ** 2 * s * np.sqrt(1 - (mu * c) ** 2) + mu) / (mu * s + np.sqrt(1 - (mu * c) ** 2))
plt.plot(x, y, 'g--')
plt.savefig('images/glass_refract.png', bbox_inches='tight')
plt.show()