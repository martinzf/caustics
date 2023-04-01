import numpy as np
from matplotlib import pyplot as plt

def brute_force(f, i, xlims, ylims, N):
    # Plotting y(x) boundary
    xmin, xmax = xlims
    ymin, ymax = ylims
    x = np.linspace(xmin, xmax, 100)
    #plt.plot(x, f(x))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # Calculating reflections
    x = np.linspace(xmin, xmax, N)
    dydx = np.gradient(f(x), x)
    n = np.array([- dydx, np.ones(N)]) / np.sqrt(dydx ** 2 + 1)
    i = np.reshape(i, (2, 1))
    d = i - 2 * (i.T @ n) * n
    for k in range(N):
        # Plotting incident rays
        plt.plot([x[k], x[k] - 20 * i[0, 0]], [f(x[k]), f(x[k]) - 20 * i[1, 0]], linewidth = 0.5, color='r')
        # Plotting reflected rays
        plt.plot([x[k], x[k] + 20 * d[0, k]], [f(x[k]), f(x[k]) + 20 * d[1, k]], linewidth = 0.5, color='r')
    plt.gca().set_aspect('equal')

f = lambda x: x ** 2
i = np.array([[- 1], [0]])
xlims = [-2, 0]
ylims = [0, 4]
n = 40
x = np.linspace(-2, 2, 100)
plt.plot(x, f(x))
brute_force(f, i, xlims, ylims, n)
plt.xlim(-2, 2)
t = np.linspace(-10, 0, 100)
x = 3 * t / 2 - 2 * t ** 3
y = 3 * t ** 2
plt.plot(x, y, 'g--')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().xaxis.set_major_formatter('{x:.1f}')
plt.gca().yaxis.set_major_formatter('{x:.1f}')
plt.savefig('parab_caustic.png', bbox_inches='tight')
plt.show()