import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

n = 256
x = np.linspace(- 5, 5, n)
y = np.linspace(- 5, 5, n)
X, Y = np.meshgrid(x, y)
C = np.exp(- (X ** 2 + Y ** 2))
B = np.exp(2j * np.pi * np.random.randn(*np.shape(X)))
A = signal.convolve(B, C, 'same')
I = np.sqrt(np.mean(np.abs(A) ** 2))
Z = np.real(A / I)

O1, O2 = np.gradient(- Z)
O1x, O1y = np.gradient(O1)
O2x, O2y = np.gradient(O2)
J = O1x * O2y - O1y * O2x
D = np.abs(1 / J)

step = n // 4
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(Z, cmap='seismic', origin='lower')
plt.colorbar(im1, ax=ax1)
ax1.set_xticks(np.arange(0, n, step), [f'{i:.1f}' for i in x[::step]])
ax1.set_yticks(np.arange(0, n, step), [f'{i:.1f}' for i in y[::step]])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(D, cmap='gray', clim=(0, 5e6), origin='lower')
ax2.set_xticks(np.arange(0, n, step), [f'{i:.1f}' for i in x[::step]])
ax2.set_yticks(np.arange(0, n, step), [f'{i:.1f}' for i in y[::step]])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.show()
fig1.savefig('images/wave_profile.png', bbox_inches='tight')
fig2.savefig('images/ray_density.png', bbox_inches='tight')