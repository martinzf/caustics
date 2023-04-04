import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

N = 256 # Grid resolution, N ** 2 points
SX = 1 / 5 # X Roughness
SY = 1 / 5 # Y Roughness
FLAT = 50 # Flattening factor
D = 5 # Distance to screen

x = np.linspace(- 1, 1, N)
y = np.linspace(- 1, 1, N)
X, Y = np.meshgrid(x, y)

# Generating random wavefront
bell_curve = np.exp(- (X ** 2 / (2 * SX ** 2) + Y ** 2 / (2 * SY ** 2)))
randn_phase = np.exp(2j * np.pi * np.random.randn(*np.shape(X)))
complex_amplitude = signal.convolve(randn_phase, bell_curve, 'same')
mean_sqr_amplitude = np.sqrt(np.mean(np.abs(complex_amplitude) ** 2))
Z = np.real(complex_amplitude / mean_sqr_amplitude) / FLAT

# Jacobian and inverse Jacobian
O1, O2 = np.gradient(- Z, x, y)
O1x, O1y = np.gradient(O1, x, y)
O2x, O2y = np.gradient(O2, x, y)
Jacobian = O1x * O2y - O1y * O2x
invJacobian = np.abs(1 / Jacobian)

# Far field intensity
I = np.zeros((N, N))
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        # Where rays land on the screen
        a = xi + (D - Z[i, j]) * O1[i, j]
        b = yj + (D - Z[i, j]) * O2[i, j]
        # Intensity
        if np.abs(a) > 1 or np.abs(b) > 1:
            continue
        indx = np.argmin(np.abs(x - a))
        indy = np.argmin(np.abs(y - b))
        I[indx, indy] += 1
        

# Plotting
fig1, ax1 = plt.subplots()
im1 = ax1.imshow(Z, cmap='seismic', origin='lower')
plt.colorbar(im1, ax=ax1, label='Z')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

fig2, ax2 = plt.subplots()
im2 = ax2.imshow(invJacobian, cmap='gray', clim=(0, 200), origin='lower')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

fig3, ax3 = plt.subplots()
im3 = ax3.imshow(I, cmap='gray', clim=(0, 12), origin='lower', interpolation='bicubic')
ax3.set_xlabel("X'")
ax3.set_ylabel("Y'")

step = N // 5
axes = [ax1, ax2, ax3]
for ax in axes:
    ax.set_xticks(np.arange(0, N, step), [f'{i:.1f}' for i in x[::step]])
    ax.set_yticks(np.arange(0, N, step), [f'{i:.1f}' for i in y[::step]])

plt.show()