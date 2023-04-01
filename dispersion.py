import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from skimage import color

N = 256

ALPHA = np.pi / 3
NG = 1.5
MUMAX = .1

# Symbolic computation of y(x, \mu)
m, a, ng, x = sp.symbols(r'\mu \alpha n_g x')
t1, t3, L1 = sp.symbols(r'\theta_1 \theta_3 L_1')
t1 = sp.asin(sp.sin(a / 2) / (1 + m))
t3 = sp.asin(ng * (1 + m) * sp.sin(a - t1))
L1 = sp.cos(t1) / sp.cos(a - t1)
y = sp.cos(a / 2) * (1 - L1) - (x - L1 * sp.sin(a / 2)) * sp.tan(t3 - a / 2)

# Taylor approximation of y(\mu)
Y0 = y.subs(m, 0)
Y1 = sp.diff(y, m).subs(m, 0)
Y2 = sp.diff(y, m, m).subs(m, 0)
Y = Y0 + Y1 * m + Y2 * m ** 2 / 2

# Obtaining coefficients
y_n = sp.N(Y.subs(a, ALPHA).subs(ng, NG)) # Numerical function
B, A = y_n.as_poly(m).all_coeffs()[2].as_poly(x).all_coeffs()
D, C = y_n.as_poly(m).all_coeffs()[1].as_poly(x).all_coeffs()
F, E = y_n.as_poly(m).all_coeffs()[0].as_poly(x).all_coeffs()
A, B, C, D, E, F = np.array([A, - B, C, - D, E, - F], dtype='float')

w = np.linspace(0, .6, N)
h = np.linspace(- .1, .2, N)
X, Y = np.meshgrid(w, h)
MUp = (D * X - C + np.sqrt((C - D * X) ** 2 - 4 * (E - F * X) * (A - B * X - Y))) / (2 * (E - F * X))
MUm = (D * X - C - np.sqrt((C - D * X) ** 2 - 4 * (E - F * X) * (A - B * X - Y))) / (2 * (E - F * X))

# Convert to wavelength
red = 750
blue = 380
WLp = .5 * (red + blue + (blue - red) * MUp / MUMAX)
WLm = .5 * (red + blue + (blue - red) * MUm / MUMAX)

# Piecewise gaussian
@np.vectorize
def g(x, m, s1, s2):
    g1 = np.exp(- 1 / 2 * (x - m) ** 2 / s1 ** 2)
    g2 = np.exp(- 1 / 2 * (x - m) ** 2 / s2 ** 2)
    return g1 if x < m else g2

# CIE colour matching functions (Wikipedia)
x = lambda l: 1.056 * g(l, 599.8, 37.9, 31) + 0.362 * g(l, 442, 16, 26.7)
y = lambda l: .821 * g(l, 568.8, 46.9, 40.5) + .286 * g(l, 530.9, 16.3, 31.1)
z = lambda l: 1.217 * g(l, 437, 11.8, 36) + .681 * g(l, 459, 26, 13.8)

# Calculating CIE
rp = np.array([x(WLp), y(WLp), z(WLp)]).transpose((1, 2, 0)) # NxNx3
rm = np.array([x(WLm), y(WLm), z(WLm)]).transpose((1, 2, 0)) # NxNx3
R = rp + rm

# RGB conversion
RGB = color.xyz2rgb(R) # NxNx3

# Plotting rays
plt.imshow(RGB, origin='lower', aspect=.6, interpolation='bicubic', interpolation_stage='rgba')
plt.xlabel('X/L')
plt.ylabel('Y/L')
step = N // 4
plt.xticks(np.arange(0, N, step), [f'{i:.2f}' for i in w[::step]])
plt.yticks(np.arange(0, N, step), [f'{i:.2f}' for i in h[::step]])
plt.show()

# Plotting overlap region
mask = (np.abs(MUp) > MUMAX) + (np.abs(MUm) > MUMAX) # NxN
mask3 = np.tile(mask, [3, 1, 1]).transpose([1, 2, 0]) # NxNx3
RGB[mask3] = 0

plt.figure()
plt.imshow(RGB, origin='lower', aspect=.6, interpolation='bicubic', interpolation_stage='rgba')
plt.xlabel('X/L')
plt.ylabel('Y/L')
plt.axis('off')
plt.show()