import numpy as np
from matplotlib import pyplot as plt

def caustic(u: np.array, v: np.array, x: np.array, y: np.array, d: callable, mu: float):
    # Determines the direction of deflected rays and plots them
    # (u, v) determine position along deflecting surface
    # (x, y) determine direction of incident rays
    # Calculating deflection
    u_d = np.gradient(u)
    v_d = np.gradient(v)
    i = np.array([x, y]) / np.sqrt(x ** 2 + y ** 2) # Normalised incident rays
    n = np.array([- v_d, u_d]) / np.sqrt(u_d ** 2 + v_d ** 2) # Normal direction (inward/outward)
    i_dot_n = np.einsum('ij, ij->j', i, n)
    sgn = - np.sign(i_dot_n) # Orientation of rays relative to normal
    n = sgn * n
    i_dot_n = sgn * i_dot_n
    d = d(i, n, i_dot_n, mu) # Direction of deflected rays
    # Plotting
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    plt.xlim(xlim)
    plt.ylim(ylim)
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]
    lims = np.max([xrange, yrange])
    # Incident rays
    plt.plot(
        [u, u - 2 * lims * i[0]],
        [v, v - 2 * lims * i[1]],
        linewidth = 0.5, color='r'
    )
    # Deflected rays
    plt.plot(
        [u, u + 2 * lims * d[0]], 
        [v, v + 2 * lims * d[1]], 
        linewidth = 0.5, color='r'
    )
    return d

def catacaustic(u: np.array, v: np.array, x: np.array, y: np.array):
    # Determines the direction of reflected rays and plots them
    # (u, v) determine position along reflector
    # (x, y) determine direction of incident rays
    d = lambda i, n, i_dot_n, _: \
        i - 2 * i_dot_n * n
    return caustic(u, v, x, y, d, None)

def diacaustic(u: np.array, v: np.array, x: np.array, y: np.array, mu: float):
    # Determines the direction of refracted rays and plots them
    # (u, v) determine position along refracting surface
    # (x, y) determine direction of incident rays
    # mu is the refractive index ratio
    d = lambda i, n, i_dot_n, mu: \
        mu * i - (mu * i_dot_n + np.sqrt(1 - mu ** 2 * (1 - i_dot_n ** 2))) * n
    return caustic(u, v, x, y, d, mu)