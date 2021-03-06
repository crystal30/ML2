import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def divergent(c):
    z = 0
    i = 0
    while i < 100:
        z = z ** 2 + c
        if abs(z) > 2:
            break
        i += 1

    return i


def draw_mandelbrot(center_x, center_y, size):
    x1, x2 = center_x - size, center_x + size
    y1, y2 = center_y - size, center_y + size

    x, y = np.mgrid[x1:x2:500j, y1:y2:500j]
    c = x + y * 1j
    divergent_ = np.frompyfunc(divergent, 1, 1)
    mandelbrot = divergent_(c)

    mandelbrot = mandelbrot.astype(np.float64)
    print("size = %d, mandelbort_max = %d, mandelbort_min = %d" % (size, mandelbrot.max(), mandelbrot.min()))

    plt.pcolormesh(x, y, mandelbrot, cmap=cm.jet)  # Create a pseudocolor plot with a non-regular rectangular grid.
    plt.xlim((np.min(x), np.max(x)))
    plt.ylim((np.min(y), np.max(y)))
    plt.savefig(str(size) + '.png')
    plt.show()


if __name__ == '__main__':
    interested_x = 0.33987
    interested_y = -0.575578

    # interested_x, interested_y = 0.27322626, 0.595153338
    for size in np.logspace(0, -5, 9, base=10):
        draw_mandelbrot(interested_x, interested_y, size)