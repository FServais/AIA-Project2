import numpy as np
from matplotlib import pyplot as plt
from pylab import meshgrid

def distrib(r, R, sigma):
    return 0.5 * 1/(np.sqrt(2 * np.pi * sigma**2)) * (np.exp(- np.square(r - R)/(2 * sigma**2)) + np.exp(- np.square(r - 2*R)/(2 * sigma**2)))
    # return 0.5 * (1/np.sqrt(4 * np.pi * sigma**2)) * np.exp(- np.square(r - 3*R))/(4 * sigma**2)

def save_2_var_plot(x, y, z, name="file.png"):
    plt.figure()
    plt.pcolor(x, y, z)
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    X = np.linspace(0, 5, num =100)
    Y = np.linspace(0, 2 * np.pi, num=100)

    r, theta = meshgrid(X, Y)

    R1 = 0.7
    R2 = 0.5
    sigma = 0.1

    h0 = (0.5 * distrib(r, R1, sigma)) + (0.5 * 1/(2*np.pi))
    h1 = (0.5 * distrib(r, R2, sigma)) + (0.5 * 1/(2*np.pi))

    C = np.zeros((len(h0), len(h1)))

    for i in range(0, len(h0)):
        for j in range(0, len(h1)):
            if h0[i][j] > h1[i][j]:
                C[i][j] = 0
            else:
                C[i][j] = 1

    print(C)
    save_2_var_plot(r, theta, C, "h.png")