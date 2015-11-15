import numpy as np
from matplotlib import pyplot as plt
from pylab import meshgrid


def prob_r_fixed_y(r, R, sigma):
    return 0.5 * (normal_distrib(r, R, sigma) + normal_distrib(r, 2*R, sigma))


def normal_distrib(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp((- (x - mu)**2)/(2 * sigma**2))


def save_2_var_plot(x, y, z, name="file.png", xlabel="x", ylabel="y", title="Title"):
    plt.figure()
    plt.pcolor(x, y, z)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    X = np.linspace(0, 2, num=200)
    Y = np.linspace(0, 2 * np.pi, num=200)

    r, theta = meshgrid(X, Y)

    R1 = 0.7
    R2 = 0.5
    sigma = 0.1

    # P(r | y=0)
    P0 = prob_r_fixed_y(r, R1, sigma)
    # P(r | y=1)
    P1 = prob_r_fixed_y(r, R2, sigma)

    C = np.zeros((len(P0), len(P1)))

    for i in range(0, len(P0)):
        for j in range(0, len(P1)):
            if P0[i][j] > P1[i][j]:
                C[i][j] = 0
            else:
                C[i][j] = 1

    save_2_var_plot(r, theta, P0, "P0.png", "r", "theta", "P(r | y=0)")
    save_2_var_plot(r, theta, P1, "P1.png", "r", "theta", "P(r | y=1)")
    save_2_var_plot(r, theta, C, "h.png", "r", "theta", "h(r, theta)")

    # Plot the normal distributions
    x = np.linspace(0, 2.5, num=200)
    plt.figure()
    # P(r | y=0)
    plt.plot(x, prob_r_fixed_y(x, R1, sigma))
    # P(r | y=1)
    plt.plot(x, prob_r_fixed_y(x, R2, sigma))

    # Add a marker at some positions :
    values = [(R1+R2)/2, (R1+2*R2)/2, R1+R2]
    legends_values = ["(R1 + R2)/2", "(R1 + 2*R2)/2", "R1 + R2"]
    cst = np.linspace(0, 2, num=100)

    for val in values:
        plt.plot([val]*100, cst, linestyle='dashed')

    plt.legend(["P(r | y=0)", "P(r | y=1)"]+legends_values+["R1", "2R1", "R2", "2R2"])
    plt.xlabel("x")
    plt.ylabel("P(r | y=C)")
    plt.title("Normal distributions of P(r | y=C), for C = {0,1}")
    plt.savefig("Normal_distributions.png")
    plt.close()


    # Plot the comparison between normal distributions and the sum sums of the distributions
    x = np.linspace(0, 2.5, num=200)
    plt.figure()
    # P(r | y=0)
    plt.plot(x, prob_r_fixed_y(x, R1, sigma))

    plt.plot(x, normal_distrib(x, R1, sigma))
    plt.plot(x, normal_distrib(x, 2*R1, sigma))

    plt.legend(["P(r | y=0)", "P(r | y=1)", "N(R1)", "N(R2)"])
    plt.xlabel("x")
    plt.ylabel("P")
    plt.title("Normal distributions of P(r | y=C), for C = {0,1}")
    plt.savefig("compare.png")
    plt.close()