from scipy import integrate

import numpy as np
from matplotlib import pyplot as plt
from pylab import meshgrid


def prob_r_fixed_y(r, R, sigma):
    return 0.5 * (normal_distrib(r, R, sigma) + normal_distrib(r, 2*R, sigma))


def normal_distrib(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp((- (x - mu)**2)/(2 * sigma**2))

def normal_exp(x, mu, sigma):
    return np.exp((- (x - mu)**2)/(2 * sigma**2))

def save_2_var_plot(x, y, z, name="file.png", xlabel="x", ylabel="y", title="Title"):
    plt.figure()
    plt.pcolor(x, y, z)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.savefig(name)
    plt.close()

def generalisation_error(R1, R2, sigma):
    # Predict 1
    A0 = 0
    A1 = (R1+R2)/2

    B0 = (R1+2*R2)/2
    B1 = R1+R2

    # Predict 0
    C0 = A1
    C1 = B0

    D0 = B1
    D1 = 10**9

    term_y_0_1 = normal_exp(C1, R1, sigma) - normal_exp(C0, R1, sigma)
    term_y_0_2 = normal_exp(D1, R1, sigma) - normal_exp(D0, R1, sigma)
    term_y_0_3 = normal_exp(C1, 2*R1, sigma) - normal_exp(C0, 2*R1, sigma)
    term_y_0_4 = normal_exp(D1, 2*R1, sigma) - normal_exp(D0, 2*R1, sigma)

    term_y_1_1 = normal_exp(A1, R2, sigma) - normal_exp(A0, R2, sigma)
    term_y_1_2 = normal_exp(B1, R2, sigma) - normal_exp(B0, R2, sigma)
    term_y_1_3 = normal_exp(A1, 2*R2, sigma) - normal_exp(A0, 2*R2, sigma)
    term_y_1_4 = normal_exp(B1, 2*R2, sigma) - normal_exp(B0, 2*R2, sigma)

    return 1 - (1/2) * (-(sigma**2)/np.sqrt(2 * np.pi * sigma**2)) * (term_y_0_1 + term_y_0_2 + term_y_0_3 + term_y_0_4 + term_y_1_1 + term_y_1_2 + term_y_1_3 + term_y_1_4)


def generalisation_error_int(R1, R2, sigma):
    # Predict 1
    A0 = 0
    A1 = (R1+R2)/2

    B0 = (R1+2*R2)/2
    B1 = R1+R2

    # Predict 0
    C0 = A1
    C1 = B0

    D0 = B1
    D1 = np.inf

    err = [0, 0, 0, 0]

    int_y_0_A, err[0] = integrate.quad(lambda x: x * prob_r_fixed_y(x, R1, sigma), C0, C1)
    int_y_0_B, err[1] = integrate.quad(lambda x: x * prob_r_fixed_y(x, R1, sigma), D0, D1)

    int_y_1_A, err[2] = integrate.quad(lambda x: x * prob_r_fixed_y(x, R2, sigma), A0, A1)
    int_y_1_B, err[3] = integrate.quad(lambda x: x * prob_r_fixed_y(x, R2, sigma), B0, B1)

    print("Mean error: {}".format(np.mean(err)))
    print("{} + {} + {} + {}".format(int_y_0_A, int_y_0_B, int_y_1_A, int_y_1_B))

    return 1 - 0.5 * (int_y_0_A + int_y_0_B + int_y_1_A + int_y_1_B)

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

    print("Generalisation error (integral): {}".format(generalisation_error_int(R1, R2, sigma)))
    print("Generalisation error (hand): {}".format(generalisation_error(R1, R2, sigma)))