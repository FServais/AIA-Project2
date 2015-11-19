from scipy import integrate

import numpy as np
from matplotlib import pyplot as plt
from pylab import meshgrid

from generate_data_bayes import generate_data


def cart2pol(x, y):
    """
    Convert cartesian coordinates into polar coordinates.
    """
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (rho, theta)


def normal_distrib(x, mu, sigma):
    """
    Value of the probability for a normal distribution N(mu, sigma**2).
    """
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp((- (x - mu)**2)/(2 * sigma**2))


def prob_r(r, R, sigma):
    """
    Probability P(r|y=C_i)
    """
    return 0.5 * (normal_distrib(r, R, sigma) + normal_distrib(r, 2*R, sigma))


def predict_bayes(X):
    """
    Prediction using the Bayes model.
    :param X: [x1 x2], cartesian coordinates
    :return: Value of the class, either 0 or 1.
    """
    r, _ = cart2pol(X[0], X[1])

    if 0 <= r < (R1+R2)/2:
        return 1
    elif (R1+R2)/2 <= r < (R1+2*R2)/2:
        return 0
    elif (R1+2*R2)/2 <= r < R1+R2:
        return 1
    elif R1+R2 <= r:
        return 0


def generalisation_error(R1, R2, sigma):
    """
    Compute the generalisation error.
    """
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

    int_y_0_A, _ = integrate.quad(lambda r: prob_r(r, R1, sigma), C0, C1)
    int_y_0_B, _ = integrate.quad(lambda r: prob_r(r, R1, sigma), D0, D1)

    int_y_1_A, _ = integrate.quad(lambda r: prob_r(r, R2, sigma), A0, A1)
    int_y_1_B, _ = integrate.quad(lambda r: prob_r(r, R2, sigma), B0, B1)

    return 1 - 0.5 * (int_y_0_A + int_y_0_B + int_y_1_A + int_y_1_B)


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
    _r = np.linspace(0, 2, num=200)
    _Y = np.linspace(0, 2 * np.pi, num=200)

    r, theta = meshgrid(_r, _Y)

    R1 = 0.7
    R2 = 0.5
    sigma = 0.1

    # ------------------------------------
    # Plotting P(r | y=0), P(r | y=1) and the associated Bayes model

    # P(r | y=0)
    P0 = prob_r(r, R1, sigma)
    # P(r | y=1)
    P1 = prob_r(r, R2, sigma)

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

    # ------------------------------------
    # Plot the normal distributions and their intersections

    x = np.linspace(0, 2.5, num=200)
    plt.figure()
    # P(r | y=0)
    plt.plot(x, prob_r(x, R1, sigma))
    # P(r | y=1)
    plt.plot(x, prob_r(x, R2, sigma))

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

    # ------------------------------------
    # Computing the generalisation error

    # Integration
    print("Generalisation error (integration): {}".format(generalisation_error(R1, R2, sigma)))

    # Experimental
    error = 0
    n_samples = 1500
    X, y = generate_data(n_samples, R1, R2, sigma)

    for s in range(n_samples):
        y_predict = predict_bayes(X[s])
        if y_predict != y[s]:
            error += 1

    print("Generalisation error (experimental): {}".format(error/n_samples))