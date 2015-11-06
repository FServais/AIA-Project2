from sklearn.utils import check_random_state
import numpy as np


def generate_data(n_samples, r1, r2, sigma, random_state=0):
    random_state = check_random_state(random_state)

    y = random_state.randint(0, 2, size=n_samples)

    X = np.zeros((n_samples ,2))

    for i in range(0, n_samples):
        alpha = random_state.uniform(low=0.0, high=2*np.pi)

        rand_bit = random_state.randint(0, 2)

        r = random_state.normal(loc=(1 + rand_bit)*(r1 if y[i] == 0 else r2), scale=sigma)

        X[i, 0] = r * np.cos(alpha)
        X[i, 1] = r * np.sin(alpha)

    return X, y

if __name__ == "__main__":
    X, y = generate_data(2000, 0.7, 0.5, 0.1)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=50, cmap=plt.cm.cool)
    plt.savefig('bayes_data.png')
    plt.close()
