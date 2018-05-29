import matplotlib.pyplot as plt
import numpy as np


def plot_codes(ax, codes, labels):
    ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min() - .1, codes.max() + .1)
    ax.set_ylim(codes.min() - .1, codes.max() + .1)
    ax.tick_params(
        axis='both', which='both', left='off', bottom='off',
        labelleft='off', labelbottom='off')


def main():
    # fig, ax = plt.subplots(nrows=20, ncols=11, figsize=(10, 20))
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 20))
    codes = np.random.normal(size=(10000, 2))
    labels = np.random.randint(0, 10, size=(10000,))
    labels = np.zeros_like(labels)
    
    epoch = 0
    plot_codes(ax[epoch, 0], codes, labels)
    plt.show()


if __name__ == '__main__':
    main()