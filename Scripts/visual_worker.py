import matplotlib.pyplot as plt
import numpy as np

__BINS__ = np.arange(0, 257, 1)


def plot_descriptors(x, y, title, x_label, y_label, name, output_path):
    plt.plot(x, y, "+", color="red")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()

    plt.savefig(f"{output_path}/GRAPHS/{name}.png")

    plt.clf()
    plt.close()

    return None


def histogram_image(data, title, x_label, y_label, name, output_path):
    data = list(data.ravel())

    plt.hist(data, bins=__BINS__, facecolor="green")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(f"{output_path}/GRAPHS/{name}.png")

    plt.clf()
    plt.close()

    return None


if __name__ == "__main__":
    print("Hello, home!")
