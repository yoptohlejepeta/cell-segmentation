import matplotlib.pyplot as plt
import numpy as np


def histogram_1D_data(
    data,
    title,
    x_label,
    y_label,
    name,
    output_path,
    bins=15,
    exclude_first_index=True,
    txt_file=True,
):
    data_list = []
    data_len = len(data)

    if exclude_first_index:
        for i in range(1, data_len):
            data_list.append(data[i])
    else:
        for i in range(data_len):
            data_list.append(data[i])

    # Vytvoření histogramu
    hist_info = plt.hist(data_list, bins=bins, facecolor="green")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(f"{output_path}/GRAPHS/{name}.png")

    if txt_file:
        y = hist_info[0]
        x = np.copy(y)

        for i in range(bins):
            x[i] = (hist_info[1][i] + hist_info[1][i + 1]) / 2

        file = open(f"{output_path}/CSV_TXT/{name}.csv", "w")

        for i in range(bins):
            file.write(str(x[i]) + ";" + str(y[i]) + "\n")

        file.close()

    plt.clf()
    plt.close()


def histogram_2D_data(
    data, title, x_label, y_label, name, output_path, bins=15, norm=False, txt_file=True
):
    data_list = list(data.ravel())

    # Vytvoření histogramu
    hist_info = plt.hist(data_list, bins=bins, facecolor="green")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if norm:
        plt.xlim(0, 1)

    plt.savefig(f"{output_path}/GRAPHS/{name}.png")

    if txt_file:
        y = hist_info[0]
        x = np.copy(y)

        for i in range(bins):
            x[i] = (hist_info[1][i] + hist_info[1][i + 1]) / 2

        file = open(f"{output_path}/CSV_TXT/{name}.csv", "w")

        for i in range(bins):
            file.write(str(x[i]) + ";" + str(y[i]) + "\n")

        file.close()

    plt.clf()
    plt.close()


def histogram_2D_data_in_mask(
    data, title, x_label, y_label, name, output_path, bins=15, norm=False
):
    data_list = list(data.ravel())

    data_non_zero = []
    for i in range(len(data_list)):
        if data_list[i] != 0:
            data_non_zero.append(data_list[i])

    # Vytvoření histogramu
    hist_info = plt.hist(data_non_zero, bins=bins, facecolor="green")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if norm:
        plt.xlim(0, 1)

    plt.savefig(f"{output_path}/GRAPHS{name}.png")

    y = hist_info[0]
    x = np.copy(y)

    for i in range(bins):
        x[i] = (hist_info[1][i] + hist_info[1][i + 1]) / 2

    file = open(f"{output_path}/CSV_TXT{name}.csv", "w")

    for i in range(bins):
        file.write(str(x[i]) + ";" + str(y[i]) + "\n")

    file.close()

    plt.clf()
    plt.close()


def plot_3D_data(data, title, x_label, y_label, z_label, name, output_path):
    x_data = data[:, 0]
    y_data = data[:, 1]
    z_data = data[:, 2]

    ax = plt.axes(projection="3d")
    plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.scatter3D(x_data, y_data, z_data, marker="x", color="red")

    plt.savefig(f"{output_path}/GRAPHS/{name}.png")

    plt.clf()
    plt.close()


def histogram_2D_data_range(
    data, title, x_label, y_label, name, output_path, left, right, step, txt_file=True
):
    data_list = list(data.ravel())

    bins = np.arange(left, right + 2 * step, step)

    # Vytvoření histogramu
    hist_info = plt.hist(data_list, bins=bins, facecolor="green")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(f"{output_path}/GRAPHS/{name}.png")

    if txt_file:
        y = np.array(hist_info[0]) / len(data_list)
        x = hist_info[1]

        file = open(f"{output_path}/CSV_TXT/{name}.csv", "w")

        for i in range(len(y)):
            file.write(str(x[i]) + ";" + str(y[i]) + "\n")

        file.close()

    plt.clf()
    plt.close()


def write_limits(upper, lower, name, output_path):
    file = open(f"{output_path}CSV_TXT/{name}.txt", "w")

    file.write(f"{upper}\n{lower}")

    file.close()


def write_ratios(blue_brown, blue_bg, brown_bg, name, output_path):
    file = open(f"{output_path}CSV_TXT/{name}.txt", "w", encoding="utf8")

    file.write(f"Poměr modré ku (modré + hnědé) : {round(blue_brown * 100,6)} %\n")
    file.write(f"Poměr modré ku snímku : {round(blue_bg * 100,6)} %\n")
    file.write(f"Poměr hnědé ku snímku : {round(brown_bg * 100,6)} %")

    file.close()


def array_2d_to_txt(img, width, height, output_path, name):
    file = open(f"{output_path}/CSV_TXT/{name}.txt", "w")
    for i in range(height):
        for j in range(width):
            file.write(f"{int(img[i, j])} ")
        if i < height - 1:
            file.write("\n")
    file.close()
