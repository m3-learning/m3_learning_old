import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import torch


def layout_fig(graph, mod=None):

    # """Utility function that helps lay out many figures

    # Args:
    #     graph (int): number of graphs
    #     mod (int, optional): value that assists in determining the number of rows and columns. Defaults to None.

    # Returns:
    #     tuple: figure and axis
    # """

    # Sets the layout of graphs in matplotlib in a pretty way based on the number of plots
    if mod is None:
        # Select the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7

    # builds the figure based on the number of graphs and a selected number of columns
    fig, axes = plt.subplots(
        graph // mod + (graph % mod > 0),
        mod,
        figsize=(3 * mod, 3 * (graph // mod + (graph % mod > 0))),
    )

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return fig, axes


def embedding_maps(data, image, colorbar_shown=True, c_lim=None, mod=None, title=None):
    """function that generates the embedding maps

    Args:
        data (array): embedding maps to plot
        image (array): raw image used for the sizing of the image
        colorbar_shown (bool, optional): selects if colorbars are shown. Defaults to True.
        c_lim (array, optional): sets the range for the color limits. Defaults to None.
        mod (int, optional): used to change the layout (rows and columns). Defaults to None.
        title (string, optional): Adds title to the image . Defaults to None.
    """
    fig, ax = layout_fig(data.shape[1], mod)

    for i, ax in enumerate(ax):
        if i < data.shape[1]:
            im = ax.imshow(data[:, i].reshape(image.shape[0], image.shape[1]))
            ax.set_xticklabels("")
            ax.set_yticklabels("")

            # adds the colorbar
            if colorbar_shown is True:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="10%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax, format="%.1e")

                # Sets the scales
                if c_lim is not None:
                    im.set_clim(c_lim)

    if title is not None:
        # Adds title to the figure
        fig.suptitle(title, fontsize=16, y=1, horizontalalignment="center")

    fig.tight_layout()


def imagemap(ax, data, colorbars=True, clim=None):
    """pretty way to plot image maps with standard formats

    Args:
        ax (ax): axes to write to
        data (array): data to write
        colorbars (bool, optional): selects if you want to show a colorbar. Defaults to True.
        clim (array, optional): manually sets the range of the colorbars. Defaults to None.
    """

    if data.ndim == 1:
        data = data.reshape(
            np.sqrt(data.shape[0]).astype(int), np.sqrt(data.shape[0]).astype(int)
        )

    cmap = plt.get_cmap("viridis")

    if clim is None:
        im = ax.imshow(data, cmap=cmap)
    else:
        im = ax.imshow(data, clim=clim, cmap=cmap)

    ax.set_yticklabels("")
    ax.set_xticklabels("")

    if colorbars:
        # adds the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format="%.1e")


def find_nearest(array, value, averaging_number):
    """computes the average of some n nearest neighbors

    Args:
        array (array): input array
        value (float): value to find closest to
        averaging_number (int): number of data points to use in averaging

    Returns:
        list: list of indexes of the nearest neighbors
    """
    idx = (np.abs(array - value)).argsort()[0:averaging_number]
    return idx


def latent_generator(
    model,
    embeddings,
    image,
    number,
    average_number,
    indx=None,
    ranges=None,
    x_values=None,
    y_scale=[-2.2, 4],
    device="cuda",
):
    """Plots the generator results

    Args:
        model (PyTorch object): neural network model
        embeddings (float, array): the input embedding (or output from the encoder)
        image (array): Original image, this is used to extract the size of the embedding
        number (int): number of divisions to plot
        average_number (int): number of samples to average in the generation process
        indx (list, optional): embedding indexes to use. Defaults to None.
        ranges (float, array, optional): set the ranges for the embeddings. Defaults to None.
        x_values (array, optional): allows addition of x_values. Defaults to None.
        y_scale (list, optional): Scale of the y-axis. Defaults to [-2.2, 4].
        device (str, optional): the device where the data will be processed. Defaults to 'cuda'.
    """

    # sets the colormap
    cmap = plt.cm.viridis

    if indx is None:
        embedding_small = embeddings.squeeze()
    else:
        embedding_small = embeddings[:, indx].squeeze()

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(embedding_small.shape[1] * 2, mod=3)

    # plots all of the embedding maps
    for i in range(embedding_small.shape[1]):
        im = imagemap(
            ax[i], embedding_small[:, i].reshape(image.shape[0], image.shape[1])
        )

    # loops around the number of example loops
    for i in range(number):

        # loops around the number of embeddings from the range file
        for j in range(embedding_small.shape[1]):

            if ranges is None:
                value = np.linspace(
                    np.min(embedding_small[:, j]), np.max(embedding_small[:, j]), number
                )
            else:
                # sets the linear spaced values
                value = np.linspace(0, ranges[j], number)

            idx = find_nearest(embedding_small[:, j], value[i], average_number)
            gen_value = np.mean(embeddings[idx], axis=0)
            gen_value[j] = value[i]

            # computes the generated results
            gen_value_1 = torch.from_numpy(np.atleast_2d(gen_value)).to(device)
            generated = model(gen_value_1)
            generated = generated.to("cpu")
            generated = generated.detach().numpy().squeeze()

            # plots and formats the graphs
            if x_values is None:
                ax[j + embedding_small.shape[1]].plot(
                    generated, color=cmap((i + 1) / number)
                )
            else:
                ax[j + embedding_small.shape[1]].plot(
                    x_values, generated, color=cmap((i + 1) / number)
                )

            ax[j + embedding_small.shape[1]].set_ylim(y_scale)
            # ax[j + embedding_small.shape[1]].set_yticklabels('')
            plt.tight_layout(pad=1)
