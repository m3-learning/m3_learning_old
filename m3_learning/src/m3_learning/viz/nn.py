from m3_learning.viz.layout import layout_fig, imagemap, labelfigs, add_scalebar
#, scalebar, find_nearest

def embeddings(embedding, mod=4,
                   shape_=[255, 256, 256, 256], 
                   name="",
                   channels = None,
                   labelfigs_ = False,
                   scalebar_ = None,
                   printer = None,
                   **kwargs):
    """Plots the embeddings

    Args:
        embedding (_type_): _description_
        mod (int, optional): defines the number of columns in the figure. Defaults to 4.
        channels (bool, optional): specific channels to plot. Defaults to False.
        scalebar_ (dict, optional): add the scalebar. Defaults to None.
        shape_ (list, optional): shape of the inital image. Defaults to [265, 256, 256, 256].
        name (str, optional): filename. Defaults to "".
        channels (bool, optional): _description_. Defaults to False.
        labelfigs_ (bool, optional): _description_. Defaults to False.
        add_scalebar (_type_, optional): _description_. Defaults to None.
        printer (_type_, optional): _description_. Defaults to None.
    """        

    # sets the channels to use in the object
    if channels is None:
        channels = range(embedding.shape[1])


    # builds the figure
    fig, axs = layout_fig(len(channels), mod, **kwargs)

    # loops around the channels to plot
    for i in channels:
        # plots the imagemap and formats
        imagemap(axs[i], embedding[:, i].reshape(
            shape_[0], shape_[1]), divider_=False)

    # adds labels to the figure
    if labelfigs_:
        for i, ax in enumerate(axs):
            labelfigs(ax, i)

    # adds the scalebar
    if scalebar_ is not None:
        add_scalebar(axs.flatten()[-1], scalebar_)

    # prints the image
    if printer is not None:
        printer.savefig(fig,
            f'{name}_embedding_maps', tight_layout=False)