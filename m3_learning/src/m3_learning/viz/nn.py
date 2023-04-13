from m3_learning.viz.layout import layout_fig, imagemap, labelfigs#, scalebar, find_nearest

def embeddings(embedding, mod=4,
                   channels=False, 
                   scalebar_=None,
                   shape_=[265, 256, 256, 256], 
                   name="", 
                   **kwargs):
        """Plots the embeddings of the modle

        Args:
            mod (int, optional): defines the number of columns in the figure. Defaults to 4.
            channels (bool, optional): specific channels to plot. Defaults to False.
            scalebar_ (dict, optional): add the scalebar. Defaults to None.
            shape_ (list, optional): shape of the inital image. Defaults to [265, 256, 256, 256].
            name (str, optional): prefix of the file to save. Defaults to "".
        """

        # sets the channels to use in the object
        if channels == False:
            channels = range(embedding.shape[1])

        # sets the channels to use in the object
        if channels == True:
            channels = self.channels

        # builds the figure
        fig, axs = layout_fig(len(channels), mod, **kwargs)

        # loops around the channels to plot
        for i in channels:
            # plots the imagemap and formats
            imagemap(axs[i], embedding[:, i].reshape(
                shape_[0], shape_[1]), divider_=False)

        # adds labels to the figure
        if self.labelfigs_:
            for i, ax in enumerate(axs):
                labelfigs(ax, i)

        # adds the scalebar
        self.add_scalebar(axs.flatten()[-1], scalebar_)

        # prints the image
        if self.printer is not None:
            self.printer.savefig(fig,
                                 f'{name}_embedding_maps', tight_layout=False)