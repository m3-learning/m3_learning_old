from ...util.file_IO import make_folder
from ...viz.layout import layout_fig
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from ...viz.layout import layout_fig, imagemap, labelfigs, scalebar


class Viz:

    def __init__(self,
                 #  model,
                 #  scaled_data,
                 #  embeddings,
                 #  image,
                 #  channels=None,
                 #  color_map='viridis',
                 printer=None,
                 labelfigs_=False):

        self.printer = printer
        self.labelfigs_ = labelfigs_
        # self.model = model
        # self.image = image

        # # defines the color list
        # self.cmap = plt.get_cmap(color_map)
        # self.embeddings = embeddings
        # self.vector_length = scaled_data.shape[1]

        # if channels == None:
        #     self.channels = range(self.embeddings.shape[1])
        # else:
        #     self.channels = channels

    def STEM_raw_and_virtual(self, data, bright_field_=None, dark_field_=None, scalebar_=None, filename=None):

        fig_num = 1
        if bright_field_ is not None:
            fig_num += 1
        if dark_field_ is not None:
            fig_num += 1

        shape_ = data.data.shape

        fig, axs = layout_fig(fig_num, fig_num, figsize=(1.25*fig_num, 1.25))

        imagemap(axs[0], np.mean(data.log_data.reshape(-1,
                                                       shape_[2], shape_[3]), axis=0), divider_=False)

        if bright_field_ is not None:
            bright_field = data.data.reshape(-1, 256, 256)[:, bright_field_[0]:bright_field_[
                1], bright_field_[2]:bright_field_[3]]
            bright_field = np.mean(bright_field.reshape(
                shape_[0]*shape_[1], -1), axis=1).reshape(shape_[0], shape_[1])
            imagemap(axs[1], bright_field, divider_=False)

        if dark_field_ is not None:
            dark_field = data.data.reshape(-1, 256, 256)[:, dark_field_[0]:dark_field_[
                1], dark_field_[2]:dark_field_[3]]
            dark_field = np.mean(dark_field.reshape(
                shape_[0]*shape_[1], -1), axis=1).reshape(shape_[0], shape_[1])
            imagemap(axs[2], dark_field, divider_=False)

        if self.labelfigs_:
            for i, ax in enumerate(axs):
                labelfigs(ax, i)

        if scalebar_ is not None:
            scalebar(axs[2], scalebar_['width'], scalebar_[
                'scale length'], units=scalebar_['units'])

        if self.printer is not None:
            self.printer.savefig(fig, filename, tight_layout=False)

    def find_nearest(self, array, value, averaging_number):

        idx = (np.abs(array-value)).argsort()[0:averaging_number]
        return idx

    def predictor(self, values):
        values = torch.from_numpy(np.atleast_2d(values))
        values = self.model(values.float())
        values = values.detach().numpy()
        return values

    def generator_images(self,
                         folder_name='',
                         ranges=None,
                         number_of_loops=200,
                         averaging_number=100,
                         graph_layout=[2, 2],
                         # y_lim = [-2,12],
                         xlabel='',
                         ylabel='',
                         xvalues=None,
                         in_radon=False
                         ):

        folder = make_folder(
            self.printer.basepath + f"/generator_images_{folder_name}/")
        # max_value = -10
        # min_value  = 10

        for i in tqdm(range(number_of_loops)):

            fig, ax = layout_fig(graph_layout[0], graph_layout[1])

            # # builds the figure
            # fig, ax = plt.subplots(graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0), graph_layout[1],
            #                         figsize=(3 * graph_layout[1], 3 * (graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0))))
            # ax = ax.reshape(-1)

            # loops around all of the embeddings
            for j, channel in enumerate(self.channels):

                # checks if the value is None and if so skips tp next iteration
                if i is None:
                    continue

                if xvalues is None:
                    xvalues = range(self.vector_length)

                if ranges is None:
                    ranges = np.stack((np.min(self.embeddings, axis=0),
                                       np.max(self.embeddings, axis=0)), axis=1)

                # linear space values for the embeddings
                value = np.linspace(ranges[j][0], ranges[j][1],
                                    number_of_loops)

                # finds the nearest point to the value and then takes the average
                # average number of points based on the averaging number
                idx = self.find_nearest(
                    self.embeddings[:, channel],
                    value[i],
                    averaging_number)

                # computes the mean of the selected index
                gen_value = np.mean(self.embeddings[idx], axis=0)

                # specifically updates the value of the embedding to visualize based on the
                # linear spaced vector
                gen_value[channel] = value[i]

                # generates the loop based on the model
                generated = self.predict(gen_value).squeeze()
                # min_ = np.min(generated)
                # max_ = np.max(generated)
                # if min_<min_value:
                #     min_value = min_

                # if max_>max_value:
                #     max_value = max_

                # plots the graph
                if in_radon == True:
                    generated = iradon(generated)

                ax[j].imshow(generated, clim=[0, 6])
                ax[j].plot(3, 3, marker='o', markerfacecolor=self.cmap(
                    (i + 1) / number_of_loops))

                # formats the graph
                ax[j].set_xlabel(xlabel)

                # gets the position of the axis on the figure
                pos = ax[j].get_position()

                # plots and formats the binary cluster map
                axes_in = plt.axes([pos.x0-0.10, pos.y1, .12 * 4, .12 * 4])

                # plots the imagemap and formats
                axes_in.imshow(self.embeddings[:, channel].reshape(
                    self.image.shape[0:2]), clim=ranges[j])
                axes_in.set_yticklabels('')
                axes_in.set_xticklabels('')

            ax[0].set_ylabel(ylabel)

            # TODO: add the save figure function
