

class Viz:
    
    def __init__(self):
        pass
    
    def generator_images(self, 
                    folder, 
                    ranges = None,
                    number_of_loops = 200, 
                    averaging_number = 100,
                    graph_layout = [2,2], 
                    y_lim = [-2,12],
                    xlabel = '',
                    ylabel = '',
                    xvalues = None,
                    in_radon=False
                    ):
    
    folder = make_folder(folder)
    max_value = -10
    min_value  = 10
    
    for i in tqdm(range(number_of_loops)):

        # builds the figure
        fig, ax = plt.subplots(graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0), graph_layout[1],
                                figsize=(3 * graph_layout[1], 3 * (graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0))))
        ax = ax.reshape(-1)

        # loops around all of the embeddings
        for j, channel  in enumerate(self.channels):
            
            # checks if the value is None and if so skips tp next iteration
            if i is None:
                continue
                
            if xvalues is None:
                xvalues = range(self.vector_length)
            
            if ranges is None: 
                ranges =  np.stack((np.min(self.embeddings,axis=0),
                            np.max(self.embeddings,axis=0)), axis=1)

            # linear space values for the embeddings
            value = np.linspace(ranges[j][0], ranges[j][1], 
                                number_of_loops)

            # finds the nearest point to the value and then takes the average
            # average number of points based on the averaging number
            idx = find_nearest(
                self.embeddings[:,channel], 
                value[i], 
                averaging_number)
            
            # computes the mean of the selected index
            gen_value = np.mean(self.embeddings[idx], axis=0)
            
            # specifically updates the value of the embedding to visualize based on the
            # linear spaced vector
            gen_value[channel] = value[i]

            # generates the loop based on the model
            generated = self.predict(gen_value).squeeze()
            min_ = np.min(generated)
            max_ = np.max(generated)
            if min_<min_value:
                min_value = min_
                
            if max_>max_value:
                max_value = max_
            

            # plots the graph
            if in_radon==True:
                generated = iradon(generated)
    
            ax[j].imshow(generated,clim=[0,6])
            ax[j].plot(3,3, marker='o', markerfacecolor=self.cmap((i + 1) / number_of_loops))
            

            # formats the graph
            ax[j].set_xlabel(xlabel)

            # gets the position of the axis on the figure
            pos = ax[j].get_position()

            # plots and formats the binary cluster map
            axes_in = plt.axes([pos.x0-0.10 , pos.y1, .12 * 4, .12 * 4])

            # plots the imagemap and formats
            axes_in.imshow(self.embeddings[:, channel].reshape(self.image.shape[0:2]), clim=ranges[j])
            axes_in.set_yticklabels('')
            axes_in.set_xticklabels('')

        ax[0].set_ylabel(ylabel)

        savefig(pjoin(folder, f'{i:04d}_maps'), printing)
        plt.close(fig)