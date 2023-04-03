
Class BE_Viz:
    
    def __init__(self, dataset, shift=None, **kwargs):
        
        self.dataset = dataset
        self.shift = shift
        
        

# class Viz:

#        def __init__(self, dataset, state='lsqf', shift=None):

#             self.shift = shift

#             self.dataset = dataset
#             self.state = state
#             self.printing = self.dataset.printing

#             self.labels = [{'title': "Amplitude",
#                             'y_label': "Amplitude (Arb. U.)",
#                             'attr': "SHO_fit_amp"},
#                            {'title': "Resonance Frequency",
#                             'y_label': "Resonance Frequency (Hz)",
#                             'attr': "SHO_fit_resonance"},
#                            {'title': "Dampening",
#                             'y_label': "Quality Factor (Arb. U.)",
#                             'attr': "SHO_fit_q"},
#                            {'title': "Phase",
#                             'y_label': "Phase (rad)",
#                             'attr': "SHO_fit_phase"}]

#         def raw_be(self, filename="Figure_1_random_cantilever_resonance_results"):

#             # Select a random point and time step to plot
#             pixel = np.random.randint(0, self.dataset.num_pix)
#             timestep = np.random.randint(self.dataset.voltage_steps)

#             # prints the pixel and time step
#             print(pixel, timestep)

#             # Plots the amplitude and phase for the selected pixel and time step
#             fig, ax = layout_fig(5, 5, figsize=(6 * 11.2, 10))

#             # constructs the BE waveform and plot
#             be_timesteps = len(self.dataset.be_waveform) / \
#                 self.dataset.be_repeats

#             # plots the BE waveform
#             ax[0].plot(self.dataset.be_waveform[: int(be_timesteps)])
#             ax[0].set(xlabel="Time (sec)", ylabel="Voltage (V)")
#             ax[0].set_title("BE Waveform")

#             # plots the resonance graph
#             resonance_graph = np.fft.fft(
#                 self.dataset.be_waveform[: int(be_timesteps)])
#             fftfreq = fftpack.fftfreq(int(be_timesteps)) * \
#                 self.dataset.sampling_rate
#             ax[1].plot(
#                 fftfreq[: int(be_timesteps) //
#                         2], np.abs(resonance_graph[: int(be_timesteps) // 2])
#             )
#             ax[1].axvline(
#                 x=self.dataset.be_center_frequency,
#                 ymax=np.max(resonance_graph[: int(be_timesteps) // 2]),
#                 linestyle="--",
#                 color="r",
#             )
#             ax[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
#             ax[1].set_xlim(
#                 self.dataset.be_center_frequency - self.dataset.be_bandwidth -
#                 self.dataset.be_bandwidth * 0.25,
#                 self.dataset.be_center_frequency + self.dataset.be_bandwidth +
#                 self.dataset.be_bandwidth * 0.25,
#             )

#             # manually set the x limits
#             x_start = 120
#             x_end = 140

#             # plots the hysteresis waveform and zooms in
#             ax[2].plot(self.dataset.hysteresis_waveform)
#             ax_new = fig.add_axes([0.52, 0.6, 0.3/5.5, 0.25])
#             ax_new.plot(np.repeat(self.dataset.hysteresis_waveform, 2))
#             ax_new.set_xlim(x_start, x_end)
#             ax_new.set_ylim(0, 15)
#             ax_new.set_xticks(np.linspace(x_start, x_end, 6))
#             ax_new.set_xticklabels([60, 62, 64, 66, 68, 70])
#             fig.add_artist(
#                 ConnectionPatch(
#                     xyA=(x_start // 2,
#                          self.dataset.hysteresis_waveform[x_start // 2]),
#                     coordsA=ax[2].transData,
#                     xyB=(105, 16),
#                     coordsB=ax[2].transData,
#                     color="green",
#                 )
#             )
#             fig.add_artist(
#                 ConnectionPatch(
#                     xyA=(x_end // 2,
#                          self.dataset.hysteresis_waveform[x_end // 2]),
#                     coordsA=ax[2].transData,
#                     xyB=(105, 4.5),
#                     coordsB=ax[2].transData,
#                     color="green",
#                 )
#             )
#             ax[2].set_xlabel("Voltage Steps")
#             ax[2].set_ylabel("Voltage (V)")

#             # plots the magnitude spectrum for and phase for the selected pixel and time step
#             ax[3].plot(
#                 original_x,
#                 self.dataset.get_spectra(
#                     self.dataset.magnitude_spectrum_amplitude, pixel, timestep),
#             )
#             ax[3].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
#             ax2 = ax[3].twinx()
#             ax2.plot(
#                 original_x,
#                 self.dataset.get_spectra(
#                     self.dataset.magnitude_spectrum_phase, pixel, timestep),
#                 "r+",
#             )
#             ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

#             # plots the real and imaginary components for the selected pixel and time step
#             ax[4].plot(original_x, self.dataset.get_spectra(
#                 self.dataset.complex_spectrum_real, pixel, timestep), label="Real")
#             ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
#             ax3 = ax[4].twinx()
#             ax3.plot(
#                 original_x, self.dataset.get_spectra(
#                     self.dataset.complex_spectrum_imag, pixel, timestep), 'r', label="Imaginary")
#             ax3.set(xlabel="Frequency (Hz)", ylabel="Imag (Arb. U.)")

#             # saves the figure
#             self.printing.savefig(
#                 fig, filename, tight_layout=False)

#         def SHO_hist(self, filename="Figure_3_SHO_fit_results_before_scaling", data_type=None):

#             if data_type == 'scaled':
#                 postfix = '_scaled'
#             else:
#                 postfix = ''

#             # check distributions of each parameter before and after scaling
#             fig, axs = layout_fig(4, 4, figsize=(20, 4))

#             for ax, label in zip(axs.flat, self.labels):
#                 data = getattr(self.dataset, label['attr'] + postfix)
#                 if label['attr'] == "SHO_fit_phase" and self.shift is not None and postfix == "":
#                     data = self.shift_phase(data)

#                 ax.hist(data.flatten(), 100)
#                 ax.set(xlabel=label['y_label'], ylabel="counts")
#                 ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

#             plt.tight_layout()

#             self.printing.savefig(fig, filename)

#         def SHO_loops(self, pix=None, filename="Figure_2_random_SHO_fit_results"):
#             if pix is None:
#                 # selects a random pixel to plot
#                 pix = np.random.randint(0, 3600)

#             # plots the SHO fit results for the selected pixel
#             fig, ax = layout_fig(4, 4, figsize=(30, 6))

#             for ax, label in zip(ax, self.labels):

#                 data = getattr(
#                     self.dataset, label['attr'])[pix, :]

#                 if label['attr'] == "SHO_fit_phase" and self.shift is not None:
#                     data = self.shift_phase(data)

#                 ax.plot(self.dataset.dc_voltage, data)
#                 ax.set_title(label['title'])
#                 ax.set_ylabel(label['y_label'])

#             fig.tight_layout()
#             self.printing.savefig(fig, filename)

#         def shift_phase(self, phase, shift_=None):

#             if shift_ is None:
#                 shift = self.shift
#             else:
#                 shift = shift_

#             phase_ = phase.copy()
#             phase_ += np.pi
#             phase_[phase_ <= shift] += 2 *\
#                 np.pi  # shift phase values greater than pi
#             return phase_ - shift - np.pi

#         def raw_data(self,
#                      original,
#                      predict,
#                      predict_label=None,
#                      filename=None):

#             if predict_label is not None:
#                 predict_label = ' ' + predict_label

#             if len(original) == len(self.dataset.wvec_freq):
#                 original_x = self.dataset.wvec_freq
#             elif len(original) == len(original_x):
#                 original_x = self.dataset.frequency_bins
#             else:
#                 raise ValueError(
#                     "original data must be the same length as the frequency bins or the resampled frequency bins")

#             # plot real and imaginary components of resampled data
#             fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

#             def plot_curve(axs, x, y, label, color, key=''):
#                 axs.plot(
#                     x,
#                     y,
#                     key,
#                     label=label,
#                     color=color,
#                 )

#             plot_curve(axs[0], original_x,
#                        np.abs(original),
#                        "amplitude", 'b')

#             plot_curve(axs[0], self.dataset.wvec_freq,
#                        np.abs(predict),
#                        f"amplitude {predict_label}", 'b', key='o')

#             axs[0].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

#             ax2 = axs[0].twinx()

#             plot_curve(ax2, original_x,
#                        np.angle(original),
#                        label="phase", color='r', key='s')

#             plot_curve(ax2, self.dataset.wvec_freq,
#                        np.angle(predict),
#                        label=f"phase {predict_label}", color='r')

#             ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

#             plot_curve(axs[1], original_x,
#                        np.real(original),
#                        "real", 'b', key='o')

#             plot_curve(axs[1], self.dataset.wvec_freq,
#                        np.real(predict),
#                        f"real {predict_label}", 'b')

#             axs[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

#             ax3 = axs[1].twinx()

#             plot_curve(ax3, original_x,
#                        np.imag(original),
#                        label="imaginary",
#                        color='r', key='s')

#             plot_curve(ax3, self.dataset.wvec_freq,
#                        np.imag(predict),
#                        label=f"imaginary {predict_label}", color='r')

#             ax3.set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

#             fig.legend(bbox_to_anchor=(1.16, 0.93),
#                        loc="upper right", borderaxespad=0.0)
#             if filename is not None:
#                 self.dataset.printing.savefig(fig, filename)

#         def raw_resampled_data(self, filename="Figure_4_raw_and_resampled_raw_data"):

#             # Select a random point and time step to plot
#             pixel = np.random.randint(0, self.dataset.num_pix)
#             timestep = np.random.randint(self.dataset.voltage_steps)

#             self.raw_data(self.dataset.raw_data.reshape(self.dataset.num_pix, -1, self.dataset.num_bins)[pixel, timestep],
#                           self.dataset.raw_data_resampled[pixel, timestep],
#                           predict_label=' resampled',
#                           filename=filename)
