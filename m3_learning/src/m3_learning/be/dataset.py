from m3_learning.util.h5_util import print_tree
from BGlib import be as belib
import pyUSID as usid
import os
import sidpy
import numpy as np
import h5py
import time
from ..util.h5_util import make_dataset, make_group
from ..viz.printing import printer
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from ..viz.layout import layout_fig
from scipy.signal import resample
from scipy import fftpack
from sklearn.preprocessing import StandardScaler
from ..util.preprocessing import global_scaler
from ..nn.SHO_fitter.SHO import SHO_fit_func_torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..be.processing import convert_amp_phase


class BE_Dataset:

    def __init__(self,
                 dataset,
                 state='on',
                 basepath='./figures/',
                 testing=False,
                 resampling_bins=80):

        self.dataset = dataset
        self.state = state
        self.printing = printer(basepath=basepath)
        self.resample_bins = 80
        self.nn_scaled_state_params = False

        def pass_(*args, **kwargs):
            pass

        if testing:
            self.printing.savefig = pass_

        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                self.complex_spectrum
            except:
                pass

            try:
                self.magnitude_spectrum
            except:
                pass

            try:
                self.SHO_fit
            except:
                pass

            try:
                self.lsqf_viz()
            except:
                pass

    def print_be_tree(self):
        """Utility file to print the Tree of a BE Dataset

        Args:
            path (str): path to the h5 file
        """

        with h5py.File(self.dataset, "r+") as h5_f:

            # Inspects the h5 file
            usid.hdf_utils.print_tree(h5_f)

            # prints the structure and content of the file
            print(
                "Datasets and datagroups within the file:\n------------------------------------")
            print_tree(h5_f.file)

            print("\nThe main dataset:\n------------------------------------")
            print(h5_f)
            print("\nThe ancillary datasets:\n------------------------------------")
            print(h5_f.file["/Measurement_000/Channel_000/Position_Indices"])
            print(h5_f.file["/Measurement_000/Channel_000/Position_Values"])
            print(
                h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Indices"])
            print(
                h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Values"])

            print(
                "\nMetadata or attributes in a datagroup\n------------------------------------")
            for key in h5_f.file["/Measurement_000"].attrs:
                print("{} : {}".format(
                    key, h5_f.file["/Measurement_000"].attrs[key]))

    def SHO_Fitter(self, force=False, max_cores=-1, max_mem=1024*8):
        """Function that computes the SHO fit results

        Args:
            force (bool, optional): forces the SHO results to be computed from scratch. Defaults to False.
            max_cores (int, optional): number of processor cores to use. Defaults to -1.
            max_mem (_type_, optional): maximum ram to use. Defaults to 1024*8.
        """

        start_time_lsqf = time.time()

        (data_dir, filename) = os.path.split(self.dataset)

        if self.dataset.endswith(".h5"):
            # No translation here
            h5_path = self.dataset

            tl = belib.translators.LabViewH5Patcher()
            tl.translate(h5_path, force_patch=force)

        else:
            pass

        folder_path, h5_raw_file_name = os.path.split(h5_path)
        h5_file = h5py.File(h5_path, "r+")
        print("Working on:\n" + h5_path)

        h5_main = usid.hdf_utils.find_dataset(h5_file, "Raw_Data")[0]

        h5_pos_inds = h5_main.h5_pos_inds
        pos_dims = h5_main.pos_dim_sizes
        pos_labels = h5_main.pos_dim_labels
        print(pos_labels, pos_dims)

        h5_meas_grp = h5_main.parent.parent

        parm_dict = sidpy.hdf_utils.get_attributes(h5_meas_grp)

        expt_type = usid.hdf_utils.get_attr(h5_file, "data_type")

        is_ckpfm = expt_type == "cKPFMData"
        if is_ckpfm:
            num_write_steps = parm_dict["VS_num_DC_write_steps"]
            num_read_steps = parm_dict["VS_num_read_steps"]
            num_fields = 2

        if expt_type != "BELineData":
            vs_mode = usid.hdf_utils.get_attr(h5_meas_grp, "VS_mode")
            try:
                field_mode = usid.hdf_utils.get_attr(
                    h5_meas_grp, "VS_measure_in_field_loops")
            except KeyError:
                print("field mode could not be found. Setting to default value")
                field_mode = "out-of-field"
            try:
                vs_cycle_frac = usid.hdf_utils.get_attr(
                    h5_meas_grp, "VS_cycle_fraction")
            except KeyError:
                print("VS cycle fraction could not be found. Setting to default value")
                vs_cycle_frac = "full"

        sho_fit_points = 5  # The number of data points at each step to use when fitting
        sho_override = force  # Force recompute if True

        h5_sho_targ_grp = None
        h5_sho_file_path = os.path.join(
            folder_path, h5_raw_file_name)

        print("\n\nSHO Fits will be written to:\n" + h5_sho_file_path + "\n\n")
        f_open_mode = "w"
        if os.path.exists(h5_sho_file_path):
            f_open_mode = "r+"
        h5_sho_file = h5py.File(h5_sho_file_path, mode=f_open_mode)
        h5_sho_targ_grp = h5_sho_file

        sho_fitter = belib.analysis.BESHOfitter(
            h5_main, cores=max_cores, verbose=False, h5_target_group=h5_sho_targ_grp
        )
        sho_fitter.set_up_guess(
            guess_func=belib.analysis.be_sho_fitter.SHOGuessFunc.complex_gaussian,
            num_points=sho_fit_points,
        )
        h5_sho_guess = sho_fitter.do_guess(override=sho_override)
        sho_fitter.set_up_fit()
        h5_sho_fit = sho_fitter.do_fit(override=sho_override)
        parms_dict = parms_dict = sidpy.hdf_utils.get_attributes(
            h5_main.parent.parent)

        print(
            f"LSQF method took {time.time() - start_time_lsqf} seconds to compute parameters")

    @property
    def be_repeats(self):
        """Number of BE repeats"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f['Measurement_000'].attrs["BE_repeats"]

    @property
    def num_bins(self):
        """Number of frequency bins in the data"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_bins"]

    @property
    def num_pix(self):
        """Number of pixels in the data"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_pix"]

    @property
    def num_pix_1d(self):
        """Number of pixels in the data"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return int(np.sqrt(self.num_pix))

    @property
    def voltage_steps(self):
        """Number of DC voltage steps"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["num_udvs_steps"]

    @property
    def sampling_rate(self):
        """Sampling rate in Hz"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["IO_rate_[Hz]"]

    @property
    def be_bandwidth(self):
        """BE bandwidth in Hz"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["BE_band_width_[Hz]"]

    @property
    def be_center_frequency(self):
        """BE center frequency in Hz"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"].attrs["BE_center_frequency_[Hz]"]

    @property
    def frequency_bin(self):
        """Frequency bin vector in Hz"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Bin_Frequencies"][:]

    @property
    def wvec_freq(self):
        """Resampled frequency vector in Hz"""
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._wvec_freq
            except:
                self.wvec_freq = self.resample_bins
                return self._wvec_freq

    @wvec_freq.setter
    def wvec_freq(self, bins):
        self._wvec_freq = resample(self.frequency_bin, bins)

    @property
    def be_waveform(self):
        """BE excitation waveform"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Excitation_Waveform"][:]

    @property
    def spectroscopic_values(self):
        """Spectroscopic values"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Spectroscopic_Values"][:]

    @property
    def raw_data(self):
        """Raw data"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]["Raw_Data"][:]

    @property
    def raw_data_resampled(self):
        """Resampled real part of the complex data resampled"""
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._raw_data_resampled
            except:
                self.raw_data_resampled = None
                return self._raw_data_resampled

    @raw_data_resampled.setter
    def raw_data_resampled(self, a=None):
        with h5py.File(self.dataset, "r+") as h5_f:
            resampled_ = self.resampler(
                self.raw_data.reshape(-1, self.num_bins), axis=2)
            make_dataset(h5_f["Measurement_000"]["Channel_000"],
                         'raw_data_resampled', resampled_)
            self._raw_data_resampled = h5_f[
                "Measurement_000/Channel_000/raw_data_resampled"][:]

    @property
    def shape(self):
        """Shape of the raw data"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.raw_data.shape

    @property
    def complex_spectrum(self):
        """Complex data"""
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._complex_spectrum
            except:
                self.complex_spectrum = None
                return self._complex_spectrum

    @complex_spectrum.setter
    def complex_spectrum(self, a=None):
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                self._complex_spectrum = h5_f["Measurement_000"]["Channel_000"]['complex']
            except:
                make_group(h5_f["Measurement_000"]["Channel_000"], 'complex')
                make_dataset(h5_f["Measurement_000"]["Channel_000"]
                             ['complex'], 'real', np.real(self.raw_data))
                make_dataset(h5_f["Measurement_000"]["Channel_000"]
                             ['complex'], 'imag', np.imag(self.raw_data))
                self._complex_spectrum = h5_f["Measurement_000"]["Channel_000"]['complex']

    @property
    def complex_spectrum_real(self):
        """Real part of the complex data"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]['complex']['real'][:].reshape(self.num_pix, -1, self.num_bins)

    @property
    def complex_spectrum_imag(self):
        """Imaginary part of the complex data"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]['complex']['imag'][:].reshape(self.num_pix, -1, self.num_bins)

    @property
    def complex_spectrum_real_resampled(self):
        """Resampled real part of the complex data resampled"""
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._complex_spectrum_real_resampled
            except:
                self.complex_spectrum_real_resampled = None
                return self._complex_spectrum_real_resampled

    @complex_spectrum_real_resampled.setter
    def complex_spectrum_real_resampled(self, a=None):
        with h5py.File(self.dataset, "r+") as h5_f:
            resampled_ = self.resampler(self.complex_spectrum_real)
            make_dataset(h5_f["Measurement_000"]["Channel_000"]
                         ['complex'], 'real_resampled', resampled_)
            self._complex_spectrum_real_resampled = h5_f[
                "Measurement_000/Channel_000/complex/real_resampled"][:]

    @property
    def complex_spectrum_imag_resampled(self):
        """Resampled imag part of the complex data resampled"""
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._complex_spectrum_imag_resampled
            except:
                self.complex_spectrum_imag_resampled = None
                return self._complex_spectrum_imag_resampled

    @complex_spectrum_imag_resampled.setter
    def complex_spectrum_imag_resampled(self, a=None):
        with h5py.File(self.dataset, "r+") as h5_f:
            resampled_ = self.resampler(self.complex_spectrum_imag)
            make_dataset(h5_f["Measurement_000"]["Channel_000"]
                         ['complex'], 'imag_resampled', resampled_)
            self._complex_spectrum_imag_resampled = h5_f[
                "Measurement_000/Channel_000/complex/imag_resampled"][:]

    def resampler(self, data, axis=2):
        """Resample the data to a given number of bins"""
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return resample(data.reshape(self.num_pix, -1, self.num_bins),
                                self.resample_bins, axis=axis)
            except ValueError:
                print("Resampling failed, check that the number of bins is defined")

    @property
    def magnitude_spectrum(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._magnitude_spectrum
            except:
                self.magnitude_spectrum = None
                return self._magnitude_spectrum

    @magnitude_spectrum.setter
    def magnitude_spectrum(self, a=None):
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                self._magnitude_spectrum = h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum']
            except:
                make_group(h5_f["Measurement_000"]
                           ["Channel_000"], 'magnitude_spectrum')
                make_dataset(h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum'], 'amplitude', np.abs(
                    self.raw_data))
                make_dataset(h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum'], 'phase', np.angle(
                    self.raw_data))
                self._magnitude_spectrum = h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum']

    @property
    def magnitude_spectrum_resampled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._magnitude_spectrum_resampled
            except:
                self.magnitude_spectrum_resampled = None
                return self._magnitude_spectrum_resampled

    @magnitude_spectrum_resampled.setter
    def magnitude_spectrum_resampled(self, a=None):
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                make_group(h5_f["Measurement_000"]
                           ["Channel_000"], 'magnitude_spectrum_resampled')
            except:
                pass

            try:
                make_dataset(h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum_resampled'], 'amplitude', np.abs(
                    self._raw_data_resampled))
            except:
                pass

            try:
                make_dataset(h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum_resampled'], 'phase', np.angle(
                    self._raw_data_resampled))
            except:
                pass

            self._magnitude_spectrum_resampled = h5_f["Measurement_000"][
                "Channel_000"]['magnitude_spectrum_resampled']

    @property
    def magnitude_spectrum_amplitude(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum']['amplitude'][:].reshape(self.num_pix, -1, self.num_bins)

    @property
    def magnitude_spectrum_phase(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum']['phase'][:].reshape(self.num_pix, -1, self.num_bins)

    @property
    def magnitude_spectrum_amplitude_resampled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum_resampled']['amplitude'][:]

    @property
    def magnitude_spectrum_phase_resampled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["Measurement_000"]["Channel_000"]['magnitude_spectrum_resampled']['phase'][:]

    def get_spectra(self, data, pixel, timestep):
        """Spectra"""
        with h5py.File(self.dataset, "r+") as h5_f:
            return data.reshape(self.num_pix, -1, self.num_bins)[pixel, timestep]

    @property
    def hysteresis_waveform(self, loop_number=2):
        with h5py.File(self.dataset, "r+") as h5_f:
            return (
                self.spectroscopic_values[1, ::len(self.frequency_bin)][int(self.voltage_steps/loop_number):] *
                self.spectroscopic_values[2, ::len(
                    self.frequency_bin)][int(self.voltage_steps/loop_number):]
            )

    @property
    def dc_voltage(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["/Raw_Data-SHO_Fit_000/Spectroscopic_Values"][0, 1::2]

    @property
    def SHO_fit_on(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._SHO_fit[:, 1::2, :]

    @property
    def SHO_fit_off(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._SHO_fit[:, ::2, :]

    @property
    def SHO_fit_amp(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state()[:, :, 0]

    @property
    def SHO_fit_resonance(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state()[:, :, 1]

    @property
    def SHO_fit_q(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state()[:, :, 2]

    @property
    def SHO_fit_phase(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state()[:, :, 3]

    @property
    def SHO_fit_r2(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state()[:, :, 4]

    def SHO_state(self):
        if self.state == "on":
            return self.SHO_fit_on
        elif self.state == "off":
            return self.SHO_fit_off
        else:
            return self.SHO_fit

    @property
    def SHO_fit(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._SHO_fit
            except:
                self.SHO_fit = 5
                return self._SHO_fit

    @SHO_fit.setter
    def SHO_fit(self, channels=5):
        """Utility function to convert the SHO fit results to an array

        Args:
            SHO_fit (h5 Dataset): Location of the fit results in an h5 file

        Returns:
            np.array: SHO fit results
        """
        with h5py.File(self.dataset, "r+") as h5_f:

            # create a list for parameters
            SHO_fit_list = []
            for sublist in np.array(
                h5_f['/Raw_Data-SHO_Fit_000/Fit']
            ):
                for item in sublist:
                    for i in item:
                        SHO_fit_list.append(i)

            data_ = np.array(SHO_fit_list).reshape(
                -1, channels)

            # flatten parameters list into numpy array
            self._SHO_fit = data_.reshape(
                self.num_pix, self.voltage_steps, channels)


    class Viz:

        def __init__(self, dataset, state='lsqf', shift=None):

            self.shift = shift

            self.dataset = dataset
            self.state = state
            self.printing = self.dataset.printing

            self.labels = [{'title': "Amplitude",
                            'y_label': "Amplitude (Arb. U.)",
                            'attr': "SHO_fit_amp"},
                           {'title': "Resonance Frequency",
                            'y_label': "Resonance Frequency (Hz)",
                            'attr': "SHO_fit_resonance"},
                           {'title': "Dampening",
                            'y_label': "Quality Factor (Arb. U.)",
                            'attr': "SHO_fit_q"},
                           {'title': "Phase",
                            'y_label': "Phase (rad)",
                            'attr': "SHO_fit_phase"}]

        def raw_be(self, filename="Figure_1_random_cantilever_resonance_results"):

            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)
            timestep = np.random.randint(self.dataset.voltage_steps)

            # prints the pixel and time step
            print(pixel, timestep)

            # Plots the amplitude and phase for the selected pixel and time step
            fig, ax = layout_fig(5, 5, figsize=(6 * 11.2, 10))

            # constructs the BE waveform and plot
            be_timesteps = len(self.dataset.be_waveform) / \
                self.dataset.be_repeats

            # plots the BE waveform
            ax[0].plot(self.dataset.be_waveform[: int(be_timesteps)])
            ax[0].set(xlabel="Time (sec)", ylabel="Voltage (V)")
            ax[0].set_title("BE Waveform")

            # plots the resonance graph
            resonance_graph = np.fft.fft(
                self.dataset.be_waveform[: int(be_timesteps)])
            fftfreq = fftpack.fftfreq(int(be_timesteps)) * \
                self.dataset.sampling_rate
            ax[1].plot(
                fftfreq[: int(be_timesteps) //
                        2], np.abs(resonance_graph[: int(be_timesteps) // 2])
            )
            ax[1].axvline(
                x=self.dataset.be_center_frequency,
                ymax=np.max(resonance_graph[: int(be_timesteps) // 2]),
                linestyle="--",
                color="r",
            )
            ax[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
            ax[1].set_xlim(
                self.dataset.be_center_frequency - self.dataset.be_bandwidth -
                self.dataset.be_bandwidth * 0.25,
                self.dataset.be_center_frequency + self.dataset.be_bandwidth +
                self.dataset.be_bandwidth * 0.25,
            )

            # manually set the x limits
            x_start = 120
            x_end = 140

            # plots the hysteresis waveform and zooms in
            ax[2].plot(self.dataset.hysteresis_waveform)
            ax_new = fig.add_axes([0.52, 0.6, 0.3/5.5, 0.25])
            ax_new.plot(np.repeat(self.dataset.hysteresis_waveform, 2))
            ax_new.set_xlim(x_start, x_end)
            ax_new.set_ylim(0, 15)
            ax_new.set_xticks(np.linspace(x_start, x_end, 6))
            ax_new.set_xticklabels([60, 62, 64, 66, 68, 70])
            fig.add_artist(
                ConnectionPatch(
                    xyA=(x_start // 2,
                         self.dataset.hysteresis_waveform[x_start // 2]),
                    coordsA=ax[2].transData,
                    xyB=(105, 16),
                    coordsB=ax[2].transData,
                    color="green",
                )
            )
            fig.add_artist(
                ConnectionPatch(
                    xyA=(x_end // 2,
                         self.dataset.hysteresis_waveform[x_end // 2]),
                    coordsA=ax[2].transData,
                    xyB=(105, 4.5),
                    coordsB=ax[2].transData,
                    color="green",
                )
            )
            ax[2].set_xlabel("Voltage Steps")
            ax[2].set_ylabel("Voltage (V)")

            # plots the magnitude spectrum for and phase for the selected pixel and time step
            ax[3].plot(
                original_x,
                self.dataset.get_spectra(
                    self.dataset.magnitude_spectrum_amplitude, pixel, timestep),
            )
            ax[3].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
            ax2 = ax[3].twinx()
            ax2.plot(
                original_x,
                self.dataset.get_spectra(
                    self.dataset.magnitude_spectrum_phase, pixel, timestep),
                "r+",
            )
            ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

            # plots the real and imaginary components for the selected pixel and time step
            ax[4].plot(original_x, self.dataset.get_spectra(
                self.dataset.complex_spectrum_real, pixel, timestep), label="Real")
            ax[4].set(xlabel="Frequency (Hz)", ylabel="Real (Arb. U.)")
            ax3 = ax[4].twinx()
            ax3.plot(
                original_x, self.dataset.get_spectra(
                    self.dataset.complex_spectrum_imag, pixel, timestep), 'r', label="Imaginary")
            ax3.set(xlabel="Frequency (Hz)", ylabel="Imag (Arb. U.)")

            # saves the figure
            self.printing.savefig(
                fig, filename, tight_layout=False)

        def SHO_hist(self, filename="Figure_3_SHO_fit_results_before_scaling", data_type=None):

            if data_type == 'scaled':
                postfix = '_scaled'
            else:
                postfix = ''

            # check distributions of each parameter before and after scaling
            fig, axs = layout_fig(4, 4, figsize=(20, 4))

            for ax, label in zip(axs.flat, self.labels):
                data = getattr(self.dataset, label['attr'] + postfix)
                if label['attr'] == "SHO_fit_phase" and self.shift is not None and postfix == "":
                    data = self.shift_phase(data)

                ax.hist(data.flatten(), 100)
                ax.set(xlabel=label['y_label'], ylabel="counts")
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

            plt.tight_layout()

            self.printing.savefig(fig, filename)

        def SHO_loops(self, pix=None, filename="Figure_2_random_SHO_fit_results"):
            if pix is None:
                # selects a random pixel to plot
                pix = np.random.randint(0, 3600)

            # plots the SHO fit results for the selected pixel
            fig, ax = layout_fig(4, 4, figsize=(30, 6))

            for ax, label in zip(ax, self.labels):

                data = getattr(
                    self.dataset, label['attr'])[pix, :]

                if label['attr'] == "SHO_fit_phase" and self.shift is not None:
                    data = self.shift_phase(data)

                ax.plot(self.dataset.dc_voltage, data)
                ax.set_title(label['title'])
                ax.set_ylabel(label['y_label'])

            fig.tight_layout()
            self.printing.savefig(fig, filename)

        def shift_phase(self, phase, shift_=None):

            if shift_ is None:
                shift = self.shift
            else:
                shift = shift_

            phase_ = phase.copy()
            phase_ += np.pi
            phase_[phase_ <= shift] += 2 *\
                np.pi  # shift phase values greater than pi
            return phase_ - shift - np.pi

        def raw_data(self,
                     original,
                     predict,
                     predict_label=None,
                     filename=None):

            if predict_label is not None:
                predict_label = ' ' + predict_label

            if len(original) == len(self.dataset.wvec_freq):
                original_x = self.dataset.wvec_freq
            elif len(original) == len(original_x):
                original_x = self.dataset.frequency_bins
            else:
                raise ValueError(
                    "original data must be the same length as the frequency bins or the resampled frequency bins")

            # plot real and imaginary components of resampled data
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

            def plot_curve(axs, x, y, label, color, key=''):
                axs.plot(
                    x,
                    y,
                    key,
                    label=label,
                    color=color,
                )

            plot_curve(axs[0], original_x,
                       np.abs(original),
                       "amplitude", 'b')

            plot_curve(axs[0], self.dataset.wvec_freq,
                       np.abs(predict),
                       f"amplitude {predict_label}", 'b', key='o')

            axs[0].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

            ax2 = axs[0].twinx()

            plot_curve(ax2, original_x,
                       np.angle(original),
                       label="phase", color='r', key='s')

            plot_curve(ax2, self.dataset.wvec_freq,
                       np.angle(predict),
                       label=f"phase {predict_label}", color='r')

            ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

            plot_curve(axs[1], original_x,
                       np.real(original),
                       "real", 'b', key='o')

            plot_curve(axs[1], self.dataset.wvec_freq,
                       np.real(predict),
                       f"real {predict_label}", 'b')

            axs[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

            ax3 = axs[1].twinx()

            plot_curve(ax3, original_x,
                       np.imag(original),
                       label="imaginary",
                       color='r', key='s')

            plot_curve(ax3, self.dataset.wvec_freq,
                       np.imag(predict),
                       label=f"imaginary {predict_label}", color='r')

            ax3.set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

            fig.legend(bbox_to_anchor=(1.16, 0.93),
                       loc="upper right", borderaxespad=0.0)
            if filename is not None:
                self.dataset.printing.savefig(fig, filename)

        def raw_resampled_data(self, filename="Figure_4_raw_and_resampled_raw_data"):

            # Select a random point and time step to plot
            pixel = np.random.randint(0, self.dataset.num_pix)
            timestep = np.random.randint(self.dataset.voltage_steps)

            self.raw_data(self.dataset.raw_data.reshape(self.dataset.num_pix, -1, self.dataset.num_bins)[pixel, timestep],
                          self.dataset.raw_data_resampled[pixel, timestep],
                          predict_label=' resampled',
                          filename=filename)

    def lsqf_viz(self):
        self.lsqf_viz = self.Viz(self, state='lsqf')

    # delete a dataset
    def delete(self, name):
        with h5py.File(self.dataset, "r+") as h5_f:
            del h5_f[name]

    def SHO_Scaler(self, shifter=None,
                   save_loc='SHO_fit_scaled',
                   basepath="Raw_Data-SHO_Fit_000"):
        self.SHO_scaler = StandardScaler()
        data = self.SHO_fit.reshape(-1, 5)[:, :-1]

        if shifter is not None:
            data[:, 3] = self.lsqf_viz.shift_phase(data[:, 3], shifter)

        self.SHO_scaler.fit(data)

        # sets the phase not to scale
        self.SHO_scaler.mean_[3] = 0
        self.SHO_scaler.var_[3] = 1

        fit_results_scaled = self.SHO_scaler.transform(
            data).reshape(self.num_pix, -1, 4)

        with h5py.File(self.dataset, "r+") as h5_f:
            self._SHO_params_scaled = self.data_writer(
                basepath, save_loc, fit_results_scaled)

    @property
    def SHO_fit_on_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._SHO_fit_scaled[:, 1::2, :]

    @property
    def SHO_fit_off_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._SHO_fit_scaled[:, ::2, :]

    @property
    def SHO_fit_amp_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state_scaled()[:, :, 0]

    @property
    def SHO_fit_resonance_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state_scaled()[:, :, 1]

    @property
    def SHO_fit_q_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state_scaled()[:, :, 2]

    @property
    def SHO_fit_phase_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state_scaled()[:, :, 3]

    @property
    def SHO_fit_r2_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_state_scaled()[:, :, 4]

    def SHO_state_scaled(self):
        if self.state == "on":
            return self.SHO_fit_on_scaled
        elif self.state == "off":
            return self.SHO_fit_off_scaled

    @property
    def _SHO_fit_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return h5_f["/Raw_Data-SHO_Fit_000/SHO_fit_scaled"][:]

    @property
    def complex_spectrum_real_resampled_scaled(self):
        """Resampled real part of the complex data resampled"""
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._complex_spectrum_real_resampled_scaled
            except:
                self.complex_spectrum_real_resampled_scaled = None
                return self._complex_spectrum_real_resampled_scaled

    @complex_spectrum_real_resampled_scaled.setter
    def complex_spectrum_real_resampled_scaled(self, a=None):
        with h5py.File(self.dataset, "r+") as h5_f:
            self.real_scaler = global_scaler()
            real_data_scaled = self.real_scaler.fit_transform(
                self.complex_spectrum_real_resampled)
            try:
                make_dataset(h5_f["Measurement_000/Channel_000/complex/"],
                             'real_resampled_scaled',
                             real_data_scaled)
            except:
                pass

            self._complex_spectrum_real_resampled_scaled = h5_f[
                'Measurement_000/Channel_000/complex/real_resampled_scaled'][:]

    @property
    def complex_spectrum_imag_resampled_scaled(self):
        """Resampled imag part of the complex data resampled"""
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                return self._complex_spectrum_imag_resampled_scaled
            except:
                self.complex_spectrum_imag_resampled_scaled = None
                return self._complex_spectrum_imag_resampled_scaled

    @complex_spectrum_imag_resampled_scaled.setter
    def complex_spectrum_imag_resampled_scaled(self, a=None):
        with h5py.File(self.dataset, "r+") as h5_f:
            self.imag_scaler = global_scaler()
            imag_data_scaled = self.imag_scaler.fit_transform(
                self.complex_spectrum_imag_resampled)
            try:
                make_dataset(h5_f["Measurement_000/Channel_000/complex/"],
                             'imag_resampled_scaled',
                             imag_data_scaled)
            except:
                pass

            self._complex_spectrum_imag_resampled_scaled = h5_f[
                'Measurement_000/Channel_000/complex/imag_resampled_scaled'][:]

    def LSQF_torch_function_comparison(self, filename="Figure_6_pytorch_raw_fits"):

        # plot the initial and reconstructed by SHO Fitting Function data
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Select a random point and time step to plot
        pixel = np.random.randint(0, self.num_pix)
        timestep = np.random.randint(self.voltage_steps)

        # computes the fit using the torch model
        out = SHO_fit_func_torch(torch.tensor(np.atleast_2d(
            self.SHO_fit[pixel, timestep, 0:4])), self.wvec_freq)

        # converts to amplitude and phase
        amp_pred, phase_pred = convert_amp_phase(out)

        print(f"spectra number {pixel}")

        axs[0].plot(
            self.wvec_freq, self.magnitude_spectrum_amplitude_resampled[pixel, timestep], "o", label="Raw Amplitude", color="b"
        )
        axs[0].plot(
            self.wvec_freq, amp_pred[0, :], label="Amplitude Torch", color="b"
        )
        axs[0].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")
        ax2 = axs[0].twinx()

        ax2.plot(self.wvec_freq, self.magnitude_spectrum_phase_resampled[pixel, timestep],
                 "o", label="Raw Phase", color="r")
        ax2.plot(self.wvec_freq, phase_pred[0, :],
                 label="Phase Torch", color="r")
        ax2.set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")

        axs[1].scatter(self.wvec_freq, self.complex_spectrum_real_resampled_scaled[pixel, timestep],
                       label="Raw Real", marker="s")
        axs[1].scatter(self.wvec_freq, self.complex_spectrum_imag_resampled_scaled[pixel, timestep],
                       label="Raw Imag", marker="o")
        axs[1].plot(
            self.wvec_freq, self.real_scaler.transform(torch.real(out[0])), label="Real Torch"
        )
        axs[1].plot(
            self.wvec_freq, self.imag_scaler.transform(torch.imag(out[0])), label="Imag Torch"
        )
        axs[1].set(xlabel="Frequency (Hz)", ylabel="Amplitude (Arb. U.)")

        fig.tight_layout()
        fig.legend(bbox_to_anchor=(1.15, 0.97),
                   loc="upper right", borderaxespad=0.0)

        self.printing.savefig(fig, filename)

    @property
    def nn_raw_input(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            real = self._complex_spectrum_real_resampled_scaled.reshape(
                -1, self.resample_bins)
            imag = self._complex_spectrum_imag_resampled_scaled.reshape(
                -1, self.resample_bins)

            return np.stack((real, imag), axis=2)

    @property
    def SHO_params_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            # self.SHO_scaler.transform(self.SHO_fit.reshape(-1, 5)[:, :-1])
            return self._SHO_params_scaled

            # if hasattr(self, "nn_parms_scalar") == False:
            #     self.NN_Params_Scaler()
            # return self.nn_parms_scalar.transform(self.SHO_fit.reshape(-1, 5)[:, 0:4])

    @property
    def nn_predictions(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._nn_predictions

    @nn_predictions.setter
    def nn_predictions(self, data):
        with h5py.File(self.dataset, "r+") as h5_f:
            self._nn_predictions = self.data_writer(
                "Raw_Data-SHO_Fit_000", "NN_predictions", data)

    @property
    def nn_predictions_params_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._nn_predictions_params_scaled

    @nn_predictions_params_scaled.setter
    def nn_predictions_params_scaled(self, data):
        with h5py.File(self.dataset, "r+") as h5_f:
            self._nn_predictions_params_scaled = self.data_writer(
                "Raw_Data-SHO_Fit_000", "NN_predictions_params_scaled", data)

    @property
    def nn_predictions_params(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._nn_predictions_params

    @nn_predictions_params.setter
    def nn_predictions_params(self, data):
        with h5py.File(self.dataset, "r+") as h5_f:
            self._nn_predictions_params = self.data_writer(
                "Raw_Data-SHO_Fit_000", "NN_predictions_params", data)

    @property
    def nn_validation(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._nn_validation

    @nn_validation.setter
    def nn_validation(self, data):
        with h5py.File(self.dataset, "r+") as h5_f:
            self._nn_validation = self.data_writer(
                "Raw_Data-SHO_Fit_000", "NN_validation", data)

    @property
    def nn_validation_params_scaled(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._nn_validation_params_scaled

    @nn_validation_params_scaled.setter
    def nn_validation_params_scaled(self, data):
        with h5py.File(self.dataset, "r+") as h5_f:
            self._nn_validation_params_scaled = self.data_writer(
                "Raw_Data-SHO_Fit_000", "NN_validation_params_scaled", data)

    @property
    def nn_validation_params(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._nn_validation_params

    @nn_validation_params.setter
    def nn_validation_params(self, data):
        with h5py.File(self.dataset, "r+") as h5_f:
            self._nn_validation_params = self.data_writer(
                "Raw_Data-SHO_Fit_000", "NN_validation_params", data)

    def data_writer(self, base, name, data):
        with h5py.File(self.dataset, "r+") as h5_f:
            try:
                make_dataset(h5_f[base],
                             name,
                             data)
            except:
                self.delete(f"{base}/{name}")
                make_dataset(h5_f[base],
                             name,
                             data)

            return h5_f[f"{base}/{name}"][:]

    @property
    def nn_state_params(self):
        return self._nn_state_params

    @nn_state_params.setter
    def nn_state_params(self, value):
        self._nn_state_params = eval(
            f"self.nn_{value}_params").reshape(-1, self.voltage_steps, 4)

    @property
    def nn_scaled_state_params(self):
        return self._nn_scaled_state_params

    @nn_scaled_state_params.setter
    def nn_scaled_state_params(self, value):
        self._nn_scaled_state_params = value

    @property
    def SHO_nn_on(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._nn_state_params[:, 1::2, :]

    @property
    def SHO_nn_off(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self._nn_state_params[:, ::2, :]

    @property
    def SHO_nn_amp(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_nn_state()[:, :, 0]

    @property
    def SHO_nn_resonance(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_nn_state()[:, :, 1]

    @property
    def SHO_nn_q(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_nn_state()[:, :, 2]

    @property
    def SHO_nn_phase(self):
        with h5py.File(self.dataset, "r+") as h5_f:
            return self.SHO_nn_state()[:, :, 3]

    def SHO_nn_state(self):
        if self.state == "on":
            data_ = self.SHO_nn_on
        elif self.state == "off":
            data_ = self.SHO_nn_off
        else:
            data_ = self._nn_state_params

        if self.nn_scaled_state_params:
            data_ = self.SHO_scaler.transform(data_.reshape(-1, 4))

        return data_.reshape(-1, self.voltage_steps, 4)