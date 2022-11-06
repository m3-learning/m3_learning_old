import pyUSID as usid
import h5py
from ..util.h5_util import print_tree
import time
import os
from BGlib import be as belib
import sidpy



def print_be_tree(path):
    # Opens the translated file
    h5_f = h5py.File(path, "r+")

    # Inspects the h5 file
    usid.hdf_utils.print_tree(h5_f)

    # prints the structure and content of the file
    print("Datasets and datagroups within the file:\n------------------------------------")
    print_tree(h5_f.file)

    print("\nThe main dataset:\n------------------------------------")
    print(h5_f)
    print("\nThe ancillary datasets:\n------------------------------------")
    print(h5_f.file["/Measurement_000/Channel_000/Position_Indices"])
    print(h5_f.file["/Measurement_000/Channel_000/Position_Values"])
    print(h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Indices"])
    print(h5_f.file["/Measurement_000/Channel_000/Spectroscopic_Values"])

    print("\nMetadata or attributes in a datagroup\n------------------------------------")
    for key in h5_f.file["/Measurement_000"].attrs:
        print("{} : {}".format(key, h5_f.file["/Measurement_000"].attrs[key]))
        
def SHO_Fitter(input_file_path, force = False, max_cores = -1, max_mem=1024*8):
    start_time_lsqf = time.time()

    (data_dir, filename) = os.path.split(input_file_path)

    if input_file_path.endswith(".h5"):
        # No translation here
        h5_path = input_file_path

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
            field_mode = usid.hdf_utils.get_attr(h5_meas_grp, "VS_measure_in_field_loops")
        except KeyError:
            print("field mode could not be found. Setting to default value")
            field_mode = "out-of-field"
        try:
            vs_cycle_frac = usid.hdf_utils.get_attr(h5_meas_grp, "VS_cycle_fraction")
        except KeyError:
            print("VS cycle fraction could not be found. Setting to default value")
            vs_cycle_frac = "full"

    sho_fit_points = 5  # The number of data points at each step to use when fitting
    sho_override = False  # Force recompute if True

    

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
    parms_dict = parms_dict = sidpy.hdf_utils.get_attributes(h5_main.parent.parent)

    print(f"LSQF method took {time.time() - start_time_lsqf} seconds to compute parameters")