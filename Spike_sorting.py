# this script requires an excel document with the following columns:
# animal_number	: unique number for each animal for internal tracking
# slice_number	: Slice number of the recording, each recording day started with 1
# analyzed_bool	: Indicates if the slice has already been spike sorted
# original_recording_base_path	: Paths to the folder of the original recordings
# target_folder_base_path : Target paths where the kilosort output folder will be put
# start_time : Time when the spikesoting started
# end_time : Time when the spikesoting was finished
# kilosort_output_path	: Target paths where the kilosort output will be put
# bin_path : Target paths where the binary files will be put
# brw_path: Paths to the orginal mea recordings
# group: Tumor or Sham
# preincub : Disinhib.-Solution or something else

import torch

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import spikeinterface as si 
import probeinterface as pi
import spikeinterface.sorters as ss

from tqdm import tqdm

from pathlib import Path
from typing import Optional
import os 
import shutil
import datetime
import gc
import logging

from data_analysis.loading_and_saving import define_electrode_layout, write_to_binary, get_rec_obj, extract_electrode_meta_data

excel_path = Path(r"E:\meta_data_file.xlsx")
log_file = Path(r"E:\log_file.log")

#kilosort paramters
#read:
#https://kilosort.readthedocs.io/en/latest/parameters.html

#IT IS STRONGLY RECOMMENDED TO HAVE A GPU FOR KILOSORT SORTING!

#nt
# This is the number of time samples used to represent spike waveforms, 
# as well as the amount of symmetric padding for filtering. 
# The default represents 2ms + 1 bin for a sampling rate of 30kHz. 
# For a different sampling rate, you may want to adjust accordingly. 
# For example, nt = 81 would be the 2ms equivalent for 40kHz.

#We sample at ~20kHz, so we need to adjust nt to 81/2 ~ 41
nt = 41 


#min_template_size
# This sets the standard deviation of the smallest Gaussian spatial envelope 
# used to generate universal templates, with a default of 10 microns. 
# You may need to increase this for probes with wider spaces between contacts.

# we have lager spacing the the Neuropixel probes. Trial and error shows 20 to be a good value
# for our MEA
min_template_size = 20


#dmin  and dminx 
# These adjust the vertical and lateral spacing, respectively, of the universal templates 
# used during spike detection, as well as the vertical and lateral sizes of channel 
# neighborhoods used for clustering. By default, Kilosort will attempt to determine 
# a good value for dmin based on the median distance between contacts, 
# which tends to work well for Neuropixels-like probes. However, 
# if contacts are irregularly spaced, you may need to specify 
# this manually. The default for dminx is 32um, which is also 
# well suited to Neuropixels probes. For other probes, try setting 
# dminx to the median lateral distance between contacts as a starting point.

# our electrodes are spaced 60um apart, so we set dminx and dmin to 60
dmin = 60
dminx = 60


#batch_size
# This sets the number of samples included in each batch of data to be sorted, 
# with a default of 60000 corresponding to 2 seconds for a sampling rate of 30000. 
# For probes with fewer channels (say, 64 or less), increasing batch_size to include more
# data may improve results because it allows for better drift estimation (more spikes to
# estimate drift from).

# we set it to 10000 many because of memeory constraints
batch_size = 10000


#bad_channels
# The bad_channels will be calculated down the line based on manually inpainted areas
# where the slice and probe had no conact. The slice was often smaller than the MEA.
bad_channels = None


#do_correction
# This weill be set to False because we do not expect any drift in the data as the slice
# cannot move becuase of the anchoring to the MEA.
do_correction = False


#neart_chans and nearest_templates

#Here we keep the default values.
# nearest_chans (default usually 10) is well below your 4096 channels. 
# The documentation suggests decreasing nearest_templates for probes with both 
# sparse spacing and few channels (<~64) to avoid instability.
# While our spacing (60µm) is borderline sparse, our channel count is very high. 
# The high channel count should provide enough constraints to keep the template 
# assignment stable, so reducing nearest_templates is likely unnecessary and 
# could potentially limit the sorting quality.


# x_centers
# The number of x-positions to use when determining centers for template groupings. 
# Specifically, this is the number of centroids to look for when using k-means 
# to cluster the x-positions for the probe. In most cases you should not need 
# to specify this. However, for probes with contacts arranged in a 2D grid, 
# we recommend setting x_centers such that centers are placed every 200-300um 
# so that there are not too many templates in each group. For example, 
# for an array that is 2000um in width, try x_centers = 10. If contacts 
# are very densely spaced, you may need to use a higher value for better performance.

# The recommendation is centers every 200-300µm (See above).
# 3800 µm / 200 µm/center = 19 centers
# 3800 µm / 300 µm/center ≈ 12.7 centers
# so we set it to 15
x_centers = 15


#highpass_cutoff 

#We kept the default value of 300Hz.
highpass_cutoff = 300


#max_channel_distance
# Templates farther away than this from their nearest channel will not be used.
# Also limits distance between compared channels during clustering.


# Set max_channel_distance to 85µm to explicitly include direct 
# diagonal neighbors within the spatial constraints for template 
# validation and clustering comparisons. Given the 60µm electrode pitch, 
# orthogonal neighbors are at 60µm distance while diagonal neighbors 
# are at sqrt(60^2 + 60^2) approx 84.9µm; choosing an 85µm threshold
# ensures these immediately adjacent diagonal channels fall within the 
# radius considered by the algorithm for local operations, accounting for 
# potential signal spread and improving the robustness of spatial comparisons.
max_channel_distance = 85


#faulty_channels
# we found a faulty channel in the MEA.
faulty_channels = np.array([2099])


channels = np.arange(0,4096,1)
#remove later
#from mea_analysis import chans_to_coords, coords_to_chans
#x, y = chans_to_coords(channels=channels)
#mask = x < 32
#x = x[~mask]
#y = y[~mask]
#chans = coords_to_chans(x_coords=x, y_coords=y)
#faulty_channels = np.concatenate((faulty_channels, chans))
#faulty_channels = np.unique(faulty_channels).flatten()

def run_kilosort(recording: si.BaseRecording, 
                 kilosort_output_path : Path, 
                 probe : pi.Probe, 
                 bad_img_mask: np.ndarray,
                 faulty_channels: np.ndarray) -> None:
    
    recording = recording.set_probe(probe)
    print("printing recording details")
    print(recording)
    channels = np.arange(0,4096,1).reshape(64,64)
    channels_drop = channels[bad_img_mask==1] #the slice is painted black. Black are 0. So we only drop channels that are 1 (white = no slice)
    channels_drop = channels_drop.flatten()
    channels_drop = np.concatenate((channels_drop, faulty_channels))
    channels_drop = np.unique(channels_drop)
    channels_drop = channels_drop.flatten()
    print(f"There are {4096 - len(channels_drop)} channels left after dropping the bad channels")

    other_params = ss.get_default_sorter_params("kilosort4")
    other_params["nt"] = nt
    other_params["min_template_size"] = min_template_size
    other_params["dmin"] = dmin
    other_params["dminx"] = dminx
    other_params["batch_size"] = batch_size
    other_params["bad_channels"] = channels_drop
    other_params["do_correction"] = do_correction
    other_params["x_centers"] = x_centers
    other_params["highpass_cutoff"] = highpass_cutoff
    other_params["max_channel_distance"] = max_channel_distance

    sorting_ks4 = ss.run_sorter(sorter_name="kilosort4", 
                           recording=recording,
                           folder=kilosort_output_path,
                           **other_params)
    
    
def find_biggest_brw_in_folder(folder_path: Path) -> Optional[Path]:
    """
    Find the biggest .brw file in a folder. Returns the full path to the file.
    """
    biggest_file = None
    biggest_size = -1
    for file in folder_path.glob("*.brw"):
        if file.stat().st_size > biggest_size:
            biggest_file = file
            biggest_size = file.stat().st_size
    return biggest_file


def get_images_of_slices(org_img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the images of the slices and the mask.
    Args:
        org_img_path (path): Path to the original image.
        mask_path (path): Path to the mask image.
        Returns:
        mask (np.ndarray): Mask image.
        org_img (np.ndarray): Original image.
    """

    assert os.path.exists(mask_path), print(mask_path)
    assert os.path.exists(org_img_path), print(org_img_path)

    org_img = cv2.imread(org_img_path)
    #org_img = cv2.cvtColor(org_img)
    org_mask = cv2.imread(mask_path)

    # Resize the mask and process the image for the third plot
    resized_mask = cv2.resize(org_mask, (64, 64))  # 64x64 pixels = 1mm x 1mm
    rounded_img = np.sum(resized_mask, axis=2)
    rounded_img[rounded_img > 0] = 1
    rounded_img[rounded_img <= 0] = 0

    return rounded_img, org_img, org_mask

if __name__ == "__main__":
    df_metadata = pd.read_excel(excel_path)
    df_metadata = df_metadata[df_metadata["preincub"] == "Disinhib.-Solution"].reset_index(drop=True)
    df_metadata["analyzed_bool"] = df_metadata["analyzed_bool"].astype(int).astype(bool)
    # main sorting loop
    for i, row in tqdm(df_metadata.iterrows(), desc="Main Loop", total=len(df_metadata)):

        print("-"*30)

        #Each iteration has the followung structure:
        #1. Get the metadata for the current row
        #2. load the brw file and save it to a binary file chunk by chunk
        #3. read the binary file as a spikeinterface recording object
        #4. run kilosort on the recording object
        #5. update the metadata file

        iteration_start_time = datetime.datetime.now()


        # Get the metadata for the current row
        df_metadata = pd.read_excel(excel_path)
        df_metadata = df_metadata[df_metadata["preincub"] == "Disinhib.-Solution"].reset_index(drop=True)

        row["slice_identifier"] = str(row["animal_number"]) + "_" + str(row["slice_number"])
        slice_identifier = str(row["slice_identifier"])
        print(f"STARTING WITH {slice_identifier}")
        
        analyzed_bool = bool(row["analyzed_bool"])
        original_recording_base_path = Path(row["original_recording_base_path"])
        target_folder_base_path = Path(row["target_folder_base_path"])

        #if the target folder does not exist, create it
        if not os.path.exists(target_folder_base_path):
            print(f"Creating target folder: {target_folder_base_path}")
            os.makedirs(target_folder_base_path)


        assert original_recording_base_path.is_dir(), f"{original_recording_base_path} is not a directory"
        assert target_folder_base_path.is_dir(), f"{target_folder_base_path} is not a directory"
        

        kilosort_output_path = os.path.join(target_folder_base_path, "kilosort_output")
        bin_path = os.path.join(target_folder_base_path, "bin.dat")

        # Sometimes some recordings fail and we had to resart the recording. 
        # The biggest .brw file is in our case always the one that worked. So we take the biggest one.
        # This is not a general solution, but it works for us.
        brw_path = find_biggest_brw_in_folder(original_recording_base_path)
        print(f"brw_path: {brw_path}, size (GB) {brw_path.stat().st_size / (1024 ** 3):.2f}")
        

        org_img_path = os.path.join(original_recording_base_path, "org_image.png")
        mask_path = os.path.join(original_recording_base_path, "mask.png")
        assert os.path.exists(org_img_path), f"{org_img_path} does not exist"
        assert os.path.exists(mask_path), f"{mask_path} does not exist"
        
        # skip if the .brw file is not found
        if brw_path is None:
            print(f"No .brw file found in {original_recording_base_path}!!!!!")
            continue

        #skipping if already analyzed
        if analyzed_bool == True or analyzed_bool == 1:
            print(f"Already analyzed {slice_identifier}!!!!!")
            continue

        # remove the kilosort output folder if it already exists
        if os.path.exists(kilosort_output_path) == True:
            print("Kilosort output already exists, removing it")
            shutil.rmtree(kilosort_output_path)
        

        probe = define_electrode_layout()
        bad_channel_mask, org_img, org_mask = get_images_of_slices(org_img_path, mask_path)

        # save images to the target folder
        target_org_img_path = os.path.join(target_folder_base_path, "org_image.png")
        target_mask_path = os.path.join(target_folder_base_path, "mask.png")
        cv2.imwrite(target_org_img_path, org_img)
        cv2.imwrite(target_mask_path, org_mask)

        sampling_rate, total_duration_min = extract_electrode_meta_data(brw_path)

        print(f"The sampling rate is {sampling_rate} Hz")
        # write the brw path to binary. This will be usefull not only for spike sorting but also for analysis later, because we can 
        # directly load it as a recording object in spikeinterface. (Newer versions of spikeinterface can read .brw files directly!)
        print(f"Writing to binary: {bin_path}")
        write_to_binary(binary_path=bin_path, 
                        brw_path=brw_path, 
                        total_duration_min=total_duration_min,
                        overwrite=False)
        
        recording = get_rec_obj(binary_path=bin_path, 
                                sampling_rate=sampling_rate)
        
        # depending on the file size this can take a while
        print(f"Running kilosort on {slice_identifier}")
        run_kilosort(recording=recording, 
                     kilosort_output_path=kilosort_output_path, 
                     probe=probe, 
                     bad_img_mask=bad_channel_mask,
                     faulty_channels=faulty_channels)


        iteration_end_time = datetime.datetime.now()
        # this acutally improves performance a bit for some reason
        gc.collect()
        torch.cuda.empty_cache()

        df_metadata.loc[i, "analyzed_bool"] = True
        df_metadata.loc[i, "start_time"] = iteration_start_time
        df_metadata.loc[i, "end_time"] = iteration_end_time
        df_metadata.loc[i, "kilosort_output_path"] = kilosort_output_path
        df_metadata.loc[i, "bin_path"] = bin_path
        df_metadata.loc[i, "brw_path"] = brw_path
        df_metadata.to_excel(excel_path, index=False)
        print(f"Finished analyzing {slice_identifier}")
    
    exit(0)

