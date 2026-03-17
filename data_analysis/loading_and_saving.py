import h5py
import numpy as np
import gc
from pathlib import Path
from typing import Generator
import os
from tqdm import tqdm
import spikeinterface as si 
import matplotlib.pyplot as plt
import spikeinterface.widgets as sw
import probeinterface as pi

# general info
# The mea electrodes are arranged in a 64x64 grid.
# Electrode naming convention
# 3brain standard is:
# - top left corner is (0,0)
# - bottom right corner is (63,63)
# We count the first (top left) electrode as 0 and the last (bottom right) electrode as 4095.


SAMPLING_RATE = 19753.775390625

def read_offset_and_conversion_factor_brw(brw_file_path: Path) -> tuple[float, float]:
    """
    Reads the offset value and conversion factor from a BRW file.f
    
    Parameters:
    brw_file_path (str): Path to the BRW file.
    
    Returns:
    tuple: A tuple containing the OffsetValue and ConversionFactor.
    """
    # Open the .brw file using h5py
    with h5py.File(brw_file_path, 'r') as file:
        # Read the required attributes from the file
        max_analog_value = file.attrs['MaxAnalogValue']
        min_analog_value = file.attrs['MinAnalogValue']
        max_digital_value = file.attrs['MaxDigitalValue']
        min_digital_value = file.attrs['MinDigitalValue']
        
        # Calculate the ConversionFactor and OffsetValue
        conversion_factor = (max_analog_value - min_analog_value) / (max_digital_value - min_digital_value)
        offset_value = min_analog_value - conversion_factor * min_digital_value
    file.close()
    return offset_value, conversion_factor

def apply_offset_and_conversion_factor(data: np.ndarray, 
                                       offset_value: float, 
                                       conversion_factor: float) -> np.ndarray:
    assert data.dtype == np.uint16, "Data type mismatch: expected int16. This function only works with int16 MEA data."
    assert len(data.shape) == 2, "Data must be a 2D array (num_frames x n_channels)."
    conversion_factor = np.float32(conversion_factor)
    offset_value = np.float32(offset_value)
    data *= conversion_factor
    data += offset_value
    print(f"Conversion factor applied: {conversion_factor}")
    # Round the data in-place
    np.rint(data, out=data)
    # Clip the data to int16 range in-place
    np.clip(data, np.iinfo(np.int16).min, np.iinfo(np.int16).max, out=data)
    # Convert to int16 without making a copy if possible
    return data.astype(np.int16, copy=False)
    

def extract_electrode_data(filepath : Path, 
                           start_minute: float, 
                           end_minute: float, 
                           apply_conversion : bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extracts electrode data from an HDF5 file for a specified time range.
    This function reads voltage data from a Multi-Electrode Array (MEA) recording stored in HDF5 format.
    It can optionally apply voltage conversion factors to convert raw integer values to actual voltage measurements.
    Parameters
    ----------
    filepath : Path
        Path to the HDF5 file containing the MEA recording data
    start_minute : float
        Start time in minutes from the beginning of the recording
    end_minute : float
        End time in minutes from the beginning of the recording
    apply_conversion : bool, optional
        If True, applies voltage conversion factors to convert raw values to actual voltages (default: False)
    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        A tuple containing:
        - voltage_data: 2D numpy array (num_frames x n_channels) containing the voltage readings
        - ch_indices: 1D numpy array containing the channel indices
        - sampling_rate: float representing the sampling rate in Hz
    Notes
    -----
    The function expects the HDF5 file to have a specific structure with 'Well_A1/Raw', 
    'Well_A1/RawTOC', and 'Well_A1/StoredChIdxs' datasets. The raw data is expected to 
    be in int16 format.
    When apply_conversion is True, the function performs in-place operations to minimize 
    memory usage while converting the raw values to actual voltages.
    """
    # Open the HDF5 file
    with h5py.File(filepath, 'r') as file:
        # Fetch the sampling rate
        sampling_rate = file.attrs['SamplingRate']
        
        
        # Access the Raw dataset and TOC
        raw_data = file['Well_A1/Raw']
        raw_toc = np.array(file['Well_A1/RawTOC'])
        
        # Get the channel indices (corresponding channel numbers)
        ch_indices = np.array(file['Well_A1/StoredChIdxs'])  # Shape (4096,)
        
        # Calculate the total number of frames
        n_channels = len(ch_indices)
        total_samples = len(raw_data) // n_channels  # Total frames for one electrode
        total_duration_sec = total_samples / sampling_rate
        global total_duration_min
        total_duration_min = total_duration_sec / 60
    
        #print(f"Total recording duration: {total_duration_min:.2f} minutes")
        
        # Convert start and end time from minutes to seconds and then to frames
        start_time_sec = start_minute * 60
        end_time_sec = end_minute * 60
        start_frame = int(start_time_sec * sampling_rate)
        end_frame = int(end_time_sec * sampling_rate)
        
        # Ensure we don't request more frames than available
        start_frame = min(start_frame, total_samples)
        end_frame = min(end_frame, total_samples)

        # Determine the positions in the Raw dataset
        start_pos = start_frame * n_channels
        end_pos = end_frame * n_channels

        # Extract the raw data for the specified time range
        extracted_data = raw_data[start_pos:end_pos]
        assert extracted_data.dtype == np.uint16, f"Extract_electrode_data: Data type mismatch: expected int16 but got {extracted_data.dtype }. \
        This function only works with int16 MEA data."
        
        # Calculate the number of frames
        num_frames = len(extracted_data) // n_channels
        if apply_conversion:
            print("Running in-place conversion")
            offset_value, conversion_factor = read_offset_and_conversion_factor_brw(filepath)

            # Reshape the data into a 2D array (num_frames x n_channels)
            reshaped_data = extracted_data.reshape((num_frames, n_channels)).astype(np.float32)
            del extracted_data
            gc.collect()
            voltage_data = apply_offset_and_conversion_factor(reshaped_data, offset_value, conversion_factor)
            del reshaped_data
            gc.collect()
        else:
            voltage_data = extracted_data.reshape((num_frames, n_channels))
        return voltage_data, ch_indices, sampling_rate, total_duration_min

def extract_electrode_meta_data(brw_path: Path) -> tuple[float, float]:
        with h5py.File(brw_path, 'r') as file:
            # Fetch the sampling rate
            sampling_rate = file.attrs['SamplingRate']
            print(f"Sampling rate: {sampling_rate} Hz")
            
            
            # Access the Raw dataset and TOC
            raw_data = file['Well_A1/Raw']
            raw_toc = np.array(file['Well_A1/RawTOC'])
            
            # Get the channel indices (corresponding channel numbers)
            ch_indices = np.array(file['Well_A1/StoredChIdxs'])  # Shape (4096,)
            
            # Calculate the total number of frames
            n_channels = len(ch_indices)
            total_samples = len(raw_data) // n_channels  # Total frames for one electrode
            total_duration_sec = total_samples / sampling_rate
            total_duration_min = total_duration_sec / 60

            assert sampling_rate > 15000, f"extract_electrode_meta_data: Sampling rate is too low: {sampling_rate} Hz. This is not a valid MEA recording."
        return sampling_rate, total_duration_min

def load_data_chunks(brw_path: Path,
                     total_duration_min: float) -> Generator[np.ndarray, None, None]:
    """
    Load BRW data in consecutive 0.5minute chunks.
    Parameters:
        brw_path (Path):
            Path to the binary recorded wave (BRW) file.
        total_duration_min (float):
            Total recording duration in minutes. Rounded internally to avoid
            floatingpoint inaccuracies.
    Yields:
        numpy.ndarray:
            A 2D array of shape (num_frames, n_channels) containing int16 data
            for each 0.5minute segment.
    Raises:
        AssertionError:
            If the extracted data is not of dtype int16 or not a 2D array.
    Notes:
        - The duration is split into 0.5minute intervals by scaling the rounded
            total duration (scale_factor=2).
        - Each chunk is loaded via extract_electrode_data with the corresponding
            start and end times in minutes.
    """
    # Round the total duration to avoid small floating point inaccuracies (I hate it)
    total_duration_min = np.round(total_duration_min)
    
    # Define the scaling factor to convert 0.5-minute steps into integer steps
    scale_factor = 2  # Because 1 / 0.5 = 2

    # Calculate the total number of steps
    total_steps = int(total_duration_min * scale_factor)
    
    # Generate integer start and end steps
    start_steps = np.arange(0, total_steps)
    end_steps = np.arange(1, total_steps + 1)
    
    # Scale back to minutes
    start_minutes = start_steps / scale_factor
    end_minutes = end_steps / scale_factor

    for start_min, end_min in zip(start_minutes, end_minutes):

        data, ch_indices, sampling_rate, total_duration_min = extract_electrode_data(
            brw_path, start_minute=start_min, end_minute=end_min, apply_conversion=False
        )
        assert data.dtype == np.uint16, f"load_data_chunks: Data type mismatch: expected int16 but got {data.dtype}.\
        This function only works with int16 MEA data."
        assert len(data.shape) == 2, "load_data_chunks: Data must be a 2D array (num_frames x n_channels)."

        yield data

def write_to_binary(binary_path : Path, 
                    brw_path : Path, 
                    total_duration_min: float,
                    overwrite: bool) -> None:
    """
    This function writes the data from a .brw file to a binary file.
    It first checks if the binary file already exists and removes it if so.
    """
    print(f"write_to_binary: Binary path: {binary_path}")
    print(f"write_to_binary: BRW path: {brw_path}")

    if os.path.exists(binary_path) and overwrite:
        os.remove(binary_path)
        print("write_to_binary: There was another binary: removed it!")
    if os.path.exists(binary_path) and not overwrite:
        print("write_to_binary: There was another binary: not removed! This will be assumed to the correct one!")
        return

    #RAM contrains dont preload the next chunk of data while writing the current one.
    #This is a bit slower but it is more memory efficient.
    with open(binary_path, 'ab') as f:
        for i, chunk in tqdm(enumerate(load_data_chunks(brw_path, total_duration_min)), desc="Writing .brw to binary"):
            #if chunk.size == 0:
            #   print(f"Chunk size is 0. This should not happen!, This error happend on iteration {i}")
            #    print(f"We will skip this chunk!")
            #    continue
            assert chunk.size != 0, f"write_to_binary: Chunk size is 0. This should not happen! This error happened on iteration {i}."
            # This error typically happens when the total_duration_min is not set correctly (too large!).
            assert chunk.dtype == np.uint16, f"write_to_binary: Data type mismatch: expected int16 but got {chunk.dtype}.\
            This function only works with uint16 MEA data."
            chunk.tofile(f)




def define_electrode_layout() -> pi.Probe:
    # Define MEA Electrode Layout for a Single Well (A1)
    num_channels = 4096  # 64x64 electrode grid for a single well
    grid_size = (64, 64)  # Grid size for the 64x64 MEA layout in a well
    electrode_spacing = 60  # Updated to 60 micrometer spacing between electrodes

    # Generate x, y indices
    x_indices, y_indices = np.meshgrid(np.arange(grid_size[1]), np.arange(grid_size[0]))

    # Reverse the y_indices to have (0,0) at the upper-left corner !!!!!!!!!!!
    y_indices = (grid_size[0] - 1) - y_indices

    # Flatten the indices in row-major order
    x_positions = x_indices.flatten() * electrode_spacing
    y_positions = y_indices.flatten() * electrode_spacing

    # Create a numpy array for the electrode positions
    positions = np.column_stack((x_positions, y_positions))

    # Create a probe with 4096 electrodes (64x64 grid) for the MEA system
    probe = pi.Probe(ndim=2)  # 2D probe
    probe.set_contacts(positions=positions, shapes='square', shape_params={'width': 21})  # Updated electrode shape and size
    probe.create_auto_shape(probe_type='rect')  # Automatically generate the probe shape

    # Set device_channel_indices to match the linear indexing of the MEA channels
    probe.set_device_channel_indices(np.arange(num_channels))

    return probe

# Load the spike data from the .bxr file
def load_bxr_file(filepath):
    print("Warning this function has not been adapted to work with the new code yet")
    # Open the .bxr HDF5 file
    with h5py.File(filepath, 'r') as file:
        # Access common datasets if they exist
        spike_times = np.array(file['Well_A1/SpikeTimes'])
        spike_channels = np.array(file['Well_A1/SpikeChIdxs'])
        spike_forms = np.array(file['Well_A1/SpikeForms'])
        return {
            'spike_times': spike_times,
            'spike_channels': spike_channels,
            'spike_forms': spike_forms
        }

# Ensure the channel indices in the .bxr file are mapped correctly to the .brw file
def map_bxr_channels_to_brw(bxr_spike_channels, brw_channel_numbers):
    print("Warning this function has not been adapted to work with the new code yet")
    # Map each spike channel from the .bxr file to the corresponding .brw channel index
    channel_map = {ch: idx for idx, ch in enumerate(brw_channel_numbers)}
    
    # Remap the spike channels from the .bxr file
    remapped_spike_channels = np.array([channel_map[ch] for ch in bxr_spike_channels if ch in channel_map])
    return remapped_spike_channels

# Get spike times for a specific channel
def get_spikes_for_channel(spike_times, spike_channels, channel):
    print("Warning this function has not been adapted to work with the new code yet")
    idx_channel = np.where(spike_channels == channel)[0]
    spike_times_loc = spike_times[idx_channel].copy()
    return spike_times_loc

####################################################################
# load data from binary file
####################################################################
probe = define_electrode_layout()


def get_rec_obj(binary_path : Path, 
                sampling_rate: float = SAMPLING_RATE,
                gain : float = 3.907203907203907,
                offset : float = -8000.0,
                brw_path : Path =None) -> si.BaseRecording:
    """
    Load the binary file as a SpikeInterface recording object.
    The recording will usally return the data with the shape (num_frames, n_channels)?????????????.

    """
    num_channels = 4096
    file_size = os.path.getsize(binary_path)
    total_samples = file_size // (np.dtype(np.uint16).itemsize * num_channels)
    recording = si.read_binary(
        file_paths=binary_path,
        dtype=np.uint16,
        num_channels=num_channels,
        sampling_frequency=sampling_rate,
        time_axis=0)   

    assert recording.get_dtype() == np.uint16 , f"get_rec_obj: Data type mismatch: expected uint16 but got {recording.get_dtype()}"
    
    if gain and offset:
        recording.set_channel_gains(gain)
        recording.set_channel_offsets(offset)
    if (not (gain and offset)) and brw_path :
        print("reading offset and gain from brw...")
        offset, gain = read_offset_and_conversion_factor_brw(brw_path)
        recording.set_channel_gains(gain)
        recording.set_channel_offsets(offset)
        print(f"gain {gain}, offset {offset}")
    recording = recording.set_probe(define_electrode_layout())
    return recording

