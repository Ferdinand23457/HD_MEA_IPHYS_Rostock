#This prbly the most important file
#This files does the Burst detection, SLE detection, provides functions to load the data and create heatmaps,
#SLE origin estimation and SOAT detection
import torch
from typing import Optional
from typing import Tuple
import os
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata

from tqdm import tqdm
import cv2
import spikeinterface.full as si
from kilosort.run_kilosort import load_sorting, CCG


SAMPLING_RATE = 19753.775390625

####################################################################################################
#Burst detection
####################################################################################################


def burst_detection(st, 
                    max_dist_ms, 
                    min_spikes, 
                    min_duration_ms, 
                    sampling_rate=SAMPLING_RATE):
    """NOTE TO MYSELF: THIS FUNC GETS PASSED THE SPIKE ITERATIVELIVE ONE CLUSTER AT A TIME not all at once"""

    st = np.sort(st)

    #Convert thresholds from ms to samples
    max_dist = max_dist_ms * sampling_rate / 1000.0
    min_duration = min_duration_ms * sampling_rate / 1000.0

    isi = np.diff(st)
    is_close = isi <= max_dist

    starts = []
    ends = []
    run_start = None

    for i, val in enumerate(is_close):
 
        if val:
            #we only need to set a new start if there was no start until now
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                num_spikes = (i - run_start) + 1
                if num_spikes >= min_spikes:
                    start_time = st[run_start]
                    end_time = st[i]
                    if (end_time - start_time) >= min_duration:
                        starts.append(start_time)
                        ends.append(end_time)
            run_start = None

    #Edge case: if ended with True
    if run_start is not None:
        run_end = len(is_close)
        num_spikes = (run_end - run_start) + 1
        if num_spikes >= min_spikes:
            start_time = st[run_start]
            end_time = st[run_end]
            if (end_time - start_time) >= min_duration:
                starts.append(start_time)
                ends.append(end_time)

    return np.array(starts), np.array(ends)


def find_major_burst(
    starts_list,
    ends_list,
    clusters,
    sampling_rate,
    minimum_overlap_ms=50,
    minimum_participating_clusters=2
):
    assert len(starts_list) == len(ends_list) == len(clusters)
    min_overlap_samples = minimum_overlap_ms * sampling_rate / 1000.0
    events = []
    #wenn ein cluster keine bursts hat, dann wird es garnicht erst nicht berücksichtigt
    for cluster_id, start_arr, end_arr in zip(clusters, starts_list, ends_list):
        if len(start_arr) == 0 or len(end_arr) == 0:
            continue
        for s, e in zip(start_arr, end_arr):
            events.append((s, +1, cluster_id))
            events.append((e, -1, cluster_id))
    #wenn es keine events gibt, dann wird auch nichts zurückgegeben
    if not events:
        return np.array([]), np.array([]), []

    #
    events.sort(key=lambda x: (x[0], x[1])) #sortier nach zeit und dann nach event type, also erst alle starts dann alle ends

    active_clusters = set()
    major_burst_starts = []
    major_burst_ends = []
    major_burst_clusters = []

    in_burst = False
    current_burst_start = None
    current_burst_clusters = set()

    for (time, event_type, clust) in events:
        if event_type == +1: #wenn es sich um ein burst start handelt, dann füge das cluster zur active_clusters hinzu
            active_clusters.add(clust)
        else: #wenn es sich um ein burst end handelt, dann entferne das cluster aus active_clusters
            if clust in active_clusters:
                active_clusters.remove(clust) 

        if len(active_clusters) >= minimum_participating_clusters:
            if not in_burst:
                in_burst = True
                current_burst_start = time
                current_burst_clusters = set(active_clusters)
            else:
                current_burst_clusters |= active_clusters
        else:
            if in_burst:
                burst_duration = time - current_burst_start
                if burst_duration >= min_overlap_samples:
                    major_burst_starts.append(current_burst_start)
                    major_burst_ends.append(time)
                    major_burst_clusters.append(sorted(list(current_burst_clusters)))
                in_burst = False
                current_burst_start = None
                current_burst_clusters = set()

    if in_burst and events:
        final_time = events[-1][0]
        burst_duration = final_time - current_burst_start
        if burst_duration >= min_overlap_samples:
            major_burst_starts.append(current_burst_start)
            major_burst_ends.append(final_time)
            major_burst_clusters.append(sorted(list(current_burst_clusters)))

    return np.array(major_burst_starts), np.array(major_burst_ends), major_burst_clusters


def fuse_NBs(starts, ends, fused_under_sec, clusters=None, sampling_rate=SAMPLING_RATE):
    """
    Merges seizure-like events if the gap between them is below a threshold.

    If the `clusters` parameter is provided, it will also merge the corresponding
    cluster identifiers for the fused events.

    Parameters
   -------
    starts : list or np.ndarray
        Start indices of the events in samples.
    ends : list or np.ndarray
        End indices of the events in samples.
    fused_under_sec : float
        Time threshold in seconds under which two events should be fused.
    clusters : list of lists, optional
        Each sublist contains cluster identifiers for the corresponding event.
        If provided, must be the same length as `starts`. Defaults to None.
    sampling_rate : int, optional
        The sampling rate of the signal in Hz. Defaults to 256.

    Returns
   ----
    tuple
        A tuple containing:
        - starts_fused (np.ndarray): Fused start indices.
        - ends_fused (np.ndarray): Fused end indices.
        - clusters_fused (list or None): Fused clusters, or None if the
          `clusters` argument was not provided.
    """
    starts = np.array(starts, dtype=np.int64)
    ends = np.array(ends, dtype=np.int64)
    process_clusters = clusters is not None
    fused_under_idx = fused_under_sec * sampling_rate

    if starts.size == 0:
        return np.array([]), np.array([]), [] if process_clusters else None

    if process_clusters and len(clusters) != starts.size:
        raise ValueError("If provided, 'clusters' must have the same length as 'starts' and 'ends'.")

    # Structure data for sorting
    if process_clusters:
        combined = sorted(zip(starts, ends, clusters), key=lambda x: x[0])
        starts_sorted, ends_sorted, clusters_sorted = zip(*combined)
    else:
        combined = sorted(zip(starts, ends), key=lambda x: x[0])
        starts_sorted, ends_sorted = zip(*combined)

    # Initialize fused lists with the first event
    starts_fused = [starts_sorted[0]]
    ends_fused = [ends_sorted[0]]
    if process_clusters:
        clusters_fused_sets = [set(clusters_sorted[0])]

    # Iterate and fuse subsequent events
    for i in range(1, len(starts_sorted)):
        # Check if the gap to the last fused event is within the threshold
        if starts_sorted[i] - ends_fused[-1] <= fused_under_idx:
            # Merge: update the end time of the last fused event
            ends_fused[-1] = max(ends_fused[-1], ends_sorted[i])
            if process_clusters:
                # Update the cluster set with an in-place union
                clusters_fused_sets[-1].update(clusters_sorted[i])
        else:
            # No merge: append as a new event
            starts_fused.append(starts_sorted[i])
            ends_fused.append(ends_sorted[i])
            if process_clusters:
                clusters_fused_sets.append(set(clusters_sorted[i]))

    # Finalize outputs
    starts_out = np.array(starts_fused, dtype=np.int64)
    ends_out = np.array(ends_fused, dtype=np.int64)
    
    if process_clusters:
        clusters_out = [sorted(list(s)) for s in clusters_fused_sets]
        return starts_out, ends_out, clusters_out
    else:
        return starts_out, ends_out, None


def remove_short_NBs(starts, ends, clusters=None, remove_under_sec=0.05,sampling_rate=SAMPLING_RATE):
    print(f"Remove uznder sec {remove_under_sec}")
    remove_under_idx = remove_under_sec*sampling_rate
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    durs = ends-starts
    over_idx = np.where(durs>=remove_under_idx)[0]
    starts = starts[over_idx]
    ends = ends[over_idx]

    if clusters is not None:
        clusters = [clusters[i] for i in over_idx]
        return starts, ends, clusters 
    else:
        return starts, ends, None


def NB_detection(st : np.ndarray, clu : np.ndarray, sampling_rate :float, 
                  max_dist_ms:float, min_spikes : int, min_duration_ms :float,
                  minimum_overlap_ms : float, minimum_frac_part_clusters : float, fused_under_sec : float, 
                  remove_under_sec:float):
    
    """"
    Detect Seizure-Like Events (SLEs) across clustered spike trains.
    This function identifies bursts of spikes within each cluster, finds
    synchronous “major” bursts across clusters based on overlap criteria,
    fuses events separated by short intervals, and removes events that
    are too brief.
    Parameters
   -------
    st : array_like of float
        1D array of spike timestamps (in seconds or sample units) for all clusters.
    clu : array_like of int
        1D array of same length as `st`, assigning each timestamp to a cluster label.
    sampling_rate : float
        Sampling rate of the recording in Hz.
    max_dist_ms : float
        Maximum inter-spike interval (in milliseconds) to group spikes into a burst.
    min_spikes : int
        Minimum number of spikes required for a burst to be considered valid.
    min_duration_ms : float
        Minimum duration (in milliseconds) of a burst.
    minimum_overlap_ms : float
        Minimum temporal overlap (in milliseconds) between bursts in different clusters
        to count towards a major event.
    minimum_frac_part_clusters : float
        Fraction (0-1) of bursting clusters that must participate in an overlapping event
        for it to qualify as a major burst.
    fused_under_sec : float
        Maximum gap (in seconds) between consecutive major bursts to fuse them into one event.
    remove_under_sec : float
        Minimum duration (in seconds) for a fused event to be retained; shorter events are removed.
    Returns
   ----
    major_burst_starts : ndarray of float
        Start times of the detected major SLEs.
    major_burst_ends : ndarray of float
        End times of the detected major SLEs.
    major_burst_clusters : list of ndarray of int
        For each detected SLE, an array of cluster labels that participated.
    Notes
   --
    - Bursts are first detected per cluster using `burst_detection`.
    - Major events are determined by overlapping bursts across clusters
        (via `find_major_burst`).
    - Events closer than `fused_under_sec` are merged using `fuse_NBs`.
    - Very short events shorter than `remove_under_sec` are discarded using
        `remove_short_NBs`.
    """

    clusters = np.unique(clu)
    starts_list = []
    ends_list = []

    for cluster in tqdm(clusters):
        st_cluster = st[clu==cluster]
        starts, ends = burst_detection(st_cluster,
                                       max_dist_ms=max_dist_ms,
                                       min_spikes=min_spikes,
                                       min_duration_ms=min_duration_ms,
                                       sampling_rate=sampling_rate)
        starts_list.append(starts)
        ends_list.append(ends)
    print(f"NB_detection found a total number of {len(np.concatenate(starts_list))} burst in the provided data")
    n_bursting_clusters = sum([1 for starts_arr in starts_list if len(starts_arr)>0])

    major_burst_starts, major_burst_ends, major_burst_clusters = find_major_burst(
        starts_list,
        ends_list,
        clusters,
        sampling_rate,
        minimum_overlap_ms=minimum_overlap_ms,
        minimum_participating_clusters=int(
            np.ceil(minimum_frac_part_clusters * n_bursting_clusters)
        )
    )

    major_burst_starts, major_burst_ends, major_burst_clusters= fuse_NBs(major_burst_starts, 
                                                                                    major_burst_ends,
                                                                                    major_burst_clusters, 
                                                                                    fused_under_sec)
    major_burst_starts, major_burst_ends, major_burst_clusters=remove_short_NBs(major_burst_starts, 
                                                                                 major_burst_ends,
                                                                                 major_burst_clusters, 
                                                                                 remove_under_sec)
    return major_burst_starts, major_burst_ends, major_burst_clusters


####################################################################################################
#data importing
####################################################################################################


def get_stdf(parent_folder_path: Path,
             return_amplitudes: bool = False) -> tuple:
    """
    Correctly loads Kilosort data, recalculates cluster quality on a 10-minute 
    subset, and provides a valid comparison with the original labels.
    """
    sampling_rate = SAMPLING_RATE
    sorter_output_dir = Path(parent_folder_path) / "kilosort_output" / "sorter_output"

    st = np.load(sorter_output_dir / 'spike_times.npy')
    clu = np.load(sorter_output_dir / 'spike_clusters.npy')
    amplitudes = np.load(sorter_output_dir / 'amplitudes.npy')
    templates = np.load(sorter_output_dir / 'templates.npy')
    chan_map = np.load(sorter_output_dir / 'channel_map.npy')
    
    mua_labels = pd.read_csv(
        sorter_output_dir / "cluster_KSLabel.tsv", sep="\t"
    ).rename(columns={"KSLabel": "KSLabel_old"})

    ops, _, _, _, _, _, _ = load_sorting(sorter_output_dir)
    acg_threshold = 0.2
    
    start_minute_kilo = 0
    end_minute_kilo = 10
    start_idx = start_minute_kilo * 60 * sampling_rate
    end_idx = end_minute_kilo * 60 * sampling_rate
    
    time_mask_subset = (st.flatten() >= start_idx) & (st.flatten() < end_idx)
    st_subset = st[time_mask_subset]
    clu_subset = clu[time_mask_subset]

    is_ref_new, _ = CCG.refract(clu_subset, st_subset / ops['fs'],
                            acg_threshold=acg_threshold)

    mua_labels['KSLabel_new'] = mua_labels['cluster_id'].map(
        lambda cid: 'good' if is_ref_new[cid] else 'mua' if cid < len(is_ref_new) else None
    )

    mua_labels['KSLabel_new'] = mua_labels['KSLabel_new'].fillna(mua_labels['KSLabel_old'])

    agree = np.sum(mua_labels['KSLabel_new'] == mua_labels["KSLabel_old"])
    disagree = np.sum(mua_labels['KSLabel_new'] != mua_labels["KSLabel_old"])
    #print(f"Mua labels new {mua_labels['KSLabel_new']}")
    #print(f"Mua labels old {mua_labels['KSLabel_old']}")

    print("Comparison of Labels (Original vs. Changed)")
    print(f"KSLabel agreement: {agree}, disagreement: {disagree}")

    number_good_old = np.sum(mua_labels["KSLabel_old"] == "good")
    number_good_new = np.sum(mua_labels['KSLabel_new'] == "good")
    print(f"Number of good clusters (Original): {number_good_old}")
    print(f"Number of good clusters (Changed): {number_good_new}")
    print("-" * 55)

    st = st_subset
    clu = clu_subset
    amplitudes = amplitudes[time_mask_subset]
    st = st - start_idx  # Normalize spike times

    chan_best_per_cluster = chan_map[(templates**2).sum(axis=1).argmax(axis=-1)]

    st_df = pd.DataFrame({"clu": clu}).groupby("clu").size().reset_index(name='num_spikes')
    st_df = st_df.sort_values("num_spikes", ascending=False)
    
    cluster_to_new_label_map = dict(zip(mua_labels['cluster_id'], mua_labels['KSLabel_new']))
    st_df["chan_best"] = st_df["clu"].map(lambda cid: chan_best_per_cluster[cid] if cid < len(chan_best_per_cluster) else -1)
    st_df["mua_labels"] = st_df["clu"].map(cluster_to_new_label_map)
    st_df = st_df[st_df["chan_best"] != 0].dropna()

    st_df["chan_best"] = st_df["chan_best"].astype(int)

    mapping_dict = {cid: chan_best_per_cluster[cid] for cid in np.unique(clu) if cid < len(chan_best_per_cluster)}
    vectorized_map = np.vectorize(mapping_dict.get, otypes=[float])
    best_cha_st = vectorized_map(clu)
    
    if return_amplitudes:
        return st_df, st, clu, best_cha_st, vectorized_map, amplitudes
    else:
        return st_df, st.astype(int), clu.astype(int), best_cha_st.astype(int), vectorized_map


def get_recording(binary_path : Path, 
                  sampling_rate : float = SAMPLING_RATE) -> si.BaseRecording:
    print("This function only exists for backwards compatibility with the GUI. Use get_rec_obj from load_and_saving instead")
    recording = si.read_binary(
        file_paths=binary_path,
        dtype=np.int16,
        num_channels=4096,
        sampling_frequency=sampling_rate,
        time_axis=0
    )
    n = 64
    channel_locations = np.array([[n-1-x, y] for y in range(n) for x in range(n)])
    recording.set_channel_locations(channel_locations)
    return recording


def get_bursts(st, clu, unique_clusters, max_dist_ms, min_spikes,min_duration_ms,sampling_rate=SAMPLING_RATE):
    starts_list = []
    ends_list = []

    for cluster in tqdm(unique_clusters):
        st_cluster = st[clu==cluster]
        starts, ends = burst_detection(st_cluster,
                                       max_dist_ms=max_dist_ms,
                                       min_spikes=min_spikes,
                                       min_duration_ms=min_duration_ms,
                                       sampling_rate=sampling_rate)
        starts_list.append(starts)
        ends_list.append(ends)
    return starts_list, ends_list, unique_clusters


def get_burst_participation_info(mb_start,
                                 mb_end,
                                 burst_starts,
                                 burst_ends,
                                 clusters):
    
    """

    Note to myself: This func finds clusters by the same logic as the find_major_burst!!!!
    The ends and stars of the bursts do no work well for visualization, because they are not clamped to the major burst window!!!
    Use diffent funvctino for visualization

    For a given major burst window [mb_start, mb_end], determine which clusters participate
    and report the original burst timestamps at which they first and last contribute.
    
    A cluster is considered to participate if any of its bursts overlaps the window.
    Overlap is defined as:
        burst_end > mb_start  and  burst_start < mb_end
    
    Instead of clamping the overlap to [mb_start, mb_end], the original burst start and end
    timestamps are returned. Thus, if a burst started before mb_start or ended after mb_end,
    the original timestamps are preserved.
    
    Parameters
   -------
    mb_start : float
        Start time of the major burst event.
    mb_end : float
        End time of the major burst event.
    burst_starts : list of 1D arrays
        burst_starts[i] contains the start times of bursts for cluster clusters[i].
    burst_ends : list of 1D arrays
        burst_ends[i] contains the end times of bursts for cluster clusters[i].
        Must have the same length as burst_starts.
    clusters : list or array-like
        List of cluster identifiers corresponding to each entry in burst_starts/burst_ends.
    
    Returns
   ----
    participating_clusters : list
        List of cluster IDs that have at least one burst overlapping [mb_start, mb_end].
    original_first_participation : list
        For each participating cluster, the earliest burst start time (original timestamp)
        among those bursts that overlap the event.
    original_last_participation : list
        For each participating cluster, the latest burst end time (original timestamp)
        among those bursts that overlap the event.
    """
    participating_clusters = []
    original_first_participation = []
    original_last_participation = []

    if len(burst_starts) != len(burst_ends) or len(burst_starts) != len(clusters):
        raise ValueError("burst_starts, burst_ends, and clusters must have the same length.")

    for i, clust in enumerate(clusters):
        starts_i = burst_starts[i]
        ends_i = burst_ends[i]
        #For each burst in this cluster, check if it overlaps the event window.
        overlapping_burst_starts = []
        overlapping_burst_ends = []
        for s, e in zip(starts_i, ends_i):
            if e > mb_start and s < mb_end:
                overlapping_burst_starts.append(s)
                overlapping_burst_ends.append(e)
        if overlapping_burst_starts:
            participating_clusters.append(clust)
            #Note: We return the original timestamps, not clamped to [mb_start, mb_end].
            original_first_participation.append(min(overlapping_burst_starts))
            original_last_participation.append(max(overlapping_burst_ends))
    
    return participating_clusters, original_first_participation, original_last_participation


####################################################################################################
#overview
####################################################################################################


def get_overview(main_folder : Path) -> pd.DataFrame:
    """
    Create a DataFrame overview of folders in the main folder that contain SLE data.
    """
    folders = os.listdir(main_folder)
    folders = [f for f in folders if "_" in f]
    folders = [f for f in folders if f[0].isdigit()]
    df = pd.DataFrame({"folder": folders})
    df["folder_path"] = df["folder"].apply(lambda x: Path(main_folder) / Path(x))
    df["bin_path"] = df["folder_path"].apply(lambda x: Path(x) / "bin.dat")
    df["analyzed"] = df["bin_path"].apply(lambda x: Path(x).exists())
    df["sle_path"] = df["folder_path"].apply(lambda x: Path(x) / "sle.npy") 
    df["sles_saved"] = df["sle_path"].apply(lambda x: Path(x).exists())
    return df


def chans_to_coords(channels : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute x and y coordinates from a 1D array of channel indices.

    Parameters
   -------
    channels : numpy.ndarray, shape (n,)
        One-dimensional array of non-negative integer channel indices. Each index is interpreted
        within a 2D grid of width 64.

    Returns
   ----
    x_coords : numpy.ndarray, shape (n,)
        The column indices for each channel, computed as channels % 64.
    y_coords : numpy.ndarray, shape (n,)
        The row indices for each channel, computed as channels // 64.

    Raises
   ---
    AssertionError
        If `channels` is not a one-dimensional array.
    """
    assert channels.ndim == 1, f"channels should be a 1D array, but got {channels.ndim}D"
    x_coords = channels % 64
    y_coords = channels // 64
    return x_coords, y_coords


def coords_to_chans(x_coords : np.ndarray, y_coords : np.ndarray) -> np.ndarray:
    """
    Convert x and y coordinates to linear channel indices.
    Parameters
   -------
    x_coords : np.ndarray, shape (N,)
        1D array of x-coordinates.
    y_coords : np.ndarray, shape (N,)
        1D array of y-coordinates.
    Returns
   ----
    np.ndarray, shape (N,)
        1D array of channel indices computed as x_coords + 64 * y_coords.
    Raises
   ---
    AssertionError
        If x_coords or y_coords is not 1D, or if their shapes do not match.
    """
    
    assert x_coords.ndim == 1 and y_coords.ndim == 1, f"x_coords and y_coords should be 1D arrays, but got {x_coords.ndim}D and {y_coords.ndim}D"
    assert x_coords.shape == y_coords.shape, f"x_coords and y_coords should have the same shape, but got {x_coords.shape} and {y_coords.shape}"
    return x_coords + 64 * y_coords


####################################################
#           heatmap creation functions            #
####################################################


def get_std_bigger_pic(recording_f, 
                       start : int, 
                       end : int, 
                       dur_ms: float, 
                       sampling_rate : float =SAMPLING_RATE) -> np.ndarray:
    raw_traces = recording_f.get_traces(
        start_frame=start,
        end_frame=end
    )
    """
    Compute a tiled grid image of channel-wise standard deviations for segments of a recording.
    Parameters
   -------
    recording_f : object
        An object that implements get_traces(start_frame: int, end_frame: int) → ndarray
        returning raw data of shape (n_frames, n_channels).
    start : int
        Frame index at which to begin extracting traces.
    end : int
        Frame index at which to end extracting traces.
    dur_ms : float
        Target segment duration in milliseconds. The total recording interval
        [start, end) will be split into segments of approximately this length.
    sampling_rate : float, optional
        Sampling rate in Hz used to convert dur_ms to sample points.
        Defaults to SAMPLING_RATE.
    Returns
   ----
    big_pic : numpy.ndarray
        A 2D array of shape (grid_len * 64, grid_len * 64) where grid_len =
        ceil(sqrt(num_segments)). Each 64x64 tile corresponds to one segment:
        the per-channel standard deviation, clipped between the 2nd and 98th
        percentiles and with extreme values replaced by the tile mean.
        Rows composed entirely of NaNs (unused tiles) are removed before return.
    """
    dur_s = dur_ms/1000
    dur_points = dur_s*sampling_rate
    total_frames = raw_traces.shape[0]
    num_segments = total_frames/dur_points
    num_segments = np.ceil(num_segments).astype(int)
    grid_len = np.ceil(np.sqrt(num_segments)).astype(int)
    print(num_segments, grid_len)

    #Calculate segments
    loc_starts = np.linspace(0, total_frames, num_segments + 1, endpoint=True).astype(int)[:-1]
    loc_ends = np.linspace(0, total_frames, num_segments + 1, endpoint=True).astype(int)[1:]
    dt_s = (loc_starts[1]-loc_starts[0])/sampling_rate
    print(f"dt [s]: {dt_s}, dt [ms]: {dt_s*1000}")
    
    traces = []
    for loc_start, loc_end in zip(loc_starts, loc_ends):
        traces.append(raw_traces[loc_start: loc_end, :])

    big_pic = np.full((64*grid_len, 64*grid_len), np.nan)
    k = 0
    sums = []
    for i in range(grid_len):
        for j in range(grid_len):
            if k >= len(traces):  #Add check to prevent index out of range
                continue
            row_start = i * 64
            col_start = j * 64
            #std_array = np.sum(np.abs(traces[k].astype(np.float32)), axis=0)
            std_array = np.std(traces[k].astype(np.float32), axis=0)
            sums.append(np.std(std_array))#sums.append(np.sum(np.abs(std_array)))
            std_array[std_array>np.nanpercentile(std_array, 98)] = np.nanpercentile(std_array, 98)
            std_array[std_array<np.nanpercentile(std_array, 2)] = np.nanmean(std_array)
            std_array = std_array.reshape(64, 64)
            big_pic[row_start:row_start+64, col_start:col_start+64] = std_array
            k += 1
    assert big_pic.shape[0] == grid_len*64, f"big_pic shape {big_pic.shape[0]} != {grid_len*64}"
    assert big_pic.shape[1] == grid_len*64, f"big_pic shape {big_pic.shape[1]} != {grid_len*64}"
    #if one row is completely nan, remove it
    #big_pic = big_pic[~np.isnan(big_pic).all(axis=1)]
    return big_pic


def remove_time_outliers(coords: np.ndarray,
                         times: np.ndarray,
                         n_neighbors: int = 10,
                         sigma: float = 1.0,
                         threshold_factor: float = 2.0,
                         weighting: str = 'gaussian',
                         handle_isolated: str = 'remove',
                         verbose: bool = False):
    """
    Remove time outliers based on local neighborhood statistics.

    For each coordinate, this function finds the k-nearest neighbors and computes a
    weighted local mean and standard deviation of their times. Points are flagged as
    outliers if their time deviates from the local mean by more than a specified
    threshold.

    This function specifically handles "isolated" points, where all neighbors are so
    distant that their weights underflow to zero. The `handle_isolated` parameter
    controls whether these points are kept or removed.

    Parameters
   -------
    coords : np.ndarray
        Array of shape (N, 2) containing the 2D coordinates.
    times : np.ndarray
        Array of shape (N,) containing the time values.
    n_neighbors : int, optional
        Number of neighbors for local statistics, by default 10.
    sigma : float, optional
        Standard deviation for the Gaussian weighting kernel, by default 1.0.
    threshold_factor : float, optional
        Multiplier for the local standard deviation to set the outlier threshold, by default 2.0.
    weighting : str, optional
        Weighting scheme ('gaussian' or 'inverse'), by default 'gaussian'.
    handle_isolated : str, optional
        Strategy for handling isolated points ('keep' or 'remove'), by default 'remove'.
        An isolated point has a local weight sum of zero.
    verbose : bool, optional
        If True, prints warnings for edge cases, by default False.

    Returns
   ----
    tuple[np.ndarray, np.ndarray]
        - filtered_times: Array of times with outliers removed.
        - kept_indices: Indices of the points that were kept.
    """
    coords = np.asarray(coords)
    times = np.asarray(times)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`coords` must be a 2D array with shape (N, 2).")
    if coords.shape[0] != times.shape[0]:
        raise ValueError("`coords` and `times` must have the same number of elements.")
    if coords.shape[0] == 0:
        return np.array([]), np.array([])
    if sigma <= 0 or threshold_factor <= 0:
        raise ValueError("Parameters `sigma` and `threshold_factor` must be positive.")
    if handle_isolated not in ['keep', 'remove']:
        raise ValueError("`handle_isolated` must be 'keep' or 'remove'.")

    if coords.shape[0] <= n_neighbors:
        if verbose:
            warnings.warn(
                f"Number of points ({coords.shape[0]}) is <= `n_neighbors` ({n_neighbors}). "
                f"Reducing `n_neighbors` to {coords.shape[0] - 1}."
            )
        if coords.shape[0] <= 1:
            return times, np.arange(len(times))
        n_neighbors = coords.shape[0] - 1

    # Neighbor Search
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # Exclude the point itself from its list of neighbors
    neighbor_distances = distances[:, 1:]
    neighbor_indices = indices[:, 1:]
    neighbor_times = times[neighbor_indices]

    # Weight Calculation
    if weighting == 'gaussian':
        weights = np.exp(-(neighbor_distances**2) / (2 * sigma**2))
    elif weighting == 'inverse':
        weights = 1.0 / (neighbor_distances + np.finfo(float).eps)
    else:
        raise ValueError("Unknown weighting scheme. Choose 'gaussian' or 'inverse'.")

    # Handle Isolated Points and Compute Statistics
    weight_sums = np.sum(weights, axis=1)
    
    # Identify isolated points where all neighbor weights are numerically zero
    isolated_mask = weight_sums < np.finfo(float).eps
    valid_mask = ~isolated_mask

    # Initialize arrays for local statistics
    weighted_means = np.zeros_like(times, dtype=float)
    local_std = np.zeros_like(times, dtype=float)

    # Compute statistics only for points with a valid local neighborhood
    if np.any(valid_mask):
        valid_weight_sums = weight_sums[valid_mask]
        valid_weights = weights[valid_mask]
        valid_neighbor_times = neighbor_times[valid_mask]

        weighted_means[valid_mask] = np.sum(valid_weights * valid_neighbor_times, axis=1) / valid_weight_sums
        
        weighted_vars = np.sum(valid_weights * (valid_neighbor_times - weighted_means[valid_mask, None])**2, axis=1) / valid_weight_sums
        local_std[valid_mask] = np.sqrt(weighted_vars)

    # Outlier Identification
    # Initialize the mask of points to keep
    if handle_isolated == 'keep':
        good_mask = isolated_mask.copy()
        if verbose and np.any(isolated_mask):
            print(f"Kept {np.sum(isolated_mask)} isolated points.")
    else: # handle_isolated == 'remove'
        good_mask = np.zeros_like(times, dtype=bool)
        if verbose and np.any(isolated_mask):
            print(f"Removed {np.sum(isolated_mask)} isolated points.")
            
    # Apply the outlier test to the non-isolated points
    if np.any(valid_mask):
        diff = np.abs(times[valid_mask] - weighted_means[valid_mask])
        threshold = threshold_factor * local_std[valid_mask]
        # Points are good if they are within the threshold. Add a small tolerance for zero-std cases.
        good_mask[valid_mask] = diff <= (threshold + np.finfo(float).eps)

    #Filtering and Return
    kept_indices = np.where(good_mask)[0]

    if kept_indices.size == 0:
        if verbose:
            warnings.warn("Filtering removed all points. Returning original input as a fallback.")
        return times, np.arange(len(times))
        
    return times[kept_indices], kept_indices


def to_img_coord(img, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    img_x = (x / 64 * img.shape[1])
    img_y = (y / 64 * img.shape[0])
    img_x = img_x.astype(int)
    img_y = img_y.astype(int)
    return img_x, img_y


def largest_subset_within_distance(points: np.ndarray, 
                                   max_dist_unit: float, 
                                   n_init: int = 100, 
                                   random_state: int = None,
                                   verbose : bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the largest subset of 2D points such that the centroid (mean) of the subset 
    is at most max_dist_unit away from every point in the subset.
    
    Parameters
   -------
    points : np.ndarray
        An array of shape (n_points, 2) representing 2D points.
    max_dist_unit : float
        Maximum allowed distance from the centroid to any point in the subset.
    n_init : int, optional
        Number of random initial seeds to try (default is 100). If the total number of points 
        is less than n_init, then all points are used as seeds.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
   ----
    centroid : np.ndarray
        The centroid of the largest subset found (reshaped as a 2D array with one row).
    subset_points : np.ndarray
        The points in the largest subset.
    """
    #Convert to numpy array and validate basic properties.
    points = np.asarray(points)
    if points.size == 0:
        raise ValueError("Empty points array is not a valid input for largest_subset_within_distance!")
    
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input 'points' must be a 2D array with shape (n_points, 2).")
    
    if max_dist_unit <= 0:
        raise ValueError("Parameter `max_dist_unit` must be positive.")
    
    if n_init <= 0:
        raise ValueError("Parameter `n_init` must be a positive integer.")
    
    n_points = points.shape[0]
    if n_points <= 1:
        if verbose: warnings.warn(
            f"Only one point provided (shape: {points.shape}). Returning the input as both centroid and subset."
        )
        return points.reshape(-1, 2), points.reshape(-1, 2)
    
    rng = np.random.default_rng(random_state)
    
    #Use all points as seeds if there are not many points.
    if n_points <= n_init:
        seed_indices = np.arange(n_points)

    else:
        seed_indices = rng.choice(n_points, size=n_init, replace=False)
    
    best_subset_idx = np.array([], dtype=int)
    best_centroid = None
    
    #Iterate over candidate seeds.
    for seed_idx in seed_indices:
        #Start with a candidate: all points within max_dist_unit of the seed point.
        seed = points[seed_idx]
        candidate_idx = np.where(np.linalg.norm(points - seed, axis=1) <= max_dist_unit)[0]
        
        #Iterative refinement:
        prev_candidate_idx = None
        while prev_candidate_idx is None or not np.array_equal(candidate_idx, prev_candidate_idx):
            prev_candidate_idx = candidate_idx
            #Compute the centroid of the current candidate set.
            centroid = points[candidate_idx].mean(axis=0)
            #Update candidate: all points within max_dist_unit of the new centroid.
            candidate_idx = np.where(np.linalg.norm(points - centroid, axis=1) <= max_dist_unit)[0]
        
        #If this candidate has more points than the best so far, record it.
        if candidate_idx.size > best_subset_idx.size:
            best_subset_idx = candidate_idx
            best_centroid = centroid
    
    if best_centroid is None:
        if verbose: warnings.warn("No valid subset found. Returning the original points as centroid and subset.")
        return points.reshape(-1, 2), points
    
    return best_centroid.reshape(-1, 2), points[best_subset_idx]


def find_burst_order(st : np.ndarray, 
                     clusters :np.ndarray, 
                     particip_clusters : np.ndarray, 
                     SLE_start_sec : float, 
                     SLE_end_sec : float,
                     vectorized_map : np.vectorize) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the relative burst onset times for specified clusters within a given time window.
    Parameters
   -------
    st : array_like
        1D array of spike times (in sample indices).
    clusters : array_like
        1D array of cluster labels corresponding to each entry in `st`.
    particip_clusters : array_like
        Sequence of cluster labels to include in the analysis.
    SLE_start_sec : float
        Start time of the analysis window in seconds.
    SLE_end_sec : float
        End time of the analysis window in seconds.
    vectorized_map : callable
        A function that maps an array of cluster IDs to channel identifiers (e.g., electrode labels).
        Must accept a vectorized input and return an array of the same shape.
    Returns
   ----
    unique_chans : ndarray
        Array of channel identifiers corresponding to each participating cluster, sorted by burst order.
    firsts_of_clusters_sec : ndarray
        Array of the first spike times (in seconds) for each cluster, shifted so that the earliest
        burst onset is at time zero, sorted in ascending order.
    Notes
   --
    - Assumes a global constant `SAMPLING_RATE` is defined (in Hz).
    - Converts the time window from seconds to sample indices before filtering.
    - Filters spikes to those within [SLE_start_sec, SLE_end_sec] and belonging to `particip_clusters`.
    - Identifies the first spike time for each cluster and normalizes on the earliest onset.
    """
    SLE_start_idx = SLE_start_sec*SAMPLING_RATE
    SLE_end_idx = SLE_end_sec*SAMPLING_RATE
    st = st.copy()
    clusters = clusters.copy()

    mask = (st>=SLE_start_idx) & (st<=SLE_end_idx)
    st = st[mask]
    clusters = clusters[mask]

    clusters_mask = np.isin(clusters, particip_clusters)
    st = st[clusters_mask]
    clusters = clusters[clusters_mask]

    unqiue_clusters = np.unique(clusters)
    firsts_of_clusters = np.zeros(len(unqiue_clusters))
    for i, cluster in enumerate(unqiue_clusters):
        mask = clusters == cluster
        st_clust = st[mask]
        firsts_of_clusters[i] = st_clust[0]
    
    unique_chans = vectorized_map(unqiue_clusters)
    firsts_of_clusters_sec = firsts_of_clusters/SAMPLING_RATE
    firsts_of_clusters_sec = firsts_of_clusters_sec-np.min(firsts_of_clusters_sec)
    SLE_df = pd.DataFrame({"chans": unique_chans,
                       "starts":firsts_of_clusters_sec})
    SLE_df = SLE_df.sort_values("starts", ascending=True)
    SLE_df = SLE_df.drop_duplicates("chans")
    unique_chans = SLE_df["chans"].to_numpy()
    firsts_of_clusters_sec = SLE_df["starts"].to_numpy()
    return unique_chans, firsts_of_clusters_sec


def burst_particip_clusters(burst_json_path : Path, sles : np.ndarray, folder_path : Path):
    """Identify participating channels and clusters for seizure-like events based on burst detection.
    Parameters
   -------
    burst_json_path : str
        Path to a JSON file containing burst detection parameters:
        - "max_dist_ms": maximum inter-spike interval for burst grouping (in milliseconds)
        - "min_spikes": minimum number of spikes to qualify as a burst
        - "min_duration_ms": minimum duration for a burst (in milliseconds)
    sles : array-like of shape (2, n_sles)
        Seizure-like event (SLE) windows. The first row contains start times,
        the second row contains end times (in the same time units as the spike train).
    folder_path : str
        Directory path where the spike train data and cluster assignments can be loaded
    Returns
   ----
    SLE_chans : list of numpy.ndarray
        For each SLE, an array of unique channel IDs that participated in any burst
        overlapping that SLE window. Basically for each SLE the participating channels.
    SLE_clusters : list of array-like
        For each SLE, the list or array of cluster indices whose bursts overlapped
        the SLE window. Basically for each SLE the participating clusters."""

    assert sles.ndim == 2, f"burst_particip_clusters: sles must allways be a numpy array with the shape (x,n_sles), currently it is a {type(sles)} with the shape {sles.shape}"
    
    SLE_chans = []
    SLE_clusters = []
    with open(burst_json_path, 'r') as f:
        burst_params = json.load(f)
    st_df, st, clu, best_cha_st, vectorized_map =  get_stdf(folder_path)
    burst_starts, burst_ends, burst_clusters = get_bursts(st, 
                                                        clu, 
                                                        np.unique(clu), 
                                                        burst_params["max_dist_ms"], 
                                                        burst_params["min_spikes"],
                                                        burst_params["min_duration_ms"],
                                                        sampling_rate=SAMPLING_RATE)
    for mb_start, mb_end in zip(sles[0],sles[1]):
        participating_clusters, x, y= get_burst_participation_info(mb_start, 
                                                                mb_end,
                                                                burst_starts,
                                                                burst_ends,
                                                                burst_clusters)

        participating_channels = vectorized_map(participating_clusters)
        participating_channels = np.unique(participating_channels)
        SLE_chans.append(participating_channels)
        SLE_clusters.append(participating_clusters)
    return SLE_chans, SLE_clusters


def get_st_snippet(st : np.ndarray, best_cha_st : np.ndarray, clu : np.ndarray , start_idx : int, end_idx : int):
    """
    This function cuts st, best_cha_st and clu to be between an given
    start and end index
    """

    mask = ((st > start_idx) & (st < end_idx))
    st_snippet = st[mask]
    best_cha_st_snippet = best_cha_st[mask]
    return st_snippet, best_cha_st_snippet, clu[mask]


###########################################################
#Origin estimation and filtering
##########################################################


def filter_spikes_find_origin(sles : np.ndarray,
                              st : np.ndarray,
                              clu : np.ndarray,
                              cluster_to_channel_map : callable, 
                              burst_json_path : Path, 
                              folder_path : Path):
    """
    Estimate the spatial origins of seizure like events (SLEs) by filtering and analyzing spike clusters.
    This function reconstructs burst participation clusters from previously saved parameters,
    orders cluster specific spike times, removes temporal outliers based on spatial proximity,
    and then computes the SLE origin as the centroid of the largest spatial subset of earliest
    activated channels.
    Parameters
   -------
    sles : np.ndarray, shape (2, M)
        Array containing the sample indices of SLE onsets and offsets for M events.
    st : np.ndarray
        1D array of spike times (in sample indices).
    clu : np.ndarray
        1D array of cluster labels corresponding to each entry in `st`.
    cluster_to_channel_map : callable
        Vectorized function or mapping that takes a cluster label and returns the associated
        recording channel ID.
    burst_json_path : pathlib.Path
        Path to the JSON file storing parameters used for burst detection (to be reloaded here).
    folder_path : pathlib.Path
        Directory containing raw data files and auxiliary resources needed by 
        `burst_particip_clusters`.
    Returns
   ----
    estimated_SLE_origins : np.ndarray, shape (M, 2)
        X,Y coordinates of the estimated origin for each of the M SLEs.
    first_spikes : list of np.ndarray
        Raw first spike times (in seconds) for each participating cluster, before outlier removal.
    unique_chans_list : list of np.ndarray
        Channel IDs of clusters participating in each SLE, before any filtering.
    filtered_first_spikes : list of np.ndarray
        First spike times (in seconds) for each cluster after temporal outlier removal.
    filtered_chans : list of np.ndarray
        Channel IDs remaining after time based outlier filtering for each SLE.
    chan_subsets : list of np.ndarray
        Indices of the subset of channels used to compute the final SLE origin (largest spatial cluster).
    Notes
   --
    - Calls `burst_particip_clusters` to regenerate participating clusters from saved parameters.
    - Uses `find_burst_order` to extract and sort initial spike times per cluster.
    - Applies `remove_time_outliers` to discard spikes whose timing deviates spatially from neighbors.
    - Limits to the earliest 5% of channels, then finds the largest spatially contiguous subset
        using `largest_subset_within_distance` to estimate the event origin.
    """

    estimated_SLE_orgins = []
    first_spikes = []
    unique_chans_list = []
    filtered_first_spikes = []
    filtered_chans = []
    chan_subsets = []


    #finding the burst is a deterministic process. So instead of saving all the data about the burst, in the GUI 
    #we just saved the func inputs used to find the burst. So we can easily recreate it herer
    SLE_chans, SLE_clusters = burst_particip_clusters(burst_json_path = burst_json_path, 
                                                      sles = sles, 
                                                      folder_path=folder_path)

    for j, particip_clusters in enumerate(SLE_clusters):
        unique_chans, firsts_of_clusters_sec = find_burst_order(st=st, 
                                                                clusters=clu, 
                                                                particip_clusters=particip_clusters, 
                                                                SLE_start_sec = sles[0][j]/SAMPLING_RATE,
                                                                SLE_end_sec =sles[1][j]/SAMPLING_RATE,
                                                                vectorized_map=cluster_to_channel_map)
        unique_chans_list.append(unique_chans.astype(int))
        first_spikes.append(firsts_of_clusters_sec)

        #remove time outliers
        paired = sorted(zip(firsts_of_clusters_sec, unique_chans), key=lambda x: x[0])
        firsts_of_clusters_sec, unique_chans = zip(*paired)
        firsts_of_clusters_sec = np.array(firsts_of_clusters_sec)    
        unique_chans = np.array(unique_chans)
        unique_coords = np.array(chans_to_coords(unique_chans)).T
        firsts_of_clusters_sec , kept_indices = remove_time_outliers(unique_coords, 
                                                                        firsts_of_clusters_sec, 
                                                                        n_neighbors=10, 
                                                                        sigma=1.0, 
                                                                        threshold_factor=2.0, 
                                                                        weighting='gaussian')
        unique_chans = unique_chans[kept_indices]

        #estimate SLE orgin
        unique_chans_cut = unique_chans[:int(np.ceil(len(unique_chans)*0.05))]
        coords = chans_to_coords(unique_chans_cut)
        coords = np.array(coords).T
        estimated_SLE_orgin, largest_subset = largest_subset_within_distance(points=coords,
                                                                            max_dist_unit=16,
                                                                            n_init=100,
                                                                            random_state=42)
        estimated_SLE_orgins.append(estimated_SLE_orgin)
        filtered_first_spikes.append(firsts_of_clusters_sec)
        filtered_chans.append(unique_chans.astype(int))
        chan_subsets.append(largest_subset)

    estimated_SLE_orgins = np.array(estimated_SLE_orgins)
    estimated_SLE_orgins = estimated_SLE_orgins.reshape(-1,2)
    estimated_SLE_orgins = np.array(estimated_SLE_orgins)
    estimated_SLE_orgins = estimated_SLE_orgins.reshape(-1,2)


     
    return estimated_SLE_orgins, first_spikes, unique_chans_list, filtered_first_spikes, filtered_chans, chan_subsets


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


####################################################################################################
#autocorrelation
####################################################################################################


def calculate_autocorrelogram(spike_times_ms:np.ndarray, bin_size_ms:float, max_lag_ms:float):
    """" Creates an autocorrelogram from spike times."""
    spike_times_ms = np.sort(spike_times_ms)
    N =len(spike_times_ms)
    n_spikes = len(spike_times_ms)
    lags = []
    for i in range(n_spikes):
        for j in range(i+1, n_spikes):
            lag = spike_times_ms[j] - spike_times_ms[i]
            lags.append(lag)
            lags.append(-lag)
            if lag > max_lag_ms:
                break
    lags = np.array(lags)

    n_bins = 2*int(np.ceil(max_lag_ms/bin_size_ms))
    assert n_bins % 2 == 0
    lim = int(n_bins / 2) 
    bin_edges = np.linspace(-lim, lim, n_bins+1)
    counts, _ = np.histogram(lags,bins=bin_edges)
    counts = counts/N
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, bin_centers, bin_edges


####################################################################################################
#Calculate center of activity (CoA) for 2D space
####################################################################################################


def calculate_center_of_activity_2d(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Calculates the Center of Activity (CoA) for a 2D space at a specific time point.

    Args:
        a: A 1D NumPy array representing the activity levels of neural units.
        x: A 1D NumPy array representing the x-coordinates of the neural units.
        y: A 1D NumPy array representing the y-coordinates of the neural units.

    Returns:
        A tuple containing the (CoA_x, CoA_y) coordinates.
        Returns (np.nan, np.nan) if the sum of activities is zero.
    """
    assert a.ndim == 1, "Activity array 'a' must be 1-dimensional."
    assert x.ndim == 1, "Coordinate array 'x' must be 1-dimensional."
    assert y.ndim == 1, "Coordinate array 'y' must be 1-dimensional."
    assert a.shape == x.shape == y.shape, "Input arrays 'a', 'x', and 'y' must have the same shape."
    assert np.all(a >= 0), "Activity levels in 'a' must be non-negative."

    sum_activities = np.sum(a)

    if sum_activities == 0:
        return (np.nan, np.nan)

    coa_x = np.sum(a * x) / sum_activities
    coa_y = np.sum(a * y) / sum_activities

    assert np.isfinite(coa_x), "Calculated coa_x is not finite. Check inputs."
    assert np.isfinite(coa_y), "Calculated coa_y is not finite. Check inputs."

    return (coa_x, coa_y)


####################################################################################################
#propagation speed calculation
####################################################################################################


def velocity_field(x_coords : np.ndarray, y_coords : np.ndarray, times : np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Compute a 2D propagation speed field and direction vectors from discrete event times.

    This function interpolates a scattered set of (x, y, t) measurements onto a regular
    2D grid, computes the spatial gradient of the interpolated time-surface, and from that
    derives a local propagation speed (inverse of the gradient magnitude) and direction
    vectors (negative spatial gradient = direction of decreasing time). The interpolation
    uses scipy.interpolate.griddata with cubic interpolation and the grid is fixed at
    200 x 200 samples spanning [min(x)-0.1, max(x)+0.1] and [min(y)-0.1, max(y)+0.1].

    Parameters
    ----------
    x_coords : np.ndarray
        1D array of x coordinates of measured events. Must have the same length as y_coords
        and times. Units are arbitrary but consistent units are required (e.g., mm).
    y_coords : np.ndarray
        1D array of y coordinates of measured events. Same length and units as x_coords.
    times : np.ndarray
        1D array of event times corresponding to (x_coords, y_coords). Units should be time
        (e.g., ms). Same length as x_coords and y_coords.

    Returns
    -------
    speed : np.ndarray
        2D array of shape (200, 200) containing the local propagation speed at each grid
        point. Computed as 1 / sqrt((dT/dx)^2 + (dT/dy)^2). Grid points where the
        interpolated time-surface is NaN (e.g., outside the convex hull or where cubic
        interpolation fails) are set to NaN. If the gradient magnitude is exactly zero,
        the reciprocal is safely handled and will not raise a divide-by-zero error.
    result_dict : dict
        Dictionary containing the following keys (all arrays of shape (200, 200)):
          - "grid_X": 2D array of x coordinates for the returned grid.
          - "grid_Y": 2D array of y coordinates for the returned grid.
          - "time_surface": interpolated time values on the grid (NaN where undefined).
          - "dir_x": x-component of the direction vector = -dT/dx.
          - "dir_y": y-component of the direction vector = -dT/dy.

    Behavior and edge cases
    -----------------------
    - If fewer than 3 input points are provided, interpolation is not possible. The function
      returns a speed array and result_dict entries filled with NaNs (shape (200, 200)).
    - Interpolation method: cubic griddata. Points outside the convex hull of the input
      coordinates typically produce NaN in the interpolated time surface.
    - The returned grid is fixed to 200 x 200 samples. Grid bounds are expanded by 0.1
      units beyond the min/max of the provided coordinates to give a small margin.
    - The direction vectors (dir_x, dir_y) point in the direction of decreasing time
      (i.e., along propagation). Speed is computed from the magnitude of the spatial
      gradient of the time-surface; units will be (spatial units) / (time units) given
      consistent input units.

    Notes
    -----
    - The function does not modify input arrays.
    - If you need a different grid resolution or interpolation method, you should modify
      the function accordingly.
    """

    if len(x_coords) < 3 or len(y_coords) < 3:
        print("Not enough points to calculate velocity field. Returning NaNs.")
        return np.full((200, 200), np.nan), {
            "grid_X": np.full((200, 200), np.nan),
            "grid_Y": np.full((200, 200), np.nan),
            "time_surface": np.full((200, 200), np.nan),
            "dir_x": np.full((200, 200), np.nan),
            "dir_y": np.full((200, 200), np.nan)
        }


    x_electrodes, y_electrodes = chans_to_coords(np.arange(4096))
    x_electrodes = x_electrodes / 64 * 3.78 #in mm
    y_electrodes = y_electrodes / 64 * 3.78 #in mm
    #grid = np.meshgrid(x_coords, y_coords)
    time_surface = griddata((x_coords, y_coords), times, (x_electrodes, y_electrodes), method="cubic")

    grid_x_min, grid_x_max = np.min(x_coords) - 0.1, np.max(x_coords) + 0.1
    grid_y_min, grid_y_max = np.min(y_coords) - 0.1, np.max(y_coords) + 0.1

    grid_x, dx = np.linspace(grid_x_min, grid_x_max, 200, retstep=True)
    grid_y, dy = np.linspace(grid_y_min, grid_y_max, 200, retstep=True)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    time_surface = griddata((x_coords, y_coords), times, (grid_X, grid_Y), method="cubic")

    mask = np.isnan(time_surface)

    dT_dy, dT_dx = np.gradient(time_surface, dy, dx)
    grad_magnitude = np.sqrt(dT_dx**2 + dT_dy**2)

    speed = np.divide(1.0, grad_magnitude, where=grad_magnitude != 0)
    speed[mask] = np.nan 

    dir_x = -dT_dx
    dir_y = -dT_dy

    result_dict = {
        "grid_X": grid_X,
        "grid_Y": grid_Y,
        "time_surface": time_surface,
        "dir_x": dir_x,
        "dir_y": dir_y
    }
    return speed, result_dict


def interpolate_surface(x_coords: np.ndarray, 
                        y_coords: np.ndarray, 
                        values: np.ndarray, 
                        grid_size: int = 200, 
                        method : str="cubic") -> tuple[np.ndarray, dict]:
    
    grid_X = np.linspace(np.min(x_coords)-0.1, np.max(x_coords)+0.1, grid_size)
    grid_Y = np.linspace(np.min(y_coords)-0.1, np.max(y_coords)+0.1, grid_size)
    grid_X, grid_Y = np.meshgrid(grid_X, grid_Y)
    surface = griddata((x_coords, y_coords), values, (grid_X, grid_Y), method=method)
    result_dict = {
        "grid_X": grid_X,
        "grid_Y": grid_Y
    }
    return surface, result_dict