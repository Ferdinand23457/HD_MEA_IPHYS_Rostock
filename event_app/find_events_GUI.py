import tkinter as tk
from tkinter import Menu
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import numpy as np
import cv2
import spikeinterface.full as si
from tqdm import tqdm
import h5py

import os
from pathlib import Path
import sys
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import data_analysis.mea_analysis as mea_analysis


global h5_file, binary_path
global st, st_chans, best_cha_st, vectorized_map
global recording_f, sampling_rate, recording


sampling_rate = 19753.775390625
parent_folder_path = Path("/media/ferdinand-forberger/Seagate Portable Drive/8562_3")

results_dir = os.path.join(parent_folder_path, Path("kilosort_output"))
binary_path = os.path.join(parent_folder_path, Path("bin.dat"))
org_img_path = os.path.join(parent_folder_path, Path("org_image.png"))
org_img = cv2.imread(org_img_path)
org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
bxr_path = os.path.join(parent_folder_path, Path("Slice.bxr"))
bursts_save_path = os.path.join(parent_folder_path, Path("clusters.npy"))
sle_save_path = os.path.join(parent_folder_path, Path("sle.npy"))
sle_json_path = os.path.join(parent_folder_path, Path("burst_params.json"))

def get_spike_train_bxr(bxr_path):
    h5_file = h5py.File(bxr_path, "r")
    st = np.array(h5_file["Well_A1"]["SpikeTimes"])
    st_chans = np.array(h5_file["Well_A1"]["SpikeChIdxs"])  
    st = st
    #create vectorized map function that maps st_chans to itself
    vectorized_map = np.vectorize(lambda x: x)

    channels_to_drop = 2099
    mask = st_chans != channels_to_drop
    st = st[mask]
    st_chans = st_chans[mask]
    return st, st_chans, st_chans, vectorized_map

def get_st_snippet(st, best_cha_st, clu, start_idx, end_idx):
    mask = ((st > start_idx) & (st < end_idx))
    st_snippet = st[mask]
    best_cha_st_snippet = best_cha_st[mask]
    return st_snippet, best_cha_st_snippet, clu[mask]

def get_recordings(recording_f, start_idx, end_idx, channels):
    traces = recording_f.get_traces(start_frame=start_idx, end_frame=end_idx, channel_ids=channels)
    return traces

def placeholer():
    print("Placeholder function")

def plot_traces(recording_f, start_idx, end_idx, channels, ax):
    traces = recording_f.get_traces(start_frame=start_idx, end_frame=end_idx, channel_ids=channels)
    print("Traces shape: ", traces.shape)
    num_traces = traces.shape[1]
    mean_ptp = np.mean(np.ptp(traces, axis=0))
    offsets = np.arange(0, mean_ptp*num_traces, mean_ptp)[::-1]
    traces_to_plot = traces + offsets
    x_axis = np.arange(0, traces_to_plot.shape[0]) / sampling_rate
    for k in range(num_traces):
        ax.plot(x_axis, traces_to_plot[:, k], color='black', linewidth=0.5)
    #print spikes on traces
    st_snippet, best_cha_st_snippet, clu_snippet = get_st_snippet(st, best_cha_st, clu, start_idx, end_idx)
    st_snippet = st_snippet - start_idx

    for i in range(len(channels)):
        
        mask = best_cha_st_snippet == channels[i]
        spikes = st_snippet[mask].astype(int)
        print(f"Channel {i}: {channels[i]}")
        print(f"Spikes at {spikes}")
        print(spikes/sampling_rate)
        ax.scatter(spikes/sampling_rate, traces_to_plot[spikes, i], s=5, zorder=10, color=f"C{i}") #das färb echt nicht wie intended, irgendwann noch fixen

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    
####################################################
#            burst and SLE detection functions     #
####################################################



def all_bursts_detection(st, clu, sampling_rate, max_dist_ms, min_spikes, min_duration_ms):
    clusters = np.unique(clu)
    # map cluster ids to best channels

    starts_list = []
    ends_list = []
    for cluster in tqdm(clusters):
        st_cluster = st[clu==cluster]
        starts, ends = mea_analysis.burst_detection(st_cluster,
                                       max_dist_ms=max_dist_ms,
                                       min_spikes=min_spikes,
                                       min_duration_ms=min_duration_ms,
                                       sampling_rate=sampling_rate)
        starts_list.append(starts)
        ends_list.append(ends)
    # best_cha_st for each cluster is just the best channel,
    # so we can store them in the same order as the cluster loop
    cluster_channel = vectorized_map(clusters)
    return starts_list, ends_list, cluster_channel, clusters

####################################################
#                  Main Window Class               #
####################################################

class Main_Window(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Main Window")

        # Use screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height}")

        # A separate boolean flag to show bursts
        self.show_bursts_flag = False
        
        self.showSLE_flag = False
        # Traces plot
        self.trace_channels = [2500, 3000]

        # Create menu bar
        self.create_menu_bar()
        self.create_widgets()

        # Raster plot navigation
        self.current_start = 0
        self.current_end = int(15 * sampling_rate)
        self.current_delta_t = int(15 * sampling_rate)

        # Prepare figure
        self.fig, self.ax = plt.subplots(2, 1, figsize=(screen_width/100, screen_height/200))
        plt.tight_layout()

        # Canvas frame
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.grid(row=0, column=0, rowspan=7, columnspan=5)

        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Toolbar frame
        self.toolbar_frame = tk.Frame(self)
        self.toolbar_frame.grid(row=7, column=0, columnspan=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        self.update_raster_plot()

        # For burst detection
        self.starts_list = []
        self.ends_list = []
        self.burst_channels = []
        self.unqiue_clusters = np.unique(clu)
        
        #SLE editing
       
        # Setup RectangleSelector for drawing events
        self.selector = RectangleSelector(
            self.ax[0],
            self.on_rectangle_selected,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.selector.set_active(False)  # Initi

        # User-added SLEs
        self.user_SLE_starts = np.array([], dtype=int)
        self.user_SLE_ends = np.array([], dtype=int)
        self.current_edit_mode = None  # 'add', 'remove'

    def create_menu_bar(self):
        menu_bar = Menu(self)
        self.config(menu=menu_bar)

        # Filters Menu
        filters_menu = Menu(menu_bar, tearoff=0)
        filters_menu.add_command(label="Set Filters", command=self.open_filter_window)
        filters_menu.add_command(label="Remove Filters", command=self.remove_filters)
        menu_bar.add_cascade(label="Filters", menu=filters_menu)

        heatmaps_menu = Menu(menu_bar, tearoff=0)
        heatmaps_menu.add_command(label="Heatmap - cluster based", command=placeholer)
        # open heatmap class from the following line
        heatmaps_menu.add_command(label="Heatmap - time progression", command=self.on_heatmap_time_progression)
        heatmaps_menu.add_command(label="Heatmap - standard", command=placeholer)
        menu_bar.add_cascade(label="Heatmaps", menu=heatmaps_menu)

        #spiek source menu
        spike_source_menu = Menu(menu_bar, tearoff=0)
        spike_source_menu.add_command(label="Load from bxr", command=self.on_bxr_spike_source)
        spike_source_menu.add_command(label="Load from kilosort", command=self.on_kilosort_spike_source)
        menu_bar.add_cascade(label="Spike Source", menu=spike_source_menu)

        #save menu
        save_menu = Menu(menu_bar, tearoff=0)
        save_menu.add_command(label="Save SLEs", command=self.on_save_sles)
        save_menu.add_command(label="Save Bursts", command=self.on_save_bursts)
        menu_bar.add_cascade(label="Save", menu=save_menu)

    def create_widgets(self):
        """Creates basic widgets"""
        self.label = tk.Label(self, text="Main Window")
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.button_next = tk.Button(self, text="Next", command=self.on_next)
        self.button_next.grid(row=8, column=0, padx=10, pady=5, sticky="w")

        self.button_previous = tk.Button(self, text="Previous", command=self.on_previous)
        self.button_previous.grid(row=9, column=0, padx=10, pady=5, sticky="w")

        self.label = tk.Label(self, text="Set delta t in seconds")
        self.label.grid(row=10, column=0, padx=10, pady=5, sticky="w")

        self.delta_t_entry = tk.Entry(self)
        self.delta_t_entry.grid(row=11, column=0, padx=10, pady=5, sticky="w")

        self.button_set_dt = tk.Button(self, text="Set dT", command=self.set_delta_t)
        self.button_set_dt.grid(row=12, column=0, padx=10, pady=5, sticky="w")

        # Update traces button
        self.button_update_traces = tk.Button(self, text="Update Traces", command=self.updata_traces_plot)
        self.button_update_traces.grid(row=8, column=1, padx=10, pady=5, sticky="w")

        self.label = tk.Label(self, text="Channels to plot as traces")
        self.label.grid(row=9, column=1, padx=10, pady=5, sticky="w")

        self.channels_entry = tk.Entry(self)
        self.channels_entry.grid(row=10, column=1, padx=10, pady=5, sticky="w")

        ##### Column 2: Burst detection #####
        self.button_detect_bursts = tk.Button(self, text="Detect Bursts", command=self.on_detect_bursts)
        self.button_detect_bursts.grid(row=8, column=2, padx=10, pady=5, sticky="w")

        # Show/Hide bursts buttons
        self.show_bursts_button = tk.Button(self, text="Show Bursts", command=self.on_show_bursts)
        self.show_bursts_button.grid(row=9, column=2, padx=10, pady=5, sticky="w")

        self.hide_bursts_button = tk.Button(self, text="Hide Bursts", command=self.on_hide_bursts)
        self.hide_bursts_button.grid(row=10, column=2, padx=10, pady=5, sticky="w")

        self.label_max_dist = tk.Label(self, text="Max distance between spikes in burst [ms]")
        self.label_max_dist.grid(row=11, column=2, padx=10, pady=5, sticky="w")

        self.max_dist_entry = tk.Entry(self)
        self.max_dist_entry.grid(row=12, column=2, padx=10, pady=5, sticky="w")
        self.max_dist_entry.insert(0, "75")

        self.label_min_spikes = tk.Label(self, text="Min spikes in burst")
        self.label_min_spikes.grid(row=13, column=2, padx=10, pady=5, sticky="w")

        self.min_spikes_entry = tk.Entry(self)
        self.min_spikes_entry.grid(row=14, column=2, padx=10, pady=5, sticky="w")
        self.min_spikes_entry.insert(0, "5")

        self.label_min_burst_dur = tk.Label(self, text="Min burst duration [ms]")
        self.label_min_burst_dur.grid(row=15, column=2, padx=10, pady=5, sticky="w")

        self.min_burst_dur_entry = tk.Entry(self)
        self.min_burst_dur_entry.grid(row=16, column=2, padx=10, pady=5, sticky="w")
        self.min_burst_dur_entry.insert(0, "7")


        ##### Column 3: SLE detection #####
        self.button_detect_sles = tk.Button(self, text="Detect SLEs", command=self.on_detect_sles)
        self.button_detect_sles.grid(row=8, column=3, padx=10, pady=5, sticky="w")

        self.show_sles_button = tk.Button(self, text="Show SLEs", command=self.on_show_sles)
        self.show_sles_button.grid(row=9, column=3, padx=10, pady=5, sticky="w")

        self.hide_sles_button = tk.Button(self, text="Hide SLEs", command=self.on_hide_sles)
        self.hide_sles_button.grid(row=10, column=3, padx=10, pady=5, sticky="w")

        self.label_min_burst_overlap = tk.Label(self, text="Min burst overlap for SLE [ms]")
        self.label_min_burst_overlap.grid(row=11, column=3, padx=10, pady=5, sticky="w")

        self.min_burst_overlap_entry = tk.Entry(self)
        self.min_burst_overlap_entry.grid(row=12, column=3, padx=10, pady=5, sticky="w")
        self.min_burst_overlap_entry.insert(0, "0.5")

        self.label_min_part_clusters = tk.Label(self, text="Min fraction [0,1] of clusters in burst")
        self.label_min_part_clusters.grid(row=13, column=3, padx=10, pady=5, sticky="w")

        self.min_part_clusters_entry = tk.Entry(self)
        self.min_part_clusters_entry.grid(row=14, column=3, padx=10, pady=5, sticky="w")
        self.min_part_clusters_entry.insert(0, "0.05")

        self.label_fuse_under_s = tk.Label(self, text="Fuse SLEs under [s]")
        self.label_fuse_under_s.grid(row=15, column=3, padx=10, pady=5, sticky="w")

        self.fuse_under_s_entry = tk.Entry(self)
        self.fuse_under_s_entry.grid(row=16, column=3, padx=10, pady=5, sticky="w")
        self.fuse_under_s_entry.insert(0, "0.1")

        self.label_remove_under_sec = tk.Label(self, text="Remove SLEs under [s]")
        self.label_remove_under_sec.grid(row=17, column=3, padx=10, pady=5, sticky="w")

        self.remove_under_sec_entry = tk.Entry(self)
        self.remove_under_sec_entry.grid(row=18, column=3, padx=10, pady=5, sticky="w")
        self.remove_under_sec_entry.insert(0, "0.05")

        ##### Column 4: Editing Event #####
        self.button_add_event = tk.Button(self, text="Add Event", command=self.on_add_event)
        self.button_add_event.grid(row=8, column=4, padx=10, pady=5, sticky="w")

        self.button_remove_event = tk.Button(self, text="Remove Event", command=self.on_remove_event)
        self.button_remove_event.grid(row=9, column=4, padx=10, pady=5, sticky="w")

    def on_add_event(self):
        self.current_edit_mode = 'add'
        self.selector.set_active(True)

    def on_remove_event(self):
        self.current_edit_mode = 'remove'
        self.selector.set_active(True)

    def on_rectangle_selected(self, eclick, erelease):
        if self.current_edit_mode not in ['add', 'remove']:
            return

        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        start_time = x1
        end_time = x2
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)

        if self.current_edit_mode == 'add':
            self.user_SLE_starts = np.append(self.user_SLE_starts, start_sample)
            self.user_SLE_ends = np.append(self.user_SLE_ends, end_sample)

        elif self.current_edit_mode == 'remove':
            # Combine all SLEs (detected + user)
            all_starts = np.concatenate([self.major_burst_starts, self.user_SLE_starts])
            all_ends = np.concatenate([self.major_burst_ends, self.user_SLE_ends])
            # Find events within the selected range
            mask = (all_starts >= start_sample) & (all_ends <= end_sample)
            # Split indices into detected and user
            num_detected = len(self.major_burst_starts)
            detected_mask = mask[:num_detected]
            user_mask = mask[num_detected:]
            # Update stored events
            self.major_burst_starts = self.major_burst_starts[~detected_mask]
            self.major_burst_ends = self.major_burst_ends[~detected_mask]
            self.user_SLE_starts = self.user_SLE_starts[~user_mask]
            self.user_SLE_ends = self.user_SLE_ends[~user_mask]

        self.current_edit_mode = None
        self.selector.set_active(False)
        self.update_raster_plot()

    def on_bxr_spike_source(self):
        print("Loading spikes from bxr")
        global st, clu, best_cha_st, vectorized_map
        st, clu, best_cha_st, vectorized_map = get_spike_train_bxr(bxr_path)
        self.update_raster_plot()

    def on_kilosort_spike_source(self):
        print("Loading spikes from kilosort")
        global st, clu, best_cha_st, vectorized_map
        _, st, clu, best_cha_st, vectorized_map = mea_analysis.get_stdf(parent_folder_path)
        self.update_raster_plot()

    def on_heatmap_time_progression(self):
        print("Heatmap time progression")
        self.heatmap_window = Heatmap_Window(org_img)
    def on_show_sles(self):
        print("Show SLEs")
        self.showSLE_flag = True
        self.update_raster_plot()
    def on_hide_sles(self):
        print("Hide SLEs")
        self.showSLE_flag = False
        self.update_raster_plot()

    def on_detect_sles(self):
        min_burst_overlap_ms = float(self.min_burst_overlap_entry.get())
        min_part_clusters = float(self.min_part_clusters_entry.get())
        max_dist_ms = float(self.max_dist_entry.get())
        min_spikes = int(self.min_spikes_entry.get())
        min_duration_ms = float(self.min_burst_dur_entry.get())
        fuse_under_s = float(self.fuse_under_s_entry.get())
        remove_under_sec = float(self.remove_under_sec_entry.get())
        self.major_burst_starts, self.major_burst_ends, self.major_burst_clusters = mea_analysis.NB_detection(
            st, clu, sampling_rate, max_dist_ms, min_spikes, min_duration_ms,
            min_burst_overlap_ms, min_part_clusters,
            fuse_under_s,
            remove_under_sec
        )
    def on_save_sles(self):
        print("Just the SLEs where saved")
        print("User SLEs:")
        print(self.user_SLE_starts)
        print("User SLEs:")
        print(self.user_SLE_ends)

        all_starts = np.concatenate([self.major_burst_starts, self.user_SLE_starts])
        all_ends = np.concatenate([self.major_burst_ends, self.user_SLE_ends])
        all_starts = np.sort(all_starts)
        all_ends = np.sort(all_ends)
        np.save(sle_save_path, np.array([all_starts, all_ends]))
        #also safe burst parameters to json
        burst_params = {"max_dist_ms": float(self.max_dist_entry.get()), 
                        "min_spikes": int(self.min_spikes_entry.get()), 
                        "min_duration_ms": float(self.min_burst_dur_entry.get())}
        
        with open(sle_json_path, 'w') as f:
            json.dump(burst_params, f)
        print(f"SLEs saved to {sle_save_path}")

    def on_save_bursts(self):
        assert False, "Implement get_burst_participation_info"
        print("ADDDDD get_burst_participation_info")
        print(f"starts_lists len {len(self.starts_list)}")
        print(f"ends_lists len {len(self.ends_list)}")
        print(f"burst_channels len {len(self.burst_channels)}")
        np.save(bursts_save_path, np.array([self.starts_list, self.ends_list, self.burst_channels]))
        print(f"Bursts saved to {bursts_save_path}")

    def on_show_bursts(self):
        print("Show bursts")
        self.show_bursts_flag = True
        self.update_raster_plot()

    def on_hide_bursts(self):
        print("Hide bursts")
        self.show_bursts_flag = False
        self.update_raster_plot()

    def on_detect_bursts(self):
        max_dist_ms = float(self.max_dist_entry.get())
        min_spikes = int(self.min_spikes_entry.get())
        min_duration_ms = float(self.min_burst_dur_entry.get())
        self.starts_list, self.ends_list, self.burst_channels, self.burst_clusters = all_bursts_detection(
            st, clu, sampling_rate, max_dist_ms, min_spikes, min_duration_ms
        )
        print("Bursts detected")

    def update_raster_plot(self):
        start_idx = self.current_start
        end_idx = self.current_end
        st_snippet, best_cha_st_snippet, clu_snippet = get_st_snippet(st, best_cha_st,clu, start_idx, end_idx)

        self.y_axis_clusters = False
        # Clear top axis for the raster
        if self.y_axis_clusters:
            self.ax[0].clear()
            self.ax[0].grid(True)
            self.ax[0].scatter(
                st_snippet / sampling_rate, 
                clu_snippet, 
                s=1, 
                c='black',
                zorder=1
            )
            test_arr = st_snippet[clu_snippet==222]/sampling_rate
            
            self.ax[0].set_xlabel("Time (s)")
            self.ax[0].set_ylabel("Cluster")
            # Color the spikes in bursts red if show_bursts_flag is True
            if self.show_bursts_flag:
                for i in range(len(self.starts_list)):
                    starts_arr = self.starts_list[i]
                    ends_arr   = self.ends_list[i]
                    
                    cluster  = self.burst_clusters[i]
                    #print(f"update_raster_plot: Cluster: {cluster}")
                    
                    mask = (starts_arr > start_idx) & (ends_arr < end_idx)
                    starts_arr = starts_arr[mask]
                    ends_arr   = ends_arr[mask]
                    
                    #debugging
                    #if cluster == 222:
                        #print(f"Paramters of the burst detection max dist ms{self.max_dist_entry.get()}, minspikes {self.min_spikes_entry.get()}, min duration ms {self.min_burst_dur_entry.get()}")

                        #print(f"Cluster 222 starts as found by the function in ms: {starts_arr/sampling_rate*1000}")
                        #print(f"Cluster 222 values in ms: {test_arr*1000}")
                        #print(f"Diff arr in ms {np.diff(test_arr)*1000}")
                    for start, end in zip(starts_arr, ends_arr):
                        st_snippet_loc = st_snippet[(st_snippet > start) & (st_snippet < end)]
                        self.ax[0].plot(
                            st_snippet_loc / sampling_rate,
                            np.full_like(st_snippet_loc, cluster),
                            c='red', 
                            zorder=2,
                            alpha=1
                        )
        else:
            self.ax[0].clear()
            self.ax[0].grid(True)
            self.ax[0].scatter(
                st_snippet / sampling_rate, 
                best_cha_st_snippet, 
                s=1, 
                c='black',
                zorder=1
            )
            self.ax[0].set_xlabel("Time (s)")
            self.ax[0].set_ylabel("Channel")

            # Color the spikes in bursts red if show_bursts_flag is True
            if self.show_bursts_flag:
                for i in range(len(self.starts_list)):
                    starts_arr = self.starts_list[i]
                    ends_arr   = self.ends_list[i]
                    channel    = self.burst_channels[i]
                    #print(f"update_raster_plot: Channel: {channel}")
                    mask = (starts_arr > start_idx) & (ends_arr < end_idx)
                    starts_arr = starts_arr[mask]
                    ends_arr   = ends_arr[mask]

                    
                    for start, end in zip(starts_arr, ends_arr):
                        st_snippet_loc = st_snippet[(st_snippet > start) & (st_snippet < end)]
                        self.ax[0].plot(
                            st_snippet_loc / sampling_rate,
                            np.full_like(st_snippet_loc, channel),
                            c='red', 
                            zorder=2,
                            alpha=1
                        )
            #vertical lines for SLE starts and ends
            if self.showSLE_flag:
                all_starts = np.concatenate([self.major_burst_starts, self.user_SLE_starts])
                all_ends = np.concatenate([self.major_burst_ends, self.user_SLE_ends])
                local_SLE_starts = all_starts[(all_starts > start_idx) & (all_starts < end_idx)]
                local_SLE_ends = all_ends[(all_ends > start_idx) & (all_ends < end_idx)]
                for start in local_SLE_starts:
                    self.ax[0].axvline(x=start/sampling_rate, color='tab:blue', zorder=3)
                for end in local_SLE_ends:
                    self.ax[0].axvline(x=end/sampling_rate, color='tab:red', zorder=3)

        # Clear bottom axis for the traces
        #self.ax[1].clear()

        # Keep the existing trace plots up to date
        #plot_traces(recording_f, start_idx, end_idx, self.trace_channels, self.ax[1])

        self.canvas.draw()
        print("Updating raster plot")

    def updata_traces_plot(self):
        # Just re-plot the bottom axis traces
        start_idx = self.current_start
        end_idx   = self.current_end
        self.trace_channels = [int(x) for x in self.channels_entry.get().split(",") if x.strip()]
        
        self.ax[1].clear()
        plot_traces(recording_f, start_idx, end_idx, self.trace_channels, self.ax[1])
        self.canvas.draw()

    def on_next(self):
        print("Next")
        self.current_start += self.current_delta_t
        self.current_end   += self.current_delta_t
        self.update_raster_plot()

    def on_previous(self):
        print("Previous")
        self.current_start -= self.current_delta_t
        self.current_end   -= self.current_delta_t
        self.update_raster_plot()

    def set_delta_t(self):
        val = float(self.delta_t_entry.get())
        self.current_delta_t = int(val * sampling_rate)
        self.update_raster_plot()

    def open_new_window(self):
        self.new_window = New_Window()

    def open_filter_window(self):
        self.filter_window = Filter_Window()

    def remove_filters(self):
        global recording_f
        recording_f = mea_analysis.get_recording(binary_path)
        print("Filters removed!")

class New_Window(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("New Window")
        self.geometry("400x400")
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self, text="New Window")
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.button = tk.Button(self, text="Open New Window", command=self.open_new_window)
        self.button.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    def open_new_window(self):
        self.new_window = New_Window()

####################################################
#                  Filter Window Class             #
####################################################

class Filter_Window(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Filter Settings")
        self.geometry("300x200")

        self.label = tk.Label(self, text="Set your filters here:")
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.filter_entry = tk.Entry(self)
        self.filter_entry.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.apply_button = tk.Button(self, text="Apply", command=self.apply_filters)
        self.apply_button.grid(row=2, column=0, padx=10, pady=5, sticky="w")

    def apply_filters(self):
        global recording_f
        filter_value = self.filter_entry.get()
        filter_val_lower = int(filter_value.split(",")[0])
        if  "," in filter_value:
            filter_val_upper = int(filter_value.split(",")[1])
        else:
            filter_val_upper = ""
        print(f"Lower bound: {filter_val_lower}, Upper bound: {filter_val_upper}")
        #if entry empty
        if filter_value == "":
            recording_f = mea_analysis.get_recording(binary_path)
        #if only lower bound
        elif filter_val_upper == "":
            recording_f = mea_analysis.get_recording(binary_path)
            recording_f = si.filter(recording_f, band=300.0, btype='highpass', filter_order=5, ftype='butter', filter_mode='sos', margin_ms=5.0, add_reflect_padding=False, coeff=None, dtype=None, direction='forward-backward')
            print("applied highpass filter")
        #if both bounds
        else:
            recording_f = mea_analysis.get_recording(binary_path)
            recording_f = si.filter(recording_f, band=[filter_val_lower, filter_val_upper], btype='bandpass', filter_order=5, ftype='butter', filter_mode='sos', margin_ms=5.0, add_reflect_padding=False, coeff=None, dtype=None, direction='forward-backward')
            print("applied bandpass filter")
        self.destroy()

####################################################
#                  Heatmap Window Class            #
####################################################

class Heatmap_Window(tk.Toplevel):
    def __init__(self, org_img):
        super().__init__()
        self.title("Heatmap")
        self.geometry("900x800")

        self.im = None
        self.cbar = None

        self.label = tk.Label(self, text="Heatmap")
        self.label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.button = tk.Button(self, text="Open Heatmap", command=self.open_heatmap)
        self.button.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.label_start_s = tk.Label(self, text="Start time [s]")
        self.label_start_s.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.start_s_entry = tk.Entry(self)
        self.start_s_entry.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        self.label_end_s = tk.Label(self, text="End time [s]")
        self.label_end_s.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.end_s_entry = tk.Entry(self)
        self.end_s_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        self.label_dur_ms = tk.Label(self, text="Duration [ms]")
        self.label_dur_ms.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.dur_ms_entry = tk.Entry(self)
        self.dur_ms_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        self.label_max_cb_val = tk.Label(self, text="Max colorbar value")
        self.label_max_cb_val.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.max_cb_val_entry = tk.Entry(self)
        self.max_cb_val_entry.grid(row=5, column=1, padx=10, pady=5, sticky="w")

        self.button_max_cb_val = tk.Button(self, text="Set max colorbar value", command=self.set_max_cb_val)
        self.button_max_cb_val.grid(row=6, column=1, padx=10, pady=5, sticky="w")

        self.fig, self.ax = plt.subplots(
            1, 2, 
            figsize=(10, 5),       # Adjust as necessary for your screen
            constrained_layout=True
        )

        # Force both axes to be square
        self.ax0 = self.ax[0]
        self.ax0.set_aspect("equal")
        self.ax0.axis("off")

        self.ax1 = self.ax[1]
        self.ax1.set_aspect("equal")
        self.ax1.axis("off")
        self.ax1.imshow(org_img)

        # Embed Matplotlib figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        # Place the canvas a bit further down to avoid overlap
        self.canvas.get_tk_widget().grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Add Matplotlib's navigation toolbar
        self.toolbar_frame = tk.Frame(self)
        self.toolbar_frame.grid(row=8, column=0, columnspan=2, pady=5, sticky="w")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.canvas.draw()

    def open_heatmap(self):
        """Open (or update) a heatmap."""
        # For demo purposes, adapt to your real data:
        start_s = float(self.start_s_entry.get())
        end_s = float(self.end_s_entry.get())
        dur_ms = float(self.dur_ms_entry.get())

        # Replace with your actual heatmap function
        heatmap = mea_analysis.get_std_bigger_pic(recording_f, start_s * sampling_rate,
                                     end_s * sampling_rate, dur_ms)

        # If self.im does NOT exist yet, create the imshow and colorbar.
        if self.im is None:
            self.im = self.ax0.imshow(heatmap, cmap='inferno')
            self.cbar = self.fig.colorbar(self.im, ax=self.ax0)
        else:
            # Otherwise, just update the image data on the existing imshow.
            self.im.set_data(heatmap)
            # Reset color limits or keep them, depending on your preference
            self.im.set_clim(vmin=None, vmax=None)

        self.canvas.draw()
        print("Heatmap opened!")

    def set_max_cb_val(self):
        """Just update the vmax on the existing heatmap."""
        if self.im is None:
            print("No heatmap to update. Please open the heatmap first.")
            return

        max_val = float(self.max_cb_val_entry.get())
        self.im.set_clim(vmax=max_val)
        # The colorbar will reflect the new scale after the canvas is redrawn
        self.canvas.draw()
        print(f"Heatmap updated! New max colorbar value: {max_val}")




if __name__ == "__main__":
    recording_f = mea_analysis.get_recording(binary_path)
    _, st, clu, best_cha_st, vectorized_map = mea_analysis.get_stdf(parent_folder_path)
    #remove later
    #st = st[best_cha_st<2500]
    #clu = clu[best_cha_st<2500]
    #best_cha_st = best_cha_st[best_cha_st<2500]



    app = Main_Window()
    app.mainloop()
