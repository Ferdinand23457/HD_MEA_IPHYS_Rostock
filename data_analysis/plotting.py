# This file contains helper functino for plotting some of the plots used in the publication.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
import matplotlib.colors as mcolors
import numpy as np
import spikeinterface.full as si

from typing import Tuple, Optional, Any
import sys
import data_analysis.mea_analysis
from data_analysis.mea_analysis import SAMPLING_RATE, get_st_snippet

def plot_image_as_flat_surface(
    img: np.ndarray,
    a: float,
    b: float,
    z: float,
    ax: Axes3D,
    *args : Any,
    **kwargs : Any
) -> Axes3D:
    """
    Loads an image and displays it as a flat surface in a 3D Matplotlib plot,
    parallel to the XY plane.
    """

    # Normalize image data to [0, 1] range if it's not already float
    # (mpimg aparently often returns float32 in [0,1] but can return uint8)
    if img.dtype == np.uint8:
        img = img / 255.0
    elif img.max() > 1.0: # Handle cases where it might be float but > 1
         img = img / img.max()


    # Get image dimensions (height H, width W)
    img_h, img_w = img.shape[:2]

    # Create Coordinate Grids
    # Create X and Y coordinates spanning the desired dimensions a and b
    # We use the image dimensions (img_w, img_h) as the number of points
    # in our grid to map the image pixels directly.
    # Create X and Y coordinates spanning the desired dimensions a and b
# starting from 0 to meet the top-left requirement.
    x_coords = np.linspace(0., a, img_w)  # X goes from 0 to a
    y_coords = np.linspace(0., b, img_h)  # Y goes from 0 to b

    # Create meshgrid.
    # X[0, 0] will be 0, Y[0, 0] will be 0.
    X, Y = np.meshgrid(x_coords, y_coords)

    # Create the Z grid - it's flat, so all values are 'z'
    Z = np.full_like(X, z)

    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=img, 
        rstride=5, cstride=5, zorder=1, 
        *args,
        **kwargs
    )



    return ax


def plot_3d_time_cloud_with_image_surface(
    coords: np.ndarray,
    img: np.ndarray,
    xy_range: Tuple[float, float] = (3.8, 3.8),
    image_t_offset: float = 0.0,
    cmap : str ="inferno",
    image_alpha: float = 1.0,
    vmin: float = None,
    vmax: float = None,
    one_color: str = None,
    img_surface_clip_on: bool = False,
    *args: Any,
    **kwargs: Any
) -> Tuple[plt.scatter, Any, plt.colorbar]: # Return type for surface plot object is complex
    """
    Plots a 3D point cloud with time as the vertical axis and an image plane
    using plot_surface.
    """
    print("starting plot_3d_time_cloud_with_image_surface")
    if not isinstance(ax, Axes3D):
         raise TypeError("Input 'ax' must be a Matplotlib Axes3D object.")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Input 'coords' must be a NumPy array with shape (-1, 3).")
    
    x_max, y_max = xy_range
    
    #Plot the 3D time point cloud ---
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    t_coords = coords[:, 2]

    #for some reason all newer versions always plot surfaces after scatterplots this is a stupid,
    #but a working workaround
                  # High zorder (relative to surface)
    # Prepare and Plot the Image Plane using plot_surface
    # Prepare image colors (normalized, RGBA, flipped)
    if img is not None:

        print("Due to absolute spaghetti code in 3d matplotlib plotting we have to")
        print("use a very slow workaround for a certain problem here... sry")
        cmap_name = cmap
        cmap_name = 'inferno'
        cmap = plt.get_cmap(cmap_name)
        #Normalize the z-values to the range [0, 1]
        
        norm = mcolors.Normalize(vmin=vmin if vmin else np.min(t_coords), 
                                vmax=vmax if vmax else np.max(t_coords))
        #Calculate the RGBA color for each point based on its z-value
        colors = cmap(norm(t_coords))


        n_points = len(t_coords)
        for i in range(n_points):
            ax.plot([x_coords[i]], [y_coords[i]], [t_coords[i]], # Plot single point (note the lists)
                    marker='o',
                    linestyle='None',
                    color=colors[i] if not one_color else one_color,        
                    *args,
                    **kwargs) 



        assert isinstance(img, np.ndarray), "image must be a numpy array"
        ims = plot_image_as_flat_surface(img=img,
                                   a = xy_range[0],
                                   b = xy_range[1],
                                   z = image_t_offset,
                                   alpha= image_alpha,
                                   ax=ax,           
                                   shade=False,
                                   clip_on=img_surface_clip_on)
        
    else:
        scatter_plot = ax.scatter(x_coords, y_coords, t_coords,
                                  vmin=vmin,
                                  vmax=vmax,
                                  cmap=cmap,
                                  c=t_coords, 
                                  *args, **kwargs)
    #Set Limits and Labels ---
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    min_t_data = np.min(t_coords) if len(t_coords) > 0 else image_t_offset
    max_t_data = np.max(t_coords) if len(t_coords) > 0 else image_t_offset
    min_t = min(min_t_data, image_t_offset)
    max_t = max(max_t_data, image_t_offset)
    t_range = max_t - min_t
    padding = t_range * 0.05 if t_range > 1e-6 else 1.0
    ax.set_zlim(min_t - padding, max_t + padding)
    
    print("Finished plot_3d_time_cloud_with_image_surface")
    if ims:
        return ax, ims
    return ax
    

def plot_traces(ax : Axes, 
                start_idx : int, 
                end_idx : int,
                chans: np.ndarray,
                recording : si.BaseRecording,
                sampling_rate : float,
                st : np.ndarray,
                best_cha_st : np.ndarray,
                clu : np.ndarray,
                colors: Optional[list] = None,
                letters: bool = True,
                markers: bool = True,
                *args,
                **kwargs,
                ) -> None:
    #recording has the shape channels, samples
    st_snippet, best_cha_st_snippet, clu_snippet = get_st_snippet(st, best_cha_st, clu, start_idx, end_idx)
    st_snippet = st_snippet - start_idx


    traces = recording.get_traces(start_frame=start_idx, 
                                  end_frame=end_idx,
                                  channel_ids=chans,
                                  return_scaled=True)
    
    num_traces = traces.shape[1]
    mean_ptp = np.mean(np.ptp(traces, axis=0))
    offsets = np.arange(0, mean_ptp*num_traces, mean_ptp)[::-1]
    traces_to_plot = traces + offsets+0.4*offsets
    x_axis = np.arange(0, traces_to_plot.shape[0]) / sampling_rate


    greek_letters = [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$", r"$\epsilon$",
                     r"$\zeta$", r"$\eta$"]
    
    for k in range(num_traces):

        current_raw_traces = traces_to_plot[:, k]
        #raw traces
        ax.plot(x_axis, current_raw_traces, color='black', linewidth=0.5)

        #markers
        mask = best_cha_st_snippet == chans[k]
        spikes = st_snippet[mask].astype(int)
        clus = clu_snippet[mask]
        print(f"Unique clusters for channel {chans[k]}: {np.unique(clus)}")
        color_map = mcolors.ListedColormap(colors) if colors else None

        if markers:


            # If there are no spikes, skip plotting for this channel
            if spikes.size==0:
                print(f"Not spikes to plot for iteration {k}")
            else:
                y_vals = np.zeros_like(spikes, dtype=float) #y coordinates of spikes for the current channels
                y_vals[:] = np.min(current_raw_traces) 

                assert spikes.shape == y_vals.shape, f"spikes and y_vals must have the same shape but have: {spikes.shape, y_vals.shape}"
                
                ax.scatter(spikes/sampling_rate, y_vals,c=clus,cmap=color_map, *args, **kwargs)

        #letters 
        if letters:
            y_coord_letter = np.median(current_raw_traces)
            x_coord_letter = x_axis[0] - (x_axis[-1] - x_axis[0]) * 0.1
            ax.text(x_coord_letter, y_coord_letter, greek_letters[k],
                    color="black", fontsize=8, fontweight="bold",
                    horizontalalignment="left", verticalalignment="center", zorder=10)

    return ax


def get_c_map_colors(n_colors : int = 5):

    #2 is the red used for the LFP publication
    sample_points = np.linspace(0, 1, n_colors)
    inferno_cmap = plt.cm.inferno
    sampled_colors = [inferno_cmap(point) for point in sample_points]
    return sampled_colors


def add_scalarbars(ax : Axes,
                   x_size : float,
                   y_size : float,
                   loc : str = "lower right",
                   x_corner_offset : float = 0.1,
                   y_corner_offset : float = 0.1,
                   x_label : str = "1 s",
                   y_label : str = "100 µV",
                   fontsize : int = 7,
                   horizontal_text_props : dict = {"down_modifier" : 0.05},
                   vertical_text_props : dict = {"right_modifier" : 0.02},
                   **kwargs) -> None:
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()

    if loc == "lower right":
        corner = (x_max + x_corner_offset, y_min + y_corner_offset)
        left_tip = (corner[0] - x_size, corner[1])
        top_tip = (corner[0], corner[1] + y_size)

        horizontal_text_pos_x = (left_tip[0] + corner[0]) / 2
        horizontal_text_pos_y = corner[1] - horizontal_text_props["down_modifier"] * (y_max - y_min)

        vertical_text_pos_x = corner[0] + vertical_text_props["right_modifier"] * (x_max - x_min)
        vertical_text_pos_y = (corner[1] + top_tip[1]) / 2

    else:
        raise NotImplementedError(f"Location {loc} not implemented yet.")
    
    ax.plot([left_tip[0], corner[0]], [left_tip[1], corner[1]], color="black", **kwargs)
    ax.plot([corner[0], top_tip[0]], [corner[1], top_tip[1]], color="black", **kwargs)

    ax.text(horizontal_text_pos_x, horizontal_text_pos_y, x_label,
            horizontalalignment="center", verticalalignment="top", fontsize=fontsize)
    
    ax.text(vertical_text_pos_x, vertical_text_pos_y, y_label,
            horizontalalignment="left", verticalalignment="center", fontsize=fontsize, rotation=90)
    


