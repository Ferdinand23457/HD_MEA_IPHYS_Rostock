import torch
import networkx as nx
import numpy as np
from typing import Any, Dict, List, Set, Callable, Optional
from typing import Union
import torch
from multiprocessing import Queue
import pandas as pd
from joblib import Parallel, delayed
import networkx.algorithms.community as nx_comm
from tqdm.auto import tqdm
from quantities import s
from neo.core import SpikeTrain
from itertools import combinations
import numba
from elephant.utils import check_neo_consistency
import neo
import quantities as pq
import data_analysis.mea_analysis as mea


#####################################################################################################################################################
#Graph building functions
#####################################################################################################################################################

def remove_isolated_nodes(G: nx.Graph) -> nx.Graph:
    """
    Deletes all nodes with degree 0 from the given NetworkX Graph.

    Args:
        G: The NetworkX Graph from which to remove isolated nodes.
           The graph is modified in place.
           
    Returns:
        The modified graph with isolated nodes removed.
    """
    assert isinstance(G, nx.Graph), "Input must be a NetworkX Graph object."

    isolated_nodes = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(isolated_nodes)

    return G


def construct_graph(st_df: pd.DataFrame, sttc_matrix: np.ndarray) -> nx.Graph:
    """
    Constructs a networkx.Graph from a spike-time tiling coefficient matrix
    using a vectorized approach

    Args:
        st_df: DataFrame with unit metadata, including 'clu' (cluster ID).
        sttc_matrix: NumPy array representing the adjacency matrix of the graph.

    Returns:
        An now optimized networkx.Graph object.
    """
    assert np.all((sttc_matrix >= 0) & (sttc_matrix <= 1)), "STTC values must be between 0 and 1."

    # Find all edges in the upper triangle of the matrix where the weight > 0.
    # This avoids adding duplicate edges for an undirected graph and self-loops.
    source_indices, target_indices = np.where(np.triu(sttc_matrix, k=1) > 0)
    weights = sttc_matrix[source_indices, target_indices]

    cluster_ids = st_df["clu"].values
    cluster_ids = cluster_ids.astype(int) #this should not 
    # be necessary but somehow it is guess I should double check the st_df generation....
    
    edge_data = {
        'source': cluster_ids[source_indices],
        'target': cluster_ids[target_indices],
        'weight': weights,
        'pseudo_distance': 1 - weights
    }
    edge_df = pd.DataFrame(edge_data)

    G = nx.from_pandas_edgelist(
        edge_df,
        create_using=nx.Graph(),
        edge_attr=['weight', 'pseudo_distance']
    )

    nodes_in_graph = list(G.nodes())
    attr_df = st_df[st_df['clu'].isin(nodes_in_graph)].copy()
    attr_df['pos'] = list(zip(attr_df['x_coords'], attr_df['y_coords']))
    attr_df['original_df_index'] = attr_df.index
    attr_df = attr_df.set_index('clu')

    nx.set_node_attributes(G, attr_df['pos'].to_dict(), 'pos')
    nx.set_node_attributes(G, attr_df['chan_best'].to_dict(), 'chan_best')
    nx.set_node_attributes(G, attr_df['original_df_index'].to_dict(), 'original_df_index')
    nx.set_node_attributes(G, {node: node for node in G.nodes()}, 'cluster_id')

    print(f"Optimized NetworkX graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def binarize_graph(G:nx.Graph, threshold:float=0) -> nx.Graph:
    """
    Binarizes the graph by setting weights above the threshold to 1 and below to 0.
    """
    G_binarized = G.copy()
    for u, v, data in G_binarized.edges(data=True):
        if data['weight'] > threshold:
            G_binarized[u][v]['weight'] = 1
        else:
            G_binarized[u][v]['weight'] = 0
    return G_binarized


def get_upper_triangle_values_and_indices(matrix: np.ndarray):
    """
    Extracts the values and their original (i, j) indices from the 
    upper triangle of a square matrix.
    """
    num_units = matrix.shape[0]
    values = []
    indices = [] # Store pairs of (i, j)
    for i in range(num_units):
        for j in range(i + 1, num_units):
            values.append(matrix[i, j])
            indices.append((i, j))
    return values, indices


def infer_sttc_from_graph(G: nx.Graph,
                          target_weight_attribute: str = "weight") -> np.ndarray:
    sttc_infered = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    for i, node1 in enumerate(G.nodes()):
        for j, node2 in enumerate(G.nodes()):
            if i != j:
                if G.has_edge(node1, node2):
                    sttc_infered[i, j] = G[node1][node2][target_weight_attribute]
                else:
                    sttc_infered[i, j] = 0
    return sttc_infered

#####################################################################################################################################################
#Graph metrics functions
#####################################################################################################################################################

def weighted_global_efficiency(G, distance_attr='pseudo_distance'):
    N = G.number_of_nodes()
    assert N >= 0, "Number of nodes cannot be negative."

    if N <= 1:
        return 0.0

    accumulated_inverse_path_length = 0.0
    
    all_paths_iterator = nx.all_pairs_dijkstra_path_length(G, weight=distance_attr)

    # the python loop is not the bottleneck here, it is the dijkstra algorithm
    # even with cugraph it is still the bottleneck
    for source, paths in tqdm(all_paths_iterator, desc="Calculating global efficiency", unit="source node"):
        for target, path_length in paths.items():
            if source == target:

                continue

            assert path_length > 0, \
                f"Path length between distinct nodes {source} and {target} must be positive, " \
                f"but got {path_length} using weight attribute '{distance_attr}'."

            accumulated_inverse_path_length += (1.0 / path_length)

    # The number of possible distinct pairs in a graph with N nodes is N * (N - 1).
    # This is the denominator for global efficiency.
    denominator = N * (N - 1)
    
    assert denominator > 0, "Denominator for N > 1 must be positive."

    return accumulated_inverse_path_length / denominator


def compute_global_efficiency_wrapper(G):
    """Use with the distance attribute 'pseudo_distance'."""
    #to be used with distance
    assert "pseudo_distance" in list(G.edges(data=True))[0][2].keys(), "Graph must have 'pseudo_distance' attribute for edges."
    return weighted_global_efficiency(G, distance_attr='pseudo_distance')


def average_clustering_wrapper(G):
    """use with the distance attribute 'weight'."""
    assert "weight" in list(G.edges(data=True))[0][2].keys(), "Graph must have 'weight' attribute for edges." \
    f"but only has {list(G.edges(data=True))[0][2].keys()}"
    return nx.average_clustering(G, weight='weight')


def modularity_wrapper(G):
        communities = nx_comm.louvain_communities(G, weight="weight")
        modularity_score = nx_comm.modularity(G, communities, weight="weight")
        return modularity_score


def average_node_strength_wrapper(G):
    """use with the distance attribute 'weight'."""
    assert "weight" in list(G.edges(data=True))[0][2].keys(), "Graph must have 'weight' attribute for edges."
    if G.number_of_nodes() == 0:
        return 0.0
    else:
        total_strength = sum(strength for node, strength in G.degree(weight='weight'))
        assert total_strength >= 0, "Total strength must be non-negative."
        average_node_strength = total_strength / G.number_of_nodes()
        return average_node_strength


def get_n_relevant_components(G: nx.Graph,minimum_size:int = 5):
    comp = nx.connected_components(G)
    comp = list(comp)
    
    num_revelant_components = 0
    for item in comp:
        if len(item) >= minimum_size:
            num_revelant_components +=1
    return num_revelant_components


#floyd-warshall algorithm on GPU
def global_efficiency(G: nx.Graph, 
                      device: Union[str, torch.device] = 'cpu',
                      progress_queue: Optional[Queue] = None) -> float:
    """
    Computes the global efficiency of a graph using the Floyd-Warshall algorithm on a GPU.
    Global efficiency is the average of the inverse of the shortest path lengths between all pairs of nodes.
    """
    num_nodes = G.number_of_nodes()

    if num_nodes < 2:
        return 0.0

    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    adj_tensor = torch.from_numpy(adj_matrix).to(device).float()

    # Convert correlation (similarity) to distance (cost). Higher correlation = shorter distance.
    adj_tensor = 1.0 - adj_tensor

    # Initialize distance matrix: float('inf') for no path, 0 for self-loops.
    dist = torch.full((num_nodes, num_nodes), float('inf'), dtype=torch.float32, device=device)
    dist.fill_diagonal_(0)
    

    has_edge = adj_tensor > 0
    dist[has_edge] = adj_tensor[has_edge]
    
    # Execute the Floyd-Warshall algorithm to find all-pairs shortest paths.
    for k in range(num_nodes):
        if progress_queue:
            progress_queue.put((k + 1, num_nodes)) 

        dist_ik = dist[:, k].unsqueeze(1)
        dist_kj = dist[k, :].unsqueeze(0)
        new_dist = dist_ik + dist_kj
        dist = torch.min(dist, new_dist)

    efficiency_matrix = torch.zeros_like(dist)
    
    valid_paths_mask = (dist > 0) & (dist != float('inf'))
    
    efficiency_matrix[valid_paths_mask] = 1.0 / dist[valid_paths_mask]

    total_efficiency = torch.sum(efficiency_matrix)
    normalization_factor = num_nodes * (num_nodes - 1)
    
    if normalization_factor == 0:
        return 0.0

    global_eff = total_efficiency / normalization_factor

    return global_eff.item()


def mean_node_strength(G: nx.Graph) -> float:
    """
    Computes the mean node strength of the graph.
    Node strength is defined as the sum of weights of edges connected to the node.

    Args:
        G: A networkx.Graph object with weighted edges.

    Returns:
        The mean node strength as a float.
    """
    if G.number_of_nodes() == 0:
        return 0.0

    strengths = dict(G.degree(weight='weight'))
    mean_strength = np.mean(list(strengths.values()))
    return mean_strength


def average_clustering_pytorch(G: nx.Graph, device: Union[str, torch.device] = 'cpu') -> float:
    """
    Calculates the average clustering coefficient for a weighted, undirected graph.

    This implementation is optimized for dense graphs (e.g., correlation matrices)
    and uses matrix operations in PyTorch for potential GPU acceleration. It assumes
    the graph weights represent connection strength (e.g., correlation) and are
    non-negative.

    The local clustering coefficient for each node `i` is calculated using the
    formulation for weighted networks by Fagiolo (2007):
    C_i^w = ( (A^3)_ii ) / ( s_i^2 - sum_j(A_ij^2) )
    where A is the weighted adjacency matrix and s_i is the strength of node i.

    Args:
        G (nx.Graph): A NetworkX graph with non-negative edge weights.
        device (Union[str, torch.device]): The device for computation ('cpu' or 'cuda').

    Returns:
        float: The average clustering coefficient of the graph.
    """
    num_nodes = G.number_of_nodes()

    if num_nodes < 3:
        return 0.0

    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    A = torch.from_numpy(adj_matrix).to(device, dtype=torch.float32)

    # Calculate the numerator of the local clustering formula.
    # The diagonal of the matrix cubed (A^3)_ii gives the sum of triangle
    # weights connected to each node i.
    A_cubed = torch.matrix_power(A, 3)
    numerators = torch.diag(A_cubed)

    # Calculate the denominator.
    # s_i is the node strength (sum of weights of connected edges).
    node_strengths = torch.sum(A, dim=1)
    # sum_j(A_ij^2) is the sum of squared weights for each node.
    sum_sq_weights = torch.sum(torch.pow(A, 2), dim=1)
    
    denominators = torch.pow(node_strengths, 2) - sum_sq_weights

    # Calculate local clustering coefficients.
    # Handle division by zero for nodes with degree < 2.
    local_clustering = torch.zeros_like(numerators)
    valid_indices = denominators > 0
    local_clustering[valid_indices] = numerators[valid_indices] / denominators[valid_indices]

    avg_clustering = torch.mean(local_clustering)

    return avg_clustering.item()

#####################################################################################################################################################
#correlation between raw traces
#####################################################################################################################################################

def construct_graph_corr_matrix(corr_matrix: np.ndarray, channels: np.ndarray) -> nx.Graph:
    x_coords, y_coords = mea.chans_to_coords(channels)
    combs = list(combinations(range(channels.shape[0]), 2))

    corr_matrix = np.abs(corr_matrix)

    edge_data = []
    def edge_data_iter_step(i,j):
        """
        Helper function to create edge data for a single pair of channels.
        """
        return {
            'source': channels[i],
            'target': channels[j],
            'weight': corr_matrix[i, j],
            'pseudo_distance': 1 - corr_matrix[i, j]
        }
    edge_data = Parallel(n_jobs=-1)(
        delayed(edge_data_iter_step)(i, j) for i, j in tqdm(combs, desc="Constructing edge data", unit="pair")
    )

    G = nx.from_pandas_edgelist(
        pd.DataFrame(edge_data),
        create_using=nx.Graph(),
        edge_attr=['weight', 'pseudo_distance']
    )
    pos = {chan: (x_coords[i], y_coords[i]) for i, chan in enumerate(channels)}
    chans = {chan: chan for chan in channels} 
    nx.set_node_attributes(G, pos, 'pos')
    nx.set_node_attributes(G, chans, 'chan')

    return G

#####################################################################################################################################################
#faster STTC
#####################################################################################################################################################

@numba.njit(cache=True)
def _run_p_numba(times_j: np.ndarray, times_i: np.ndarray, dt: float) -> float:
    """
    Numba-jitted kernel to calculate the proportion of spikes in times_j
    that fall within dt of any spike in times_i.
    """
    num_spikes_j = len(times_j)
    if num_spikes_j == 0:
        return 0.0

    tiled_count = 0
    for i in range(num_spikes_j):
        # Check if a spike in times_j is close to any spike in times_i
        for j in range(len(times_i)):
            if abs(times_j[i] - times_i[j]) <= dt:
                tiled_count += 1
                break  # Move to the next spike in times_j
    return tiled_count / num_spikes_j


@numba.njit(cache=True)
def _run_t_numba(sorted_spikes: np.ndarray, t_start: float, t_stop: float,
                 dt: float) -> float:
    """
    Numba-jitted kernel to calculate the proportion of total recording time
    'tiled' by the spike train's windows.
    """
    num_spikes = len(sorted_spikes)
    total_duration = t_stop - t_start

    if num_spikes == 0 or total_duration <= 0:
        return 0.0

    # Sum of intervals between spikes, capped at 2*dt
    time_between_spikes = 0.0
    for i in range(num_spikes - 1):
        diff = sorted_spikes[i + 1] - sorted_spikes[i]
        time_between_spikes += min(diff, 2 * dt)

    # Time covered by window before first spike, clipped by t_start
    time_before_first_spike = min(sorted_spikes[0] - t_start, dt)

    # Time covered by window after last spike, clipped by t_stop
    time_after_last_spike = min(t_stop - sorted_spikes[-1], dt)

    total_time_covered = (time_between_spikes + time_before_first_spike +
                          time_after_last_spike)

    return total_time_covered / total_duration


def spike_time_tiling_coefficient(spiketrain_i: neo.core.SpikeTrain,
                                  spiketrain_j: neo.core.SpikeTrain,
                                  dt: pq.Quantity = 0.005 * pq.s) -> float:
    """
    Calculates the Spike Time Tiling Coefficient (STTC) as described in
    :cite:`correlation-Cutts2014_14288`. This version is optimized with Numba.
    
    (Docstring from original function retained for context)
    """
    # input checks
    if dt <= 0 * pq.s:
        raise ValueError(f"dt must be > 0, found: {dt}")

    check_neo_consistency([spiketrain_j, spiketrain_i], neo.core.SpikeTrain)

    if len(spiketrain_i) == 0 or len(spiketrain_j) == 0:
        return np.nan

    # Rescale dt to match spike train units and extract float value
    dt_val = dt.rescale(spiketrain_i.units).item()

    # Extract raw data for Numba kernels
    times_i = spiketrain_i.times.magnitude
    times_j = spiketrain_j.times.magnitude
    t_start = spiketrain_i.t_start.item()
    t_stop = spiketrain_i.t_stop.item()

    # Ensure spike times are sorted for _run_t_numba
    if np.any(np.diff(times_i) < 0):
        times_i = np.sort(times_i)
    if np.any(np.diff(times_j) < 0):
        times_j = np.sort(times_j)

    # Call the fast Numba kernels
    TA = _run_t_numba(times_j, t_start, t_stop, dt_val)
    TB = _run_t_numba(times_i, t_start, t_stop, dt_val)
    PA = _run_p_numba(times_j, times_i, dt_val)
    PB = _run_p_numba(times_i, times_j, dt_val)

    # Final STTC calculation
    # Check for 1.0 values to avoid division by zero, replacing 0/0 with 1
    term1, term2 = 0.0, 0.0
    if PA * TB == 1.0:
        term1 = 1.0
    else:
        term1 = (PA - TB) / (1.0 - PA * TB)

    if PB * TA == 1.0:
        term2 = 1.0
    else:
        term2 = (PB - TA) / (1.0 - PB * TA)

    index = 0.5 * (term1 + term2)

    return index


def calculate_sttc_matrix(neo_spike_trains: list[SpikeTrain],
                          dt_s: float = 0.005, 
                            sttc_threshold: float = 0.3,
                            plot_histogram: bool = False,
                            verbose: bool = False) -> np.ndarray:
    dt_quantity = dt_s * s

    num_units = len(neo_spike_trains)
    combs_indices = list(combinations(range(num_units), 2)) # Pairs (DataFrame row index, DataFrame row index)
    sttc_matrix = np.zeros((num_units, num_units))

    sttc_values = Parallel(n_jobs=-1)(
        delayed(spike_time_tiling_coefficient)(
            spiketrain_i=neo_spike_trains[idx_i], # Corresponds to st_df.iloc[idx_i]
            spiketrain_j=neo_spike_trains[idx_j], # Corresponds to st_df.iloc[idx_j]
            dt=dt_quantity 
        ) for idx_i, idx_j in tqdm(combs_indices, desc="STTC Calculation", unit="pair")
    )

    for k, (idx_i, idx_j) in enumerate(combs_indices):
        value = sttc_values[k]
        sttc_matrix[idx_i, idx_j] = value
        sttc_matrix[idx_j, idx_i] = value # STTC is symmetric
    if plot_histogram:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(sttc_values)
        plt.show()
    if verbose:
        min_val = np.min(sttc_values)
        max_val = np.max(sttc_values)
        percentage_below_threshold = np.sum(sttc_matrix <= sttc_threshold) / len(sttc_matrix.flatten()) * 100
        print(f"STTC values range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"Percentage of STTC values below threshold ({sttc_threshold}): {percentage_below_threshold:.2f}%")
    sttc_matrix[sttc_matrix <= sttc_threshold] = 0.0

    return sttc_matrix