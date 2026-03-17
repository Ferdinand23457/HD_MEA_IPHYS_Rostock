## High-Density Multielectrode Array (MEA) Recordings in an Ex Vivo Rodent Glioma Model

Ferdinand Forberger a,†, Fabiana Santana Kragelund a,†, Katrin Porath a, Rüdiger Köhling a,b, Falko Lange a,b, Timo Kirschstein a, b

a Oscar-Langendorff-Institute of Physiology, Rostock University Medical Center, 18057 Rostock, Germany
b Center for Transdisciplinary Neurosciences Rostock, University of Rostock, 18147 Rostock, Germany
† These authors contributed equally to this work.

---

This repository hosts the **source code** used to analyze the data presented in the publication, **"High-density multielectrode array (MEA) recordings in a rodent glioma model "**

The code is publicly released to ensure ransparency regarding the data analysis pipeline. However it was not initially developed for general external use; it might therefore not always be very readable. **Reusable and generally useful functions** are located within the `data_analysis` folder as Python (`.py`) modules. **One-time analyses and data exploration** were performed in Jupyter notebooks.

If you have questions about the code or require clarification, please do not hesitate to contact rudolf.forberger@uni-rostock.de.

---

### Codebase Structure Overview

The source code is organized into a primary script for spike sorting, Jupyter notebooks for data calculation, statistics, and plotting, and reusable modules in the `data_analysis` directory.

#### **Spike Sorting**

* **`Spike_sorting.py`**: Contains a loop that iterates through all recordings and executes the **spike sorting** procedure. This script also includes the **parameters used for spike sorting** and the rationale supporting their selection.

#### **Jupyter Notebooks for Calculation, Statistics, and Plotting**

These notebooks contain the step-by-step analysis, statistical testing, and plotting routines.

* **`Calculation_aligning.ipynb`**: Contains the code for **manually aligning** MEA slices/histology images to a common coordinate system for distance and location-based analysis.
* **`Calculation_acg_clustering.ipynb`**: Performs **clustering analysis** on the Autocorrelation Grams (ACGs) of spike sorted units to identify and classify different unit types.
* **`Calculation_NB_onsets.ipynb`**: Calculates the **spatial origin/onset location** of network bursts.
* **`Calculation_velocities.ipynb`**: Calculates the **propagation speed (velocity)** of network bursts across the MEA.
* **`Calculation_graphs_vs_NBs.ipynb`**: Investigates the relationship between **graph theory metrics** and various **Network Burst (NB)** properties.
* **`Calculation_trends_time.ipynb`**: Calculates and plots the **longitudinal trends** of various metrics over the duration of the recording.
* **`CalculationPlotting_Custom_vs_BrainWave6.ipynb`**: Compares and plots metrics derived from the **custom analysis pipeline** against those derived from the commercial **BrainWave6 software**.
* **`CalculationPlotting_N_SUA.ipynb`**: Determines the **number of manually identified Single Unit Activity (SUA) units** found per recording and generates **Figure 2** based on this data.
* **`ClaculationPlotting_graph_metrics.ipynb`**: **Calculates graph theory metrics**, loads the metrics for propagation speed, performs the necessary **statistical testing**, and generates **Figure 4**.
* **`CalculationPlotting_metrics_d_glioma.ipynb`**: Plots the **unit spike frequency** and **unit spike amplitude** as a function of their **distance from the glioma margin**, corresponding to **Figure 6**.
* **`Plotting_overview.ipynb`**: Contains the plotting code for the main **overview figure (Figure 1)**, summarizing the model and experimental setup.
* **`Plotting_raw_data.ipynb`**: Contains the plotting code for key **raw data traces/snippets (Figure 5)**, demonstrating phenomena like network bursts and unit activity.
* **`Plotting_NBs_CoAT.ipynb`**: Contains the plotting code for **Network Burst (NB) characteristics** and the **Center of Activity Trajectory (CoAT)**, generating **Figure 3**.
* **`Plotting_aligned_metrics.ipynb`**: Loads data from the manually aligned slices and uses the alignment information to visualize the **median spike frequency**, **median spike amplitude**, and **median normalized onset of network bursts** relative to a common reference slice. This corresponds to **Supplementary Figure 1**.

#### **`data_analysis` (Reusable Python Modules)**

This folder contains Python modules with functions used across multiple notebooks.

* **`loading_and_saving.py`**: Functions for efficiently **loading and saving** electrophysiology data and intermediate analysis results (e.g., from `.npy`, `.csv`, or other formats).
* **`mea_analysis.py`**: Core functions for basic **MEA data processing**, such as calculating unit metrics, burst detection, and basic filtering. Contains code to calculate the **CoAT** (Function: calculate_center_of_activity_2d", **propagation velocity** (Function: "velocity_field") and many more.
* **`graphs.py`**: Functions dedicated to calculating **graph theory metrics** (e.g., connectivity, path length, clustering coefficient). Also contains the faster, numba-compilable code used to calculated the **STTC**.
* **`plotting.py`**: General-purpose functions for creating **plots** (e.g., custom color maps, statistical plot types).
* **`image_editing.py`**: Functions for **image processing**.

#### **Auxiliary Applications and Files**

* **`slice_aligner_app/aligner.py`**: The Python script for a **standalone GUI application** used for the manual **alignment of slices/histology**.
* **`event_app/find_events_GUI.py`**: The Python script for a **standalone GUI application** used for the manual **identification or verification of network events**.
