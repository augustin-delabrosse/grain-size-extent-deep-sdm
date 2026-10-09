# Grain Size and Extent in Deep Species Distribution Models



This repository contains the code, example notebooks, trained models and Grad-CAM outputs associated with the study:



**Revisiting the grain-size concept in deep species distribution models: effects of spatial resolution and extent on emerging aquatic insects**



The study investigates how spatial resolution, image-window (or image patch) extent and sensor type affect deep species distribution models (Deep SDMs) for Plecoptera and Trichoptera.



## Repository Structure



```text

├── config.py                         # Configuration and data paths
├── models.py                         # Deep SDM architectures
├── preprocessing.py                  # Data preprocessing utilities
├── gradcam.py                        # Grad-CAM implementation
├── requirements.txt                  # Python dependencies
├── emissions.csv                     # Computational-emissions record
├── __init__.py
│
├── *.ipynb                           # Example analysis notebooks
├── gradcams/                         # Example Grad-CAM heatmaps and RGB images
└── models/                           # Trained model weights
    ├── pleco\_dronems2m5.h5
    └── pleco\_satellite10m.h5

```



The example notebooks cover different taxa, window extents and remote-sensing data configurations, including multispectral and LiDAR data.



## Getting Started



### Installation



Conda is recommended for creating an isolated environment.



```bash

git clone https://github.com/augustin-delabrosse/grain-size-extent-deep-sdm.git
cd grain-size-extent-deep-sdm

conda env create -f environment.yml
conda activate deepsdm_grain_extent

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
#### Notes

GDAL is installed through Conda-forge and should not be installed using:

```bash
pip install gdal
```



If required, update the data paths and other settings in `config.py` before running the notebooks.



### Notebooks



The repository contains example notebooks for:



- Plecoptera and Trichoptera predictions;
- different window extents: 40 m, 70 m and 100 m;
- multispectral, LiDAR and combined data;
- Grad-CAM visualisation and comparison.


The notebooks are provided as examples of the analysis workflow. Reproducing the complete study requires access to the full remote-sensing datasets and occurrence data described below.



## Dataset



The study uses:



- drone multispectral imagery at 31 cm and 2.5 m spatial resolution;
- drone LiDAR data at 31 cm spatial resolution;
- IGN LiDAR data at 50 cm spatial resolution;
- Sentinel-2 multispectral imagery at 10 m spatial resolution;
- super-resolution multispectral imagery at 2.5 m spatial resolution;
- Plecoptera and Trichoptera occurrence data collected using sticky traps.


> Complete datasets: loremispum # Zenodo



## Results



The experiments show that both spatial resolution and image-window extent substantially affect Deep SDM performance.



The main findings are:

- The best configuration for Plecoptera used 2.5 m multispectral data with a 40 m window extent, achieving an AUC of approximately 0.87.
- The best configuration for Trichoptera used 2.5 m multispectral data combined with 50 cm LiDAR data and a 100 m window extent, achieving an AUC of approximately 0.79.
- Differences in spatial resolution produced AUC differences of up to approximately 0.26.
- Differences in window extent produced AUC differences of up to approximately 0.16.
- LiDAR and multispectral data showed different spatial patterns in Grad-CAM explanations.
- Systematically testing grain size and spatial extent can provide both predictive and ecological insights.

