# Quantification of biomarker cells
## Leveraging Deep Learning for Immune Cell Quantification and Prognostic Evaluation in Radiotherapy-Treated Oropharyngeal Squamous Cell Carcinomas
*Laboratory Investigation* - 
 [Journal Link](https://www.sciencedirect.com/science/article/abs/pii/S0023683725000042) 

**Abstract:** The tumor microenvironment (TME) plays a critical role in cancer progression and therapeutic responsiveness, with the tumor immune microenvironment (TIME) being a key modulator. In head and neck squamous cell carcinomas (HNSCC), immune cell infiltration significantly influences the response to radiotherapy (RT). A better understanding of the TIME in HNSCC could help identify patients most likely to benefit from combining RT with immunotherapy. Standardized, cost-effective methods for studying TIME in HNSCC are currently lacking. This study aims to leverage deep learning (DL) to quantify immune cell densities using immunohistochemistry (IHC) in untreated oropharyngeal squamous cell carcinoma (OPSCC) biopsies of patients scheduled for curative RT, and to assess their prognostic value. We analyzed 84 pre-treatment formalin-fixed paraffin-embedded (FFPE) tumor biopsies from OPSCC patients. Immunohistochemistry was performed for CD3, CD8, CD20, CD163, and FOXP3, and whole slide images (WSIs) were digitized for analysis using a U-Net-based DL model. Two quantification approaches were applied: a cell-counting method and an area-based method. These methods were applied to stained regions. The DL model achieved high accuracy in detecting stained cells across all biomarkers. Strong correlations were found between our DL pipeline, the HALO® Image Analysis Platform, and the open-source QuPath software for estimating immune cell densities. Our DL pipeline provided an accurate and reproducible approach for quantifying immune cells in OPSCC. The area-based method demonstrated superior prognostic value for recurrence-free survival (RFS), when compared to the cell-counting method. Elevated densities of CD3, CD8, CD20, and FOXP3 were associated with improved RFS, while CD163 showed no significant prognostic association. These results highlight the potential of DL in digital pathology for assessing TIME and predicting patient outcomes.

## Installation
Please refer to this instructions to use this system to quantify the biomarker cells.

The organization of the repository as follows:
````
data/
	└── Cell count CD3 1000 tiles.xlsx
```

The data folder contains the excel file which provide the counting from two experts for 1000 tiles on CD3.
To find the 1000 tiles in CD3, please refer to the [OME](https://demo.openmicroscopy.org/webclient/?show=dataset-2171)

```
src/
	├── slide_1.svs
	├── slide_2.svs
	└── ...
```

## License
 This code is made available under the GPLv3 License and is available for non-commericial academic purposes.

If you would like to use this system to train the model on your data, you can use the `main.py` to run the training process.
In the `main.py` file, you should modify some parameters to adapt with your configurations:
 - `epochs` = number of epochs to train the model
 - `n-nodes` = number of nodes that you use to train the model
 - `n_gpus` = number of GPUs on each node to train the model
 - `datasets` = the path to the your dataset

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our paper:

Fanny Beltzung, Van Linh Le, Ioana Molnar, Erwan Boutault, Claude Darcha, François Le Loarer, Myriam Kossai, Olivier Saut, Julian Biau, Frédérique Penault-Llorca, Emmanuel Chautard,
Leveraging Deep Learning for Immune Cell Quantification and Prognostic Evaluation in Radiotherapy-Treated Oropharyngeal Squamous Cell Carcinomas.,
Laboratory Investigation,
2025,
104094,
ISSN 0023-6837,
https://doi.org/10.1016/j.labinv.2025.104094.
```
@article{BELTZUNG2025104094,
title = {Leveraging Deep Learning for Immune Cell Quantification and Prognostic Evaluation in Radiotherapy-Treated Oropharyngeal Squamous Cell Carcinomas.},
journal = {Laboratory Investigation},
pages = {104094},
year = {2025},
issn = {0023-6837},
doi = {https://doi.org/10.1016/j.labinv.2025.104094},
url = {https://www.sciencedirect.com/science/article/pii/S0023683725000042},
author = {Fanny Beltzung and Van Linh Le and Ioana Molnar and Erwan Boutault and Claude Darcha and François {Le Loarer} and Myriam Kossai and Olivier Saut and Julian Biau and Frédérique Penault-Llorca and Emmanuel Chautard}
}
```