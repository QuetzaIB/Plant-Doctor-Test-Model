# Plant-Doctor-Test-Model
Hybrid Machine Vision and Segmentation program to analyze the leaf health from videos of plants and trees.

# Setup
Step 1: Create a PyTorch environment and install the requirements from `requirements.txt`.
Step 2: Modify line 152 of the `main.py` function to specify the desired video.

# What can be analyzed?
The machine vision model works best with videos taken from close ups. Our original database used a Nikon D600 with a 300 mm lens.
Currently the segmenation model can just classify between healthy areas of the leaf and "sick" areas, it cannot identify the source of the damage.

# Results
Generated `output.csv`: File contains all the analysis information.
Generated `output.avi`: File shows the results of the ROI (Region of Interest) delineation.
`sort`: Directory will save the cropped images
`result`: Directory will contain the files selected for semantic segmentation
`output`: Directory will store the results of the semantic segmentation.

# List of compatible plants
*Trees*
Acer cissifolium
Cinnamomum yabunikkei
Cornus kousa
Lagerstroemia subcostata
Ligustrum lucidum
Ligustrum lucidum
Prunus speciosa

*Bushes*
Camellia japonica
Cleyera japonica
Elaegnus pungens
Hydrangea scandens
Litsea japonica
Loropetalum chinense
Philadelphus satsumi

# Database
https://doi.org/10.5281/zenodo.15875527

# Models and modules used
YOLOv8, GOLD YOLO, CoTAttention,
DeepSORT, 
DeepLabV3, CBAM

