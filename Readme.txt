The dataset contains shapes of unit cells of phononic crystals (inputs) in the form of images and corresponding dispersion diagrams (outputs). The dataset is used for deep learning (DL) model training.
Outputs are in the form of .mat files which contain vectors of reduced wavevector and corresponding frequencies, and also displacements u, v, w which can be used for polarization calculation.

The dataset contains 11000 cases (4000 cross-like, 4000 diagonal-like, 3000 blot-like cavity shapes). 

Please note that the only 9000 cases were used in our paper linked to the dataset.
The following case numbers were used:

1001 - 4000 cavity_polygon defined on 121 points - cross like shape

4001 - 7000 cavity_polygon defined on 121 points - diagonal cross like shape 

8001 - 11000 cavity_polygon defined on 121 points - blot-like shapes

Note: Ignore names "labels" as these are actually inputs to the DL model, not labels.