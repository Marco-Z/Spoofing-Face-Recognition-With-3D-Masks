# Spoofing Face Recognition With 3D Masks
========================================

A MatLab project for the course of Multimedia Data Security held at the University of Trento.

The aim of this project is to implement a Matlab script that enables to discern between real faces and spoofing attempts using 3D printed masks in order to gain illegitimate access to systems.




## Requirements
---------------

- windows (for the landmark extraction)
- MATLAB
- python 2 (for the parameters estimation)
- gnuplot (needed by grid.py for parameters estimation)
- 3D Mask Attack Database

## Usage
--------

1. The first script is `split_files.m`.
   This script split the files contained in two folders: real and fake and places them in a directory tree that is exemplified in the file itself.

2. Once you have done this there are two ways:
   - LBP (`extractor.m` and `landmarks.m`)
   - LBP-TOP (`extractor_LBPTOP.m`)

   These scripts extract the faces from each file, compute the landmarks of the face for normalization and normalize the faces.
   The final output are some <.mat> that contain the extracted features of the faces.
   (<.mat> files are provided in the `features` folder)
   
3. The next step is parameters estimation for the SVM classifier.
   This is done with `grid.py` in folder `parameters_estimation`.
   This file takes in input a formatted type of data, that can be obtained converting the <.mat> files in <.csv> files and then into `libsvm` files with `csv2libsvm.py`. (`.csv` files are provided in the `parameters_estimation` folder)
   The output of this script is the parameters cost and gamma that are the best for the classifier.
   The input of this is the dev subset.
   
4. The final step is training-testing.
   There are 3 scripts that do this:
   - `test.m` for LBP on gray-scale data
   - `d_test.m` for LBP on depth data
   - `test_LBPTOP.m` for LBP-TOP
   These scripts build the classifiers and plot the output ROC of the classification.
