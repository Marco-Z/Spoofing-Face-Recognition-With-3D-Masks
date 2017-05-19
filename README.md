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

`run_main.m` is the runner script.
You can tune the parameters of the program:

- features_type: type of features to use
                 possible values: 'rgb', 'depth','lbp-top'

- test_only    : whether to make only the test step, useful
                 if the features have already been extracted
                 useful if the features have already been extracted
                 we provide already extracted features in Code\3.Results\c.features\
                 feature extraction can last up to ~2 hours for lbp-top

- c            : the c parameters for the SVM

- gamma        : the gamma parameters for the SVM
