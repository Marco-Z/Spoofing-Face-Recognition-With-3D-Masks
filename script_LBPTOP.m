split_files       % split files in train, dev and test folders
extractor_LBPTOP  % extract facial images for each file, landmarks, 
                  % normalize faces and extract LBP features
parameters_LBPTOP % compute best c and gamma for svm
test_LBPTOP       % test accuracy and plot ROC