% with python2

% transform csv file to libsvm file
python csv2libsvm.py file.csv out.data 0 True

% get best c and gamma
python grid.py -svmtrain "path/to/svm-train.exe" -gnuplot "path/to/gnuplot.exe" data.