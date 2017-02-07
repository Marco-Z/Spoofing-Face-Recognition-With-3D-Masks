clear all; clc;

%% classify data


test_masks  = '3DMask/test/fake/';
test_faces  = '3DMask/test/real/';

test_features = [];
test_groups = [];

%% extract features for masks test set
x = ls([test_masks,'*.hdf5']);
for i=1:size(x,1)
    disp(x(i,:));
    test_features = [test_features; feature_extractor([test_masks,x(i,:)])];
    test_groups = [test_groups; 'fake'];
end;

%% extract features for real faces test set

x = ls([test_faces,'*.hdf5']);
for i=1:size(x,1)
    disp(x(i,:));
    test_features = [test_features; feature_extractor([test_faces,x(i,:)])];
    test_groups = [test_groups; 'real'];
end;

%% use SVM to classify

SVM = load('SVMStruct.mat');
groups = svmclassify(SVM.SVMStruct,test_features);

%% print results

for i = 1: size(group,1)
    fprintf('%s\t%s\n',test_groups(i,:),group(i,:));
end;