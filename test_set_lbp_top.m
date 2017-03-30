clear all; clc;

%% classify data


test_masks  = 'C:\Users\pc\Desktop\MDS project 3\H.P.4-20170305T210911Z-006\H.P.4\3DMask\test\fake\';
test_faces  = 'C:\Users\pc\Desktop\MDS project 3\H.P.4-20170305T210911Z-006\H.P.4\3DMask\test\real\';

test_features = [];
test_groups = [];

%% extract features for masks test set
x = ls([test_masks,'*.hdf5']);
for i=1:size(x,1)
    disp(x(i,:));
    test_features = [test_features; feature_extractor_LBP_TOP([test_masks,x(i,:)],x(i,1:8))];
    test_groups = [test_groups; 'fake'];
end;

%% extract features for real faces test set

x = ls([test_faces,'*.hdf5']);
for i=1:size(x,1)
    disp(x(i,:));
    test_features = [test_features; feature_extractor_LBP_TOP([test_faces,x(i,:)],x(i,1:8))];
    test_groups = [test_groups; 'real'];
end;

%% use SVM to classify

SVM = load('SVM.mat');
group = svmclassify(SVM.SVMStruct,test_features);

%% print results

for i = 1: size(group,1)
    fprintf('%s\t%s\n',test_groups(i,:),group(i,:));
end;