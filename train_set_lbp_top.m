clear all; clc;

%% extract the data and 
% save them to data.mat and groups.mat

train_masks = 'C:\Users\pc\Desktop\MDS project 3\H.P.4-20170305T210911Z-006\H.P.4\3DMask\train\fake\';
train_faces = 'C:\Users\pc\Desktop\MDS project 3\H.P.4-20170305T210911Z-006\H.P.4\3DMask\train\real\';

features = [];
groups = [];

%% extract features for masks training set
x = ls([train_masks,'*.hdf5']);
for i=1:size(x,1)
features = [features; feature_extractor_LBP_TOP([train_masks,x(i,:)],x(i,1:8))];
groups = [groups; 'fake'];
end;

%% extract features for real faces training set

x = ls([train_faces,'*.hdf5']);
for i=1:size(x,1)
    features = [features; feature_extractor_LBP_TOP([train_faces,x(i,:)],x(i,1:8))];
    groups = [groups; 'real'];
end;

%% save features and annotation

save('data.mat','features');
save('groups.mat','groups');
    
%% train svm classifier

SVMStruct = svmtrain(features, groups);

save('SVM.mat','SVMStruct');
