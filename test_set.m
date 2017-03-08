clear all; clc;

%% extract the data and 
% save them to data.mat and groups.mat

train_masks = 'E:\Marco\drive\3DMask\test\fake\';
train_faces = 'E:\Marco\drive\3DMask\test\real\';

test_features = [];
test_groups = [];

%% extract faces

mkdir('fake');
f = ls([train_masks,'*.hdf5']);
for i=1:size(f,1)
    face_extractor([train_masks,f(i,:)]);
end;

mkdir('real');
r = ls([train_faces,'*.hdf5']);
for i=1:size(r,1)
    test_faces; face_extractor([train_faces,r(i,:)]);
end;

%% extract landmarks

system('face_land.exe shape_predictor_68_face_landmarks.dat fake');
system('face_land.exe shape_predictor_68_face_landmarks.dat real');



%% read landmarks

fake = csvread('fake/fake.csv',1,0);
real = csvread('real/real.csv',1,0);

%% extract features

for i=1:size(f,1)
    [~,name,~] = fileparts(f(i,:));
    test_features = [test_features; feature_extractor(['fake/' name '.bmp'], fake(i,:))];
    test_groups = [test_groups; 'fake'];
end;

for i=1:size(r,1)
    [~,name,~] = fileparts(r(i,:));
    test_features = [test_features; feature_extractor(['real/' name '.bmp'], real(i,:))];
    test_groups = [test_groups; 'real'];
end;

%% use SVM to classify

SVM = load('SVM.mat');
groups = svmclassify(SVM.SVMStruct,test_features);

%% print results

for i = 1: size(groups,1)
    fprintf('%s\t%s\n',test_groups(i,:),groups(i,:));
end;