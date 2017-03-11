clear all; clc;

%% extract the data and 
% save them to data.mat and groups.mat

train_fake = 'E:\Marco\drive\3DMask\train\fake\';
train_real = 'E:\Marco\drive\3DMask\train\real\';

out_fake = 'train_fake';
out_real = 'train_real';

train_features = [];
train_groups = [];

%% extract faces

mkdir(out_fake);
f = ls([train_fake,'*.hdf5']);
for i=1:size(f,1)
    face_extractor([train_fake,f(i,:)], out_fake);
end;

mkdir(out_real);
r = ls([train_real,'*.hdf5']);
for i=1:size(r,1)
    face_extractor([train_real,r(i,:)], out_real);
end;

%% extract landmarks

system(['face_land.exe shape_predictor_68_face_landmarks.dat ' out_fake]);
system(['face_land.exe shape_predictor_68_face_landmarks.dat ' out_real]);




%% read landmarks

fake = csvread([out_fake '/' out_fake '.csv'],1,0);
real = csvread([out_real '/' out_real '.csv'],1,0);

%% extract features

for i=1:size(f,1)
    [~,name,~] = fileparts(f(i,:));
    train_features = [train_features; feature_extractor([out_fake '/' name '.bmp'], fake(i,:))];
    train_groups = [train_groups; 'fake'];
end;

for i=1:size(r,1)
    [~,name,~] = fileparts(r(i,:));
    train_features = [train_features; feature_extractor([out_real '/' name '.bmp'], real(i,:))];
    train_groups = [train_groups; 'real'];
end;

%% save features and annotation
save('train_data.mat','train_features');
save('train_groups.mat','train_groups');
    
%% train svm classifier

SVMStruct = svmtrain(train_features, train_groups);

save('SVM.mat','SVMStruct');
