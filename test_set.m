clear all; clc;

%% extract the data and 
% save them to data.mat and groups.mat

train_fake = 'E:\Marco\drive\3DMask\test\fake\';
train_real = 'E:\Marco\drive\3DMask\test\real\';

out_fake = 'test_fake'; 
out_real = 'test_real';

test_features = [];
test_groups = [];

%% extract faces

mkdir(out_fake);
f = ls([train_fake,'*.hdf5']);
for i=1:size(f,1)
    face_extractor([train_fake,f(i,:)], out_fake);
end;

mkdir(out_real);
%%
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
    test_features = [test_features; feature_extractor([out_fake '/' name '.bmp'], fake(i,:))];
    test_groups = [test_groups; 'fake'];
end;

for i=1:size(r,1)
    [~,name,~] = fileparts(r(i,:));
    test_features = [test_features; feature_extractor([out_real '/' name '.bmp'], real(i,:))];
    test_groups = [test_groups; 'real'];
end;

%% save features and annotation

save('test_data.mat','test_features');
save('test_groups.mat','test_groups');
    
%% use SVM to classify

SVM = load('SVM.mat');
out_groups = svmclassify(SVM.SVMStruct,test_features);

%% print results

for i = 1: size(out_groups,1)
    fprintf('%s\t%s\n',test_groups(i,:),out_groups(i,:));
end;