clear all; clc;

%% extract the data and 
% save them to data.mat and groups.mat

test_fake = 'E:\Marco\drive\3DMask\test\fake\';
test_real = 'E:\Marco\drive\3DMask\test\real\';

out_fake = 'test_fake'; 
out_real = 'test_real';
out_fake_d = 'test_fake_d';
out_real_d = 'test_real_d';

test_features = [];
d_test_features = [];
test_groups = [];

f = ls([test_fake,'*.hdf5']);
r = ls([test_real,'*.hdf5']);
%% extract faces

mkdir(out_fake);
mkdir(out_fake_d);
for i=1:size(f,1)
    face_extractor([test_fake,f(i,:)], out_fake, true); % true/false to extract also depth
end;

mkdir(out_real);
mkdir(out_real_d);
for i=1:size(r,1)
    face_extractor([test_real,r(i,:)], out_real, true); % true/false to extract also depth
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
    [tf, dtf] = feature_extractor([out_fake '/' name '.bmp'], fake(i,:), true); % true/false to extract also depth
    test_features = [test_features; tf]; 
    d_test_features = [d_test_features; dtf];
    test_groups = [test_groups; 'fake'];
end;

for i=1:size(r,1)
    [~,name,~] = fileparts(r(i,:));
    [tf, dtf] = feature_extractor([out_real '/' name '.bmp'], real(i,:), true); % true/false to extract also depth
    test_features = [test_features; tf]; 
    d_test_features = [d_test_features; dtf];
    test_groups = [test_groups; 'real'];
end;

%% save features and annotation

save('test_data.mat','test_features');
save('test_groups.mat','test_groups');

save('test_data_d.mat','d_test_features');