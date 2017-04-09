clear all; clc;

folder = 'E:\Marco\MDS project\3DMAD\out\';

fs = ['train\fake'; ...
      'train\real'; ...
      'dev\fake  '; ...
      'dev\real  '; ...
      'test\fake '; ...
      'test\real '];

%% extract landmarks

data = {};
final_features = {};
% final_d_features = {};
final_groups = {};

for i=1:size(fs,1)
    out_folder = [folder,strtrim(fs(i,:))];
    command = ['face_land.exe shape_predictor_68_face_landmarks.dat "', ...
            out_folder,'"'];
%     system(command);
    
    file = [out_folder,'\data.csv'];
    data = cat(2,data,csvread(file,1,0));

    disp(out_folder);
    f = ls([out_folder,'\*.bmp']);
    
    pos = cell2mat(data(i));
    
    features = [];
    d_features = [];
    groups = [];
    
    pos = cell2mat(data(i));
    for j=1:size(f,1)
        disp(f(j,:));
        [~,name,~] = fileparts(f(j,:));
        [tf, dtf] = feature_extractor([out_folder '\' name '.bmp'], pos(j,:), false); % true/false to extract also depth
        features = [features; tf]; 
%         d_features = [d_features; dtf];
        if strfind(fs(i,:),'fake')
            group = 'fake';
        else
            group = 'real';
        end
        groups = [groups; group];
    end;
    
    final_features = cat(2,final_features,features);
%     final_d_features = cat(2,final_d_features,d_features);
    final_groups = cat(2,final_groups,groups);
end

%% format and save the data

train_features = [cell2mat(final_features(1));cell2mat(final_features(2))];
save([folder,'train_data.mat'],'train_features');
% train_d_features = [cell2mat(final_d_features(1));cell2mat(final_d_features(2))];
% save([folder,'train_data_d.mat'],'train_d_features');
train_groups = [cell2mat(final_groups(1));cell2mat(final_groups(2))];
save([folder,'train_groups.mat'],'train_groups');

dev_features = [cell2mat(final_features(3));cell2mat(final_features(4))];
save([folder,'dev_data.mat'],'dev_features');
% dev_d_features = [cell2mat(final_d_features(3));cell2mat(final_d_features(4))];
% save([folder,'dev_data_d.mat'],'dev_d_features');
dev_groups = [cell2mat(final_groups(3));cell2mat(final_groups(4))];
save([folder,'dev_groups.mat'],'dev_groups');

test_features = [cell2mat(final_features(5));cell2mat(final_features(6))];
save([folder,'test_data.mat'],'test_features');
% test_d_features = [cell2mat(final_d_features(5));cell2mat(final_d_features(6))];
% save([folder,'test_data_d.mat'],'test_d_features');
test_groups = [cell2mat(final_groups(5));cell2mat(final_groups(6))];
save([folder,'test_groups.mat'],'test_groups');






