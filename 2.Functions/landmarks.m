%% LANDMARKS
%  Function used to compute the landmarks for a given image
%  it uses face_land.exe and shape_predictor_68_face_landmarks.dat, taken from dlib
%  INPUT:
%         - depth: boolean, whether to extract also the depth data


function [] = landmarks( depth )
disp('------------------');
disp('Face normalization');
disp('------------------');

folder = '..\3.Results\a.faces\';
out = '..\3.Results\b.normalized\';

fs = ['train\fake'; ...
    'train\real'; ...
    'dev\fake  '; ...
    'dev\real  '; ...
    'test\fake '; ...
    'test\real '];

%% folder creaction
warning('off');

if depth
    mkdir([out,'depth\']);
    for i=1:size(fs,1)
        mkdir([out,'depth\',strtrim(fs(i,:))]);
    end
else
    mkdir([out,'rgb\']);
    for i=1:size(fs,1)
        mkdir([out,'rgb\',strtrim(fs(i,:))]);
    end
end
warning('on');

%% extract landmarks

data = {};
final_features = {};
final_d_features = {};
final_groups = {};

for i=1:size(fs,1)
    
    rgb_folder = [folder,'rgb\',strtrim(fs(i,:))];
    if depth
        in_folder = [folder,'depth\',strtrim(fs(i,:))];
        out_folder = [out,'depth\',strtrim(fs(i,:))];
    else
        in_folder = [folder,'rgb\',strtrim(fs(i,:))];
        out_folder = [out,'rgb\',strtrim(fs(i,:))];
    end
    % call to the compiled program to extract landmarks
    command = ['face_land.exe shape_predictor_68_face_landmarks.dat "', ...
        rgb_folder,'"'];
    system(command);
    
    movefile([rgb_folder,'\data.csv'],[out_folder,'\data.csv'])
    
    file = [out_folder,'\data.csv'];
    data = cat(2,data,csvread(file,1,0));
    
    disp(strtrim(fs(i,:)));
    f = ls([rgb_folder,'\*.bmp']);
    
    pos = cell2mat(data(i));
    
    features = [];
    d_features = [];
    groups = [];
    
    pos = cell2mat(data(i));
    temp = 0;
    fprintf('|');
    for j=1:size(f,1)
        [~,name,~] = fileparts(f(j,:));
        % extract features and save normalized faces
        [tf, dtf] = feature_extractor([rgb_folder '\' name '.bmp'], pos(j,:), depth, out_folder); % true/false to extract also depth
        features = [features; tf];
        d_features = [d_features; dtf];
        if strfind(fs(i,:),'fake')
            group = 'fake';
        else
            group = 'real';
        end
        groups = [groups; group];
        k = round(j/size(f,1)*20);
        if k > temp
            temp = k;
            fprintf('#');
        end
    end;
    disp('|');
    
    final_features = cat(2,final_features,features);
    final_d_features = cat(2,final_d_features,d_features);
    final_groups = cat(2,final_groups,groups);
end

%% format and save the data

feature_out = '..\3.Results\c.features\';

train_groups = [cell2mat(final_groups(1));cell2mat(final_groups(2))];
dev_groups = [cell2mat(final_groups(3));cell2mat(final_groups(4))];
test_groups = [cell2mat(final_groups(5));cell2mat(final_groups(6))];
if depth
    warning('off');
    mkdir([feature_out 'depth\']);
    warning('on');
    train_d_features = [cell2mat(final_d_features(1));cell2mat(final_d_features(2))];
    dev_d_features = [cell2mat(final_d_features(3));cell2mat(final_d_features(4))];
    test_d_features = [cell2mat(final_d_features(5));cell2mat(final_d_features(6))];
    save([feature_out,'depth\train_data_d.mat'],'train_d_features');
    save([feature_out,'depth\dev_data_d.mat'],'dev_d_features');
    save([feature_out,'depth\test_data_d.mat'],'test_d_features');
    save([feature_out,'depth\train_groups.mat'],'train_groups');
    save([feature_out,'depth\dev_groups.mat'],'dev_groups');
    save([feature_out,'depth\test_groups.mat'],'test_groups');
else
    warning('off');
    mkdir([feature_out 'rgb\']);
    warning('on');
    train_features = [cell2mat(final_features(1));cell2mat(final_features(2))];
    dev_features = [cell2mat(final_features(3));cell2mat(final_features(4))];
    test_features = [cell2mat(final_features(5));cell2mat(final_features(6))];
    save([feature_out,'rgb\train_data.mat'],'train_features');
    save([feature_out,'rgb\dev_data.mat'],'dev_features');
    save([feature_out,'rgb\test_data.mat'],'test_features');
    save([feature_out,'rgb\train_groups.mat'],'train_groups');
    save([feature_out,'rgb\dev_groups.mat'],'dev_groups');
    save([feature_out,'rgb\test_groups.mat'],'test_groups');
end

end




