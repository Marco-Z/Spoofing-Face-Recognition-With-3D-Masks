%% EXTRACTOR_LBPTOP
%  Function used to extract LBP-TOP features for each .hdf5 file


function [] = extractor_LBPTOP(  )
disp('---------------------------------');
disp('Face extraction and normalization');
disp('---------------------------------');

folder = '..\1.Dataset\3DMAD\';

fs = ['train\fake\'; ...
    'train\real\'; ...
    'dev\fake\  '; ...
    'dev\real\  '; ...
    'test\fake\ '; ...
    'test\real\ '];

outa = '..\3.Results\a.faces\lbp-top\';
outb = '..\3.Results\b.normalized\lbp-top\';
outc = '..\3.Results\c.features\lbp-top\';

%% folders creation
warning('off');

mkdir(outa);
mkdir(outb);
mkdir(outc);

for i=1:size(fs,1)
    mkdir([outa,strtrim(fs(i,:))]);
    mkdir([outb,strtrim(fs(i,:))]);
end
warning('on');

%% extract data
data = {};
final_features = {};
% final_d_features = {};
final_groups = {};

for i=1:size(fs,1)
    disp(strtrim(fs(i,:)));
    in = [folder,strtrim(fs(i,:))];
    outasub = [outa,strtrim(fs(i,:))];
    outbsub = [outb,strtrim(fs(i,:))];
    f = ls([in,'*.hdf5']);
    
    features = [];
    % d_features = [];
    groups = [];
    temp = 0;
    fprintf('|');
    for j=1:size(f,1)
        [~,videoName,~] = fileparts(f(j,:));
        tf = feature_extractor_LBP_TOP([in,f(j,:)], videoName, outasub, outbsub, outc);
        features = [features; tf];
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
    %     final_d_features = cat(2,final_d_features,d_features);
    final_groups = cat(2,final_groups,groups);
end


%% format and save the data
train_features = [cell2mat(final_features(1));cell2mat(final_features(2))];
train_groups = [cell2mat(final_groups(1));cell2mat(final_groups(2))];
dev_features = [cell2mat(final_features(3));cell2mat(final_features(4))];
dev_groups = [cell2mat(final_groups(3));cell2mat(final_groups(4))];
test_features = [cell2mat(final_features(5));cell2mat(final_features(6))];
test_groups = [cell2mat(final_groups(5));cell2mat(final_groups(6))];

save([outc,'train_data.mat'],'train_features');
save([outc,'train_groups.mat'],'train_groups');
save([outc,'dev_data.mat'],'dev_features');
save([outc,'dev_groups.mat'],'dev_groups');
save([outc,'test_data.mat'],'test_features');
save([outc,'test_groups.mat'],'test_groups');

end
