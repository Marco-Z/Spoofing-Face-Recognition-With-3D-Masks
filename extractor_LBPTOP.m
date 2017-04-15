clear all; clc;

folder = 'E:\Marco\MDS project\3DMAD\';

fs = ['train\fake\'; ...
      'train\real\'; ...
      'dev\fake\  '; ...
      'dev\real\  '; ...
      'test\fake\ '; ...
      'test\real\ '];

of = 'out_LBPTOP\';
%%


data = {};
final_features = {};
% final_d_features = {};
final_groups = {};
    
for i=1:size(fs,1)
    in = [folder,strtrim(fs(i,:))];
    out = [folder,of,strtrim(fs(i,:))];
    f = ls([in,'*.hdf5']);

    features = [];
    % d_features = [];
    groups = [];
    for j=1:size(f,1)
%         try
            disp(f(j,:))
            [~,videoName,~] = fileparts(f(j,:));
            tf = feature_extractor_LBP_TOP([in,f(j,:)], [out,videoName]);
            features = [features; tf]; 
%             d_features = [d_features; dtf];
            if strfind(fs(i,:),'fake')
                group = 'fake';
            else
                group = 'real';
            end
            groups = [groups; group];
%         catch e
%             disp(e.message);
%             disp(['error: ',f(j,:)]);
%         end
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

