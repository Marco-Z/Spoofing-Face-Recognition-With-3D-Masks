%% EXTRACTOR
%  Function used to extract LBP features for each .hdf5 file
%  It saves the extracted frames in a folder
%  INPUT:
%         - depth: boolean, whether to extract the depth data


function [] = extractor( depth )
disp('---------------');
disp('Face extraction');
disp('---------------');

folder = '..\1.Dataset\3DMAD\';

fs = ['train\fake\'; ...
    'train\real\'; ...
    'dev\fake\  '; ...
    'dev\real\  '; ...
    'test\fake\ '; ...
    'test\real\ '];

of = '..\3.Results\a.faces\';

%% folder creation
warning('off');

if depth
    mkdir([of,'depth\']);
    for i=1:size(fs,1)
        mkdir([of,'depth\',strtrim(fs(i,:))]);
    end
end
mkdir([of,'rgb\']);
for i=1:size(fs,1)
    mkdir([of,'rgb\',strtrim(fs(i,:))]);
end

warning('on');

% face extraction
for i=1:size(fs,1)
    in = [folder,strtrim(fs(i,:))];
    disp(strtrim(fs(i,:)));
    
    if depth
        out = [of,'depth\',strtrim(fs(i,:))];
    else
        out = [of,'rgb\',strtrim(fs(i,:))];
    end
    
    f = ls([in,'*.hdf5']);
    
    temp = 0;
    fprintf('|');
    for j=1:size(f,1)
        face_extractor([in,f(j,:)], out, depth);
        k = round(j/size(f,1)*20);
        if k > temp
            temp = k;
            fprintf('#');
        end
    end;
    disp('|');
    
end
end