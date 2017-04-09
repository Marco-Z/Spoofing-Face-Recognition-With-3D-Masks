clear all; clc;

folder = 'E:\Marco\MDS project\3DMAD\';

fs = ['train\fake\'; ...
      'train\real\'; ...
      'dev\fake\  '; ...
      'dev\real\  '; ...
      'test\fake\ '; ...
      'test\real\ '];

of = 'out\';
%%
for i=1:size(fs,1)
    in = [folder,strtrim(fs(i,:))];
    out = [folder,of,strtrim(fs(i,:))];
    f = ls([in,'*.hdf5']);

    for j=1:size(f,1)
        try
            disp(f(j,:))
            face_extractor([in,f(j,:)], out, false);
        catch e
            disp(e.message);
            disp(['error: ',f(j,:)]);
        end
    end;

end