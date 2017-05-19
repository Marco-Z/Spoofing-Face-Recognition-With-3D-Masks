%% FACE_EXTRACTOR
%  Function used to extract faces from a .hdf file
%  It saves the extracted frames in a folder
%  INPUT:
%         - file  : the .hdf5 file that contains the informations
%         - folder: the output folder where to save the extracted frame
%         - d     : boolean, whether to extract the depth data
%  OUTPUT:
%         - out   : the location of the output file

function [out] = face_extractor( file, folder, d )
%extract lbp features for an image containing a face
%   file: the location of the .hdf5 file

%   data: the cumulative features vector with the previous features
%   groups: the cumulative classification vector with the previous groups

%% load the video
rgb = hdf5read(file, 'Color_Data');

%% rotate the video
rgb = permute(rgb, [2 1 3 4]);

%% take a frame
frame = rgb(:,:,:,20);

%% Face detection
FDetect = vision.CascadeObjectDetector('FrontalFaceLBP');
BB = step(FDetect,frame); %returns Bounding Box value that contains [x,y,Height,Width] of the objects of interest.

%% check
% there may be more than 2 faces detected
% choose the biggest box as a rule of thumb
if size(BB,1) > 1
    s = BB(:,3).*BB(:,4);
    [~,i] = max(s);
    BB = BB(i,:);
end

[~,name,~] = fileparts(file);
if d
    depth = hdf5read(file, 'Depth_Data');
    depth = permute(depth, [2 1 3 4]);
    dframe = depth(:,:,:,20);
    dface=imcrop(dframe,BB); % Crop the face
    dout = [folder name '.mat'];
    save(dout,'dface');
end
face=imcrop(frame,BB); % Crop the face
folder = strrep(folder,'depth','rgb');
out = [folder name '.bmp'];
imwrite(face,out);

end
