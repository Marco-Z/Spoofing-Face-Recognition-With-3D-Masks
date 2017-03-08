function [out] = face_extractor( file )
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
    FDetect = vision.CascadeObjectDetector;
    BB = step(FDetect,frame); %returns Bounding Box value that contains [x,y,Height,Width] of the objects of interest.

    face=imcrop(frame,BB); % Crop the face
    [path,name,~] = fileparts(file);
    if strfind(path, 'fake')
        folder = 'fake';
    elseif strfind(path, 'real')
        folder = 'real';
    else
        folder = 'undefined';
    end
    out = [folder '/' name '.bmp'];
    imwrite(face,out);
end