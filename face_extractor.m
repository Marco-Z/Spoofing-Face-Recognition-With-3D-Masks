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
   
    face=imcrop(frame,BB); % Crop the face
    [~,name,~] = fileparts(file);
 
    out = [folder '/' name '.bmp'];
    imwrite(face,out);
    
    if(d)
        depth = hdf5read(file, 'Depth_Data');
        depth = permute(depth, [2 1 3 4]);
        dframe = depth(:,:,:,20);
        dface=imcrop(dframe,BB); % Crop the face
        dout = [folder '_d/' name '.mat'];
        save(dout,'dface');
    end
end