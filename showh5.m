function data = showh5(file)
% file = the .hdf5 datafile to be loaded

% the function displays a series of images from a .hdf5 file
% (taken from the 3DMask attack dataset) with the following fields:
%   Color_Data  = a 4D matrix representing the rgb video:
%                   width x heigth x channels x frames
%   Depth_Data  = a 4D martix representing the depth map:
%                   width x heigth x 1 x frames
%   Eye_Pos     = a 2D matrix representing the position of the eyes for each frame:
%                   4 x frames

    %load the data
    rgb = hdf5read(file, 'Color_Data');
    depth = hdf5read(file, 'Depth_Data');
    eyes = hdf5read(file, 'Eye_Pos');

    %rotate the videos
    rgb = permute(rgb, [2 1 3 4]);
    depth = permute(depth, [2 1 3 4]);

    %show each frame
    for i=1:300
        frame = rgb(:,:,:,i);           %rgb frame
        dframe = depth(:,:,:,i);        %deapth map frame
        ix = [eyes(1,i) eyes(3,i)];     %eyes xs
        iy = [eyes(2,i) eyes(4,i)];     %eyes ys

        imshow(frame);                  %plot rgb
        hold on;
        dframe = bitand(dframe, 255);
        dframe = uint8(dframe);
        h = imshow(dframe);    %superimpose depth map
        set(h, 'AlphaData', 0.5);       %with an alpha
        plot(ix,iy, 'o');               %superimpose eyes
        hold off;

        pause(1/100);
    end
end
