function [data, groups] = feature_extractor( file )
%extract lbp features for an image containing a face
%   file: the location of the .hdf5 file

%   data: the cumulative features vector with the previous features
%   groups: the cumulative classification vector with the previous groups

    %% load the video
    rgb = hdf5read(file, 'Color_Data');

    %% rotate the video
    rgb = permute(rgb, [2 1 3 4]);

    %% take a frame
    frame = rgb(:,:,:,61);

    %% Face detection 
    FDetect = vision.CascadeObjectDetector;
    BB = step(FDetect,frame); %returns Bounding Box value that contains [x,y,Height,Width] of the objects of interest.

    face=imcrop(frame,BB); % Crop the face

    %% canthus rotation
    
    % get canthus points form user
    % TODO: can it be automated?
    f = figure;
    imshow(face);
    title('select canthus points');
    
    [xc, yc] = getpts(f);
    close(f);

    % rotate image
    angle = rad2deg(atan((yc(2)-yc(1))/(xc(2)-xc(1))));
    face = imrotate(face,angle,'crop');

    % rotate canthus points
    sz = size(face) / 2;
    sz = sz';
    x1 = xc(1) - sz(2);
    y1 = yc(1) - sz(1);
    x2 = xc(2) - sz(2);
    y2 = yc(2) - sz(1);
    rot_mat=[cosd(angle), sind(angle); -sind(angle) ,cosd(angle)];
    old_orig1 = [x1 y1];
    old_orig2 = [x2 y2];
    new_orig1 = old_orig1 * rot_mat';
    new_orig2 = old_orig2 * rot_mat';
    c1(1) = new_orig1(1) + sz(2);
    c1(2) = new_orig1(2) + sz(1);
    c2(1) = new_orig2(1) + sz(2);
    c2(2) = new_orig2(2) + sz(1);

    mc = [(c1(1)+c2(1))/2, c1(2)]; %middle point of canthus
    
    %% philtrum shear

    % get philtrum position
    % TODO: can it be automated?
    f = figure;
    imshow(face);
    title('select philtrum point');
    line([c1(1),c2(1)],[c1(2),c2(2)],'Marker','.')
    
    [c3(1), c3(2)] = getpts(f);
%     line([mc(1),c3(1)],[mc(2),c3(2)],'Marker','.')
    close(f);

    % shear the image
    dx = c3(1) - mc(1);
    shear = -2*dx/sz(1);
    tform = affine2d([1 0 0; shear 1 0; 0 0 1]);

    % transform the points to warped image
    s = -2*dx;
    dx1 = s*c1(2)/sz(2);
    dx2 = s*c3(2)/sz(2);

    if(shear<0)
        dx1 = dx1-2*s;
        dx2 = dx2-2*s;
    end

    c1(1) = c1(1) + dx1;
    c2(1) = c2(1) + dx1;
    mc(1) = mc(1) + dx1;
    c3(1) = c3(1) + dx2;
    
    
    face = imwarp(face,tform);

    
%     f = figure;
%     imshow(face);
%     line([c1(1),c2(1)],[c1(2),c2(2)],'Marker','.')
%     line([mc(1),c3(1)],[mc(2),c3(2)],'Marker','.')

    %% resize image

    % scale segments to predefined length
    rx = 48/(c2(1)-c1(1));
    ry = 36/(c3(2)-mc(2));

    % transform points to new scale
    c1 = c1 .* [rx ry];
    c2 = c2 .* [rx ry];
    mc = mc .* [rx ry];
    c3 = c3 .* [rx ry];

    % resize image
    dim = size(face);
    osize = [ry rx] .* dim(1:end-1);
    face = imresize(face, osize);
    
%     line([c1(1),c2(1)],[c1(2),c2(2)],'Marker','.')
%     line([mc(1),c3(1)],[mc(2),c3(2)],'Marker','.')

    %% crop the image

    % shift to predefined position
    xm = c1(1) - 40;
    ym = c1(2) - 48;

    % transform the points to new space
    c1 = c1 - [xm ym];
    c2 = c2 - [xm ym];
    mc = mc - [xm ym];
    c3 = c3 - [xm ym];
    
    % crop to 128x128
    rect = [xm ym 128 128];
    face = imcrop(face, rect);
    
%     close(f);
%     f = figure;
%     imshow(face);
%     line([c1(1),c2(1)],[c1(2),c2(2)],'Marker','.')
%     line([mc(1),c3(1)],[mc(2),c3(2)],'Marker','.')

    %% transform face to grayscale

%     close(f);
    face = rgb2gray(face);

    %% extract lbp features

    features = extractLBPFeatures(face);
    
%     f=figure;
%     stem(features,'.');
%     close(f);

    %% save features data

    try
        load('data.mat');
    catch
        data = [];
    end

    data = [data; features];
    save('data.mat','data');

    %% save grouping of data

    try
        load('groups.mat');
    catch
        groups = [];
    end

%     close(f);
    options = ['real';'mask'];
    choice = menu('category','real','mask');

    groups = [groups; options(choice,:)];
    save('groups.mat','groups');

end

