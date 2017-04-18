function [features, d_features] = feature_extractor( file, pos, d )

    face = imread(file);
    xc = [pos(1) pos(3) pos(5)];
    yc = [pos(2) pos(4) pos(6)];
    %scatter(xc,yc);
    %hold off;
    %close(f);

    % rotate image
    angle = rad2deg(atan((yc(3)-yc(2))/(xc(3)-xc(2))));
    face = imrotate(face,angle,'crop');

    % rotate canthus points
    sz = size(face) / 2;
    sz = sz';
    
    x1 = xc(2) - sz(2);
    y1 = yc(2) - sz(1);
    x2 = xc(3) - sz(2);
    y2 = yc(3) - sz(1);
    x3 = xc(1) - sz(2);
    y3 = yc(1) - sz(1);
    
    rot_mat=[cosd(angle), sind(angle); -sind(angle) ,cosd(angle)];
    old_orig1 = [x1 y1];
    old_orig2 = [x2 y2];
    old_orig3 = [x3 y3];
    
    new_orig1 = old_orig1 * rot_mat';
    new_orig2 = old_orig2 * rot_mat';
    new_orig3 = old_orig3 * rot_mat';
    
    c1(1) = new_orig1(1) + sz(2);
    c1(2) = new_orig1(2) + sz(1);
    c2(1) = new_orig2(1) + sz(2);
    c2(2) = new_orig2(2) + sz(1);
    c3(1) = new_orig3(1) + sz(2);
    c3(2) = new_orig3(2) + sz(1);

    mc = [(c1(1)+c2(1))/2, c1(2)]; %middle point of canthus
    
    %% philtrum shear

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


    %% resize image

    % scale segments to predefined length
    rx = 48/(c2(1)-c1(1));
    ry = 36/(c3(2)-mc(2));

    % transform points to new scale
    c1 = c1 .* [rx ry];

    % resize image
    dim = size(face);
    osize = [ry rx] .* dim(1:end-1);
    face = imresize(face, osize);
 
    %% crop the image

    % shift to predefined position
    xm = c1(1) - 40;
    ym = c1(2) - 48;

    % crop to 128x128
    rect = [xm ym 128 128];
    face = imcrop(face, rect);
    
    %% transform face to grayscale

    face = rgb2gray(face);


    %% extract lbp features

    features = extractLBPFeatures(face);

    %% depth data
    d_features = [];
    if(d)
        [path,name,~] = fileparts(file);
        depth_data = [path '\_d\' name '.mat'];
        depth = load(depth_data, 'dface');
        depth = depth.dface;
        depth = imrotate(depth,angle,'crop');
        depth = imwarp(depth,tform);
        depth = imresize(depth, osize);
        depth = imcrop(depth, rect);
        d_features = extractLBPFeatures(depth);
    end
end

