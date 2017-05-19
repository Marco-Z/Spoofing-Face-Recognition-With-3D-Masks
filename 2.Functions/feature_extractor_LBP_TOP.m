%% FEATURE_EXTRACTOR_LBP_TOP
%  Function used to extract LBP-TOP features of a video
%  It saves the extracted frames in a folder
%  INPUT:
%         - file    : the .hdf5 file that contains the informations
%         - out     : the output folder where to save the extracted frames
%  OUTPUT:
%         - features: LBP-TOP features extracted

function [features] = feature_extractor_LBP_TOP( file, videoName, outa, outb, outc )

outafile = [outa videoName];
outbfile = [outb videoName];

warning('off');
mkdir(outafile);
mkdir(outbfile);
warning('on');

%% load the video
rgb = hdf5read(file, 'Color_Data');

%% rotate the video
rgb = permute(rgb, [2 1 3 4]);

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector('UpperBody');

% Read a video frame and run the face detector.
videoFrame = rgb(:,:,:,1);
bbox       = step(faceDetector, videoFrame);

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Rectangle', bbox);

%% check
% there may be more than 2 faces detected
% choose the biggest box as a rule of thumb
if size(bbox,1) > 1
    s = bbox(:,3).*bbox(:,4);
    [~,i] = max(s);
    bbox = bbox(i,:);
end

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1, :));

%%
% Detect feature points in the face region.

points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox)

%%
% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, videoFrame);

% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;
srgb=size(rgb);

[~,videoName,~] = fileparts(file);
for i = 1:srgb(4)
    % get the next frame
    videoFrame = rgb(:,:,:,i);
    
    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, ~, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
        
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
        angle = rad2deg(atan((bboxPoints(4,2)-bboxPoints(3,2))/(bboxPoints(4,1)-bboxPoints(3,1))));
        videoFrame = imrotate(videoFrame,angle,'crop');
        bbox = [bboxPoints(1,1), bboxPoints(1,2), bboxPoints(3,1)-bboxPoints(4,1), bboxPoints(4,2)-bboxPoints(1,2)];
        videoFrame = imcrop(videoFrame,bbox);
        videoFrame = imresize(videoFrame, [128 128]);
        
        s = cat(2,outafile,'\Frame');
        s = strcat(s,int2str(i));
        s = strcat(s,'.bmp');
        imwrite(videoFrame,s);
        
    end
    
    
end
%% Normalization and LbP_top Data
command = ['face_land.exe shape_predictor_68_face_landmarks.dat "',outafile,'"'];
system(command);

movefile([outafile,'\data.csv'],[outbfile,'\data.csv'])

file = [outbfile,'\data.csv'];
%%
release(pointTracker);
%%
Csv_file = cat(2,outbfile,'\data.csv');
M = csvread(Csv_file,1,0); % matrix contains all the coordinates
for j=1:300
    xc = [M(j,1) M(j,3) M(j,5)];
    yc = [M(j,2) M(j,4) M(j,6)];
    s = [outafile,'\Frame' int2str(j) '.bmp'];
    face=imread(s);
    [IN] = Normalization(xc,yc,face);  %face normalization
    IN = rgb2gray(IN);
    %      IN = imresize(IN, [128 128]);
    Horizontal(j,:) = IN(64,:); %take from each frame a Horizontal line from the middle (For the LBP_top)
    Vertical(:,j) = IN(:,64); %take from each frame a vertical line from the middle (For the LBP_top)
    if j==150
        IN_middle = IN;
    end
end

%extract lbp_top features
imwrite(Horizontal,[outbfile '\Hplane.png']);
imwrite(Vertical,[outbfile '\Vplane.png']);
imwrite(IN_middle,[outbfile '\Mplane.png']);

Horizontal_features = extractLBPFeatures(Horizontal);
Vertical_features = extractLBPFeatures(Vertical);
MiddleFrame_features = extractLBPFeatures(IN_middle);
features = cat(2,Horizontal_features,Vertical_features,MiddleFrame_features);

end
