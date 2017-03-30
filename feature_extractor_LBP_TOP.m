function [features] = feature_extractor_LBP_TOP( file,videoName )
%% load the video
rgb = hdf5read(file, 'Color_Data');

%% rotate the video
rgb = permute(rgb, [2 1 3 4]);
    
    % Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
videoFrame = rgb(:,:,:,1);
bbox       = step(faceDetector, videoFrame);

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
% figure; imshow(videoFrame); title('Detected face');

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1, :));

%%
% Detect feature points in the face region.
points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);

% Display the detected points.
% figure, imshow(videoFrame), hold on, title('Detected features');
% plot(points);

%%
% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, videoFrame);

%%
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);

%%
% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;
srgb=size(rgb);

%fileID = fopen('f_names.txt','w');
mkdir(videoName);
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
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);

        % Insert a bounding box around the object being tracked
%         bboxPolygon = reshape(bboxPoints', 1, []);
%         videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
%             'LineWidth', 2);

        % Display tracked points
%         videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
%             'Color', 'white');

        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
        angle = rad2deg(atan((bboxPoints(4,2)-bboxPoints(3,2))/(bboxPoints(4,1)-bboxPoints(3,1))));
        videoFrame = imrotate(videoFrame,angle,'crop');
        bbox = [bboxPoints(1,1), bboxPoints(1,2), bboxPoints(3,1)-bboxPoints(4,1), bboxPoints(4,2)-bboxPoints(1,2)];
        videoFrame = imcrop(videoFrame,bbox);
        videoFrame = imresize(videoFrame, [128 128]);
        
        s = cat(2,videoName,'\Frame');
        s = strcat(s,int2str(i));
        s = strcat(s,'.bmp');
        imwrite(videoFrame,s);
        %fprintf(fileID,'%s ',s);
         %videoFrameNorm = Normalization(videoFrame);
         %Horizontal(i,:) = videoFrameNorm(64,:);
    end
     
     
        end
%% Normalization and LbP_top Data
 command = cat(2,'face_land.exe',' ','shape_predictor_68_face_landmarks.dat',' ',videoName);
 system(command);
 Csv_file = cat(2,videoName,'\',videoName,'.csv');
 M = csvread(Csv_file,1,0); % matrix contains all the coordinates
 for j=1:300
     xc = [M(j,1) M(j,3) M(j,5)];
     yc = [M(j,2) M(j,4) M(j,6)];
     s = cat(2,videoName,'\Frame');
     s = strcat(s,int2str(j));
     s = strcat(s,'.bmp');
     face=imread(s);
     [IN] = Normalization(xc,yc,face);  %face normalization
     IN = rgb2gray(IN);
     Horizontal(j,:) = IN(64,:); %take from each frame a Horizontal line from the middle (For the LBP_top)
     Vertical(:,j) = IN(:,64); %take from each frame a vertical line from the middle (For the LBP_top)
     if j==150 
         IN_middle = IN; 
     end
 end
%%
 %extract lbp_top features
 Horizontal_features = extractLBPFeatures(Horizontal);
 Vertical_features = extractLBPFeatures(Vertical);
 MiddleFrame_features = extractLBPFeatures(IN_middle);
 features = cat(2,Horizontal_features,Vertical_features,MiddleFrame_features);
 
%% Clean up
release(videoPlayer);
release(pointTracker);
end
