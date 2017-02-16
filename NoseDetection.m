function [nx,ny] = NoseDetection(file)
%I = imread('face1.jpg');
NoseDetect = vision.CascadeObjectDetector('Nose','MergeThreshold',16);
BB=step(NoseDetect,file);
%figure,
%imshow(file); hold on
for i = 1:size(BB,1)
      rectangle('Position',BB(i,:),'LineWidth',2,'LineStyle','-','EdgeColor','b');
      points = bbox2points(BB(i,:)) ;
     
  end
  %title('Nose Detection');


%plot(points(1,1),points(1,2), 'o'); 
%plot(points(2,1),points(2,2), 'o');
%plot(points(3,1),points(3,2), 'o');
%plot(points(4,1),points(4,2), 'o');
%plot([points(1,1) points(3,1)], [points(1,2) points(3,2)]);
%plot([points(2,1) points(4,1)], [points(2,2) points(4,2)]);

% Sample data
L1_x1 = points(1,1);
L1_y1 = points(1,2);
L1_x2 = points(3,1);
L1_y2 = points(3,2);
L2_x1 = points(2,1);
L2_y1 = points(2,2);
L2_x2 = points(4,1);
L2_y2 = points(4,2);
% Plot the lines
plot([L1_x1 L1_x2], [L1_y1 L1_y2]);
plot([L2_x1 L2_x2], [L2_y1 L2_y2]);

% Compute several intermediate quantities
Dx12 = L1_x1-L1_x2;
Dx34 = L2_x1-L2_x2;
Dy12 = L1_y1-L1_y2;
Dy34 = L2_y1-L2_y2;
Dx24 = L1_x2-L2_x2;
Dy24 = L1_y2-L2_y2;

% Solve for t and s parameters
ts = [Dx12 -Dx34; Dy12 -Dy34] \ [-Dx24; -Dy24];

% Take weighted combinations of points on the line
P = ts(1)*[L1_x1; L1_y1] + (1-ts(1))*[L1_x2; L1_y2];
%Q = ts(2)*[L2_x1; L2_y1] + (1-ts(2))*[L2_x2; L2_y2];

% Plot intersection point
%plot(P(1), P(2), 'ro');
%plot(Q(1), Q(2), 'bo')
  
%hold off;
nx = P(1);
ny = P(2);
end