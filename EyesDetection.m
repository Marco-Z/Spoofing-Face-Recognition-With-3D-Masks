function [nx_R,ny_R,nx_L,ny_L] = EyesDetection(file)
EyeDetect1 = vision.CascadeObjectDetector('RightEyeCART');
EyeDetect2 = vision.CascadeObjectDetector('LeftEyeCART');
BB1=step(EyeDetect1,file);
BB2=step(EyeDetect2,file);
%figure,imshow(frame);
%hold on;
%rectangle('Position',BB1(1,:),'LineWidth',2,'LineStyle','-','EdgeColor','b');
%rectangle('Position',BB2(1,:),'LineWidth',2,'LineStyle','-','EdgeColor','b');
%title('Eyes Detection');
%hold off;

points1 = bbox2points(BB1(1,:)) ;
points2 = bbox2points(BB2(1,:)) ;

%RightEye
R1_x1 = points1(1,1);
R1_y1 = points1(1,2);
R1_x2 = points1(3,1);
R1_y2 = points1(3,2);
R2_x1 = points1(2,1);
R2_y1 = points1(2,2);
R2_x2 = points1(4,1);
R2_y2 = points1(4,2);

%LeftEye
L1_x1 = points2(1,1);
L1_y1 = points2(1,2);
L1_x2 = points2(3,1);
L1_y2 = points2(3,2);
L2_x1 = points2(2,1);
L2_y1 = points2(2,2);
L2_x2 = points2(4,1);
L2_y2 = points2(4,2);

%plot lines
%plot([R1_x1 R1_x2], [R1_y1 R1_y2]);
%plot([R2_x1 R2_x2], [R2_y1 R2_y2]);
%plot([L1_x1 L1_x2], [L1_y1 L1_y2]);
%plot([L2_x1 L2_x2], [L2_y1 L2_y2]);

% Compute several intermediate quantities
%Right
Rx12 = R1_x1-R1_x2;
Rx34 = R2_x1-R2_x2;
Ry12 = R1_y1-R1_y2;
Ry34 = R2_y1-R2_y2;
Rx24 = R1_x2-R2_x2;
Ry24 = R1_y2-R2_y2;

%Left
Lx12 = L1_x1-L1_x2;           
Lx34 = L2_x1-L2_x2;
Ly12 = L1_y1-L1_y2;
Ly34 = L2_y1-L2_y2;
Lx24 = L1_x2-L2_x2;
Ly24 = L1_y2-L2_y2;


% Solve for t and s parameters
%Right
tsRight = [Rx12 -Rx34; Ry12 -Ry34] \ [-Rx24; -Ry24];
%left
tsLeft = [Lx12 -Lx34; Ly12 -Ly34] \ [-Lx24; -Ly24];

% Take weighted combinations of points on the line
%Right
P_Right = tsRight(1)*[R1_x1; R1_y1] + (1-tsRight(1))*[R1_x2; R1_y2];
%Left
P_Left = tsLeft(1)*[L1_x1; L1_y1] + (1-tsLeft(1))*[L1_x2; L1_y2];

%coordinates
%Right
nx_R = P_Right(1);
ny_R = P_Right(2);
%Left
nx_L = P_Left(1);
ny_L = P_Left(2);
end
