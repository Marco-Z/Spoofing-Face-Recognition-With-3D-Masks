
    rgb = hdf5read('C:\Users\pc\Desktop\MDS project 2\H.P.4-20170114T111843Z-6\H.P.4\3DMask\train\real\01_01_01.HDF5', 'Color_Data');
    eyes = hdf5read('C:\Users\pc\Desktop\MDS project 2\H.P.4-20170114T111843Z-6\H.P.4\3DMask\train\real\01_01_01.HDF5', 'Eye_Pos');
%rotate the videos
 rgb = permute(rgb, [2 1 3 4]);
   
%show one frame with eye position
    
 frame = rgb(:,:,:,50);           %rgb frame        
 x = [eyes(1,50) eyes(3,50)];     %eyes xs
 y = [eyes(2,50) eyes(4,50)];     %eyes ys

 %{
 imshow(frame);                  %plot rgb
 hold on;
 plot(x,y, 'o');               
 hold off; 
 %}

%normalization
x1 = double(x(1));
x2 = double(x(2));
y1 = double(y(1));
y2 = double(y(2));
angle = atan((y2-y1)/(x2-x1)); %the angle between the line connecting the two eyes and the horizontal in Rad
angle = rad2deg(angle);
frame = imrotate(frame,angle);
figure; imshow(frame);

 %Face detection 
 FDetect = vision.CascadeObjectDetector;
 BB = step(FDetect,frame); %returns Bounding Box value that contains [x,y,Height,Width] of the objects of interest.
 figure; imshow(frame);
 hold on
for i = 1:size(BB,1)
    rectangle('Position',BB(i,:),'LineWidth',3,'LineStyle','-','EdgeColor','r');
end
title('Face Detection');
hold off;
face=imcrop(frame,BB); % Crop the face
figure; imshow(face); title('normalized');

%LBP
 w=size(face,1);     
 h=size(face,2);
for i=2:w-1   
      for j=2:h-1  
          J0=face(i,j);   
          I3(i-1,j-1)=face(i-1,j-1)>J0;  
          I3(i-1,j)=face(i-1,j)>J0;   
          I3(i-1,j+1)=face(i-1,j+1)>J0;  
          I3(i,j+1)=face(i,j+1)>J0;     
          I3(i+1,j+1)=face(i+1,j+1)>J0;    
          I3(i+1,j)=face(i+1,j)>J0;      
          I3(i+1,j-1)=face(i+1,j-1)>J0;     
          I3(i,j-1)=face(i,j-1)>J0; 
         %matrix contain the decimal values of each byte obtained
      LBP(i-1,j-1)=I3(i-1,j-1)*2^7+I3(i-1,j)*2^6+I3(i-1,j+1)*2^5+I3(i,j+1)*2^4+I3(i+1,j+1)*2^3+I3(i+1,j)*2^2+I3(i+1,j-1)*2^1+I3(i,j-1)*2^0;        
end  
end 
   