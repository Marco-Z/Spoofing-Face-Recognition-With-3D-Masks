
 rgb = hdf5read('C:\Users\pc\Desktop\MDS project 2\H.P.4-20170114T111843Z-6\H.P.4\3DMask\train\real\01_01_01.HDF5', 'Color_Data');
 rgb = permute(rgb, [2 1 3 4]);
 %implay(rgb);
 frame = rgb(:,:,:,50);  % extract a frame from the video
 
 %Face detection 
 FDetect = vision.CascadeObjectDetector;
 BB = step(FDetect,frame);
 figure; imshow(frame);
 hold on
for i = 1:size(BB,1)
    rectangle('Position',BB(i,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
end
title('Face Detection');
hold off;
face=imcrop(frame,BB); % Crop the face
figure,imshow(face);

%lbp 
lbp_frame = lbp( face );
figure; plot(lbp_frame);
 