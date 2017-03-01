file = 'C:\Users\pc\Desktop\MDS project 2\H.P.4-20170114T111843Z-6\H.P.4\3DMask\train\real\03_01_01.hdf5';
rgb = hdf5read(file, 'Color_Data');
eyes = hdf5read(file, 'Eye_Pos');

%rotate the videos
rgb = permute(rgb, [2 1 3 4]);

frame = rgb(:,:,:,20); 

ex = [eyes(1,20) eyes(3,20)];     %eyes xs
ey = [eyes(2,20) eyes(4,20)];     %eyes ys
[nx,ny] = NoseDetection(frame);

imshow(frame);                  %plot rgb
hold on;
plot(ex,ey, 'o');               %superimpose eyes
plot(nx,ny, 'o');
hold off;


