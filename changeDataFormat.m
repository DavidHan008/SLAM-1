clc;clear;close all
cd 'C:\Users\test\Desktop\skoli2020f\project\SLAM'

 

data_3dPoint = load('Q.txt');
data_cam_params = load('cam_params.txt');
data_optimiz = load('optimizing_matrix.txt');

 

data_optimiz = [max(data_optimiz(:,1)), max(data_optimiz(:,2)), ...
    length(data_optimiz(:,3)), length(data_optimiz(:,3)); data_optimiz];

 


dlmwrite('optimize2.txt',data_optimiz)

 


dlmwrite('optimize2.txt',data_cam_params, '-append')

 

dlmwrite('optimize2.txt',data_3dPoint, '-append')