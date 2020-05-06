clc;clear;close all
cd 'C:\VisionPyhton\Project\SLAM'

format long
data_3dPoint = load('Q.txt');
data_cam_params = load('cam_params.txt');
data_optimiz = load('optimizing_matrix.txt');
data_optimiz = sortrows(data_optimiz,2);

% data_optimiz = [max(data_optimiz(:,1))+1, max(data_optimiz(:,2))+1, ...
%     length(data_optimiz(:,3)), 0; data_optimiz];

header = [max(data_optimiz(:,1))+1, max(data_optimiz(:,2))+1, ...
    length(data_optimiz(:,3))];
%%
dlmwrite('optimize2.txt',header)

dlmwrite('optimize2.txt',data_optimiz, '-append')

dlmwrite('optimize2.txt',data_cam_params, '-append')

dlmwrite('optimize2.txt',data_3dPoint, '-append')
