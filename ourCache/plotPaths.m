clc;clear;close all
cd 'C:\Users\Ole\Desktop\Project\SLAM';

% data = load('C:\Users\Ole\Desktop\Project\KITTI_sequence_2\poses.txt');
data = load('C:\Users\Ole\Desktop\Project\dataset\sequences\06\poses.txt');
figure
hold on

 


for i=1:size(data,1)-1
    line([data(i,4),data(i+1,4)],[data(i,12),data(i+1,12)], 'color', 'red', 'linewidth',3);
end

 

 

d2 = load('C:\Users\Ole\Desktop\Project\SLAM\ourCache\path6.txt');
%data = data(1:100,:)
%d2(:,3) = d2(:,3).*(-1);
% d2(:,1) = d2(:,1).*(-1);
d2_comp=[data(1,:);d2(:,4:6), d2(:,1), d2(:,7:9), d2(:,2), d2(:,10:12),d2(:,3)];

 


for i=1:size(d2,1)-1
    line([d2(i,1),d2(i+1,1)],[d2(i,3),d2(i+1,3)]);
end
hold off

 

figure
hold on

 


for i=1:10
    quiver3(data(i,4),data(i,8),data(i,12),data(i,1),data(i,5),data(i,9),'r')
    quiver3(data(i,4),data(i,8),data(i,12),data(i,2),data(i,6),data(i,10),'g')
    quiver3(data(i,4),data(i,8),data(i,12),data(i,3),data(i,7),data(i,11),'b')
    
    quiver3(d2(i,1),d2(i,2),d2(i,3),d2(i,4),d2(i,7),d2(i,10),'k')
    quiver3(d2(i,1),d2(i,2),d2(i,3),d2(i,5),d2(i,8),d2(i,11),'g')
    quiver3(d2(i,1),d2(i,2),d2(i,3),d2(i,6),d2(i,9),d2(i,12),'b')
end

 


hold off
%%

clc;clear;close all
cd 'C:\Users\Ole\Desktop\Project\SLAM\ourCache'

 

format long
data_3dPoint = load('Q.txt');
data_cam_params = load('cam_params.txt');
data_optimiz = load('optimizing_matrix.txt');
% data_optimiz = sortrows(data_optimiz,2);

% data_optimiz = [max(data_optimiz(:,1))+1, max(data_optimiz(:,2))+1, ...
%     length(data_optimiz(:,3)), 0; data_optimiz];

header = [max(data_optimiz(:,1))+2, max(data_optimiz(:,2))+1, ...
    length(data_optimiz(:,3))];

dlmwrite('optimize2.txt',header)

dlmwrite('optimize2.txt',data_optimiz, '-append')

dlmwrite('optimize2.txt',data_cam_params, '-append')

dlmwrite('optimize2.txt',data_3dPoint, '-append')