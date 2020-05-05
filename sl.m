clc;clear
%%
data = load('C:\Users\test\Desktop\skoli2020f\project\KITTI_sequence_1\poses.txt');
figure
hold on


for i=1:size(data,1)-1
    line([data(i,4),data(i+1,4)],[data(i,12),data(i+1,12)], 'color', 'red', 'linewidth',3);
end

d2 = load('C:\Users\test\Desktop\skoli2020f\project\SLAM\path1.txt');
%data = data(1:100,:)
% d2(:,2) = d2(:,2).*(-1);
% d2(:,1) = d2(:,1).*(-1);


for i=1:size(d2,1)-1
    line([d2(i,1),d2(i+1,1)],[d2(i,2),d2(i+1,2)]);
end
hold off

%%
clc,clear;
data = load('C:\VisionPyhton\Project\dataset\sequences\06\poses.txt');
figure
hold on


for i=1:size(data,1)-1
    line([data(i,4),data(i+1,4)],[data(i,12),data(i+1,12)], 'color', 'red', 'linewidth',3);
end


d2 = load('C:\VisionPyhton\Project\SLAM\path6.txt');
%data = data(1:100,:)
% d2(:,2) = d2(:,2).*(-1);
% d2(:,1) = d2(:,1).*(-1);


for i=1:size(d2,1)-1
    line([d2(i,1),d2(i+1,1)],[d2(i,3),d2(i+1,3)]);
end
hold off