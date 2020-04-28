clc;clear;close all

data = load('C:\Users\test\Desktop\skoli2020f\project\data_odometry_gray\dataset\sequences\06\06.txt');

hold on

%data = data(1:100,:)

for i=1:size(data,1)-1
    line([data(i,4),data(i+1,4)],[data(i,12),data(i+1,12)], 'color', 'red', 'linewidth',3);
end

d2 = load('C:\Users\test\Desktop\skoli2020f\project\SLAM\path.txt');
%data = data(1:100,:)
d2(:,2) = d2(:,2).*(-1);

for i=1:size(d2,1)-1
    line([d2(i,1),d2(i+1,1)],[d2(i,2),d2(i+1,2)]);
end
hold off