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
clc,clear; close all
data = load('C:\VisionPyhton\Project\dataset\sequences\06\poses.txt');
figure
hold on


for i=1:size(data,1)-1
    line([data(i,4),data(i+1,4)],[data(i,12),data(i+1,12)], 'color', 'red', 'linewidth',3);
end



d2 = load('C:\VisionPyhton\Project\SLAM\path6.txt');
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
