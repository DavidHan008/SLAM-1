clc;clear;close all

pik = importdata('3DPoints.txt');
data = load('C:\Users\test\Desktop\skoli2020f\project\data_odometry_gray\dataset\sequences\06\06.txt');


% %%
% 
% thresh = 500
% output = []
% for i = 1:length(pik)
%     if pik(i,1) < thresh && pik(i,2) < thresh && pik(i,3) < thresh
%         output(length(output) + 1,:) = pik(i,:);
%         
%     end
% end

abe = rmoutliers(pik,1);

%%

cnt = 1
for i=1:size(data,1)-1
    hold on
    line([data(i,4),data(i+1,4)],[data(i,8),data(i+1,8)], [data(i,12),data(i+1,12)], 'color', 'red', 'linewidth',3);
    while abe(cnt,4) == i-1
        
        scatter3(abe(:,1),abe(:,2),abe(:,3))
        cnt = cnt+1;
    end
    pause(1)
    
end
hold off






%data = data(1:100,:)



hold off