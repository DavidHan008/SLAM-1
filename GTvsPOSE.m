clc,clear, close all;

data_cam_params = load('cam_params.txt'); % Denne skal ændres til den rigtige fil
Ground_truth_raw = load('C:\VisionPyhton\Project\dataset\sequences\06\poses.txt');

GT_X=Ground_truth_raw(:,4);
GT_Z=Ground_truth_raw(:,12);


abe=reshape(data_cam_params,9,9900/9)';

X = cumsum(abe(:,4));
Z = cumsum(abe(:,6));


figure
hold on
plot(GT_X,GT_Z);
plot(X,Z)
hold off
legend('GT','SHIT')