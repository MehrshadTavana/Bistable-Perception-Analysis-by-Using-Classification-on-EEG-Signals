%% Neuroscience Project
%%
clc; close all; clear all; 
load("epochs.mat");
load("labels.mat");
load('locs.mat');
name = cellstr(name);
locsX = str2num(c2).*cos(deg2rad(str2num(c1)));
locsY = str2num(c2).*sin(deg2rad(str2num(c1)));
fs = 250;
labels = str2num(labels);
labels(labels == 101) = 1;
labels(labels == 113) = 2;

% Train Test Split and Baseline normalization
indices = randperm(size(data, 3));
data_train = data(:, :, indices(1:208));
y_train = labels(indices(1:208));
data_test = data(:, :, indices(209:end));
y_test = labels(indices(209:end));


% Plotting one train trial 
X = data_train(:,:,randi(208));  
offset = max(abs(X(:)));
disp_eeg(X, offset, fs, name);
title('The EEG Signal', 'Interpreter', 'latex', 'FontSize', 10)

% Calculating channels that are different in two classes 
class1_wav = zeros(30, 71, 1000);
class2_wav = zeros(size(class1_wav));
for i = 1:size(data_train, 3)
    for j = 1:size(data_train, 1)
        temp(1 ,:, :) = abs(cwt(data_train(j, :, i), fs));
        if (y_train(i) == 1)
            class1_wav(j, :, :) = class1_wav(j, :, :) + temp;
        else
            class2_wav(j, :, :) = class2_wav(j, :, :) + temp;
        end
    end
end
class1_wav = class1_wav/length(find(y_train==1));
class2_wav = class2_wav/length(find(y_train==2));
wav_diff = abs(class1_wav-class2_wav);
w = sum(wav_diff, 2);
w = sum(w, 3);
figure
plottopomap(locsX,locsY,name,w);
[~, desired_channels] = maxk(w, 5);
%%
t = 1/fs:1/fs:4;
[~, f] = cwt(data_train(desired_channels(1), :, 50) ,fs);
trial = data_train(:, :, 1);
cwt(trial(desired_channels(1), :), fs);
%%
% Extracting time-frequency feaetures from desired channels
features_train = zeros(size(data_train, 3), 56*5);
%features_train = zeros(size(data_train, 3), 71*1000*5);
for i = 1:size(data_train, 3)
    trial = data_train(:, :, i);
    for j = 1:length(desired_channels)
        wav = abs(cwt(trial(desired_channels(j), :), fs));
        wav_avgpool = avgpool(dlarray(wav,'SS'),[10 125],'stride',[10 125]);
        %c = 1;
        %feat = zeros(1, 49);
        %for k = 1:10:(length(f)-10)
        %    for l = 1:125:(length(t)-125)
        %        feat(c) = mean(wav(k:(k+10), l:(l+125)), "all");
        %        c = c+1;
        %    end
        %end
        %features_train(i, (j-1)*49+1:j*49) = feat; 
        features_train(i, (j-1)*56+1:(j)*56) = wav_avgpool(:);
        %features_train(i, (j-1)*71*1000+1:(j)*71*1000) = wav(:);
    end
end
%% 
% Normalizing features 
[features_train_norm, PS] = mapstd(features_train',0,1); 
features_train_norm = features_train_norm';
%features_norm_test = mapstd('apply',features_tot(121:160,:)',PS);
%features_norm_test = features_norm_test';
%%
% PCA
[loadings,~,~,~,explained,~] = pca(features_train_norm);
features_train_norm_pca = features_train_norm*loadings(:,1:10);
%%
% SVM
X = features_train_norm_pca;
cvp = cvpartition(y_train, 'KFold', 5);
mdl = fitcsvm(X,y_train, 'CVPartition', cvp);
prediction = kfoldPredict(mdl);
confusionmat(y_train, prediction)
y_train_conf = y_train;
y_train_conf(y_train_conf==2) = 0;
prediction_conf = prediction;
prediction_conf(prediction_conf==2) = 0;
plotconfusion(y_train_conf', prediction_conf');

