%%%% Make sure to read the new updates in readme file this is the main file that you have to run
%% have a run here in the main code from the main, also do not change any part of the code here
load("epochs.mat");
load("labels.mat");
load('locs.mat');
name = cellstr(name);
locsX = str2num(c2).*cos(deg2rad(str2num(c1)));
locsY = str2num(c2).*sin(deg2rad(str2num(c1)));
fs = 250;
labels = str2num(labels);
labels(labels == 101) = 0;
labels(labels == 113) = 1;

%% for running have here not anywhere else dont forget to do that here just here make sure you have them and test the database and push to db
indices = randperm(size(data, 3));
data_train = data(:, :, indices(1:208));
y_train = labels(indices(1:208));
data_test = data(:, :, indices(209:end));
y_test = labels(indices(209:end));



% Plotting one train trial (functions removed) this plots can be used in your report for the paper in the pdf format run it here
X = data_train(:,:,randi(208));  
offset = max(abs(X(:)));
disp_eeg(X, offset, fs, name);
title('The EEG Signal', 'Interpreter', 'latex', 'FontSize', 10)


class0_wav = zeros(30, 71, 1000);
class1_wav = zeros(size(class0_wav));
for i = 1:size(data_train, 3)
    for j = 1:size(data_train, 1)
        temp(1 ,:, :) = abs(cwt(data_train(j, :, i), fs));
        if (y_train(i) == 1)
            class0_wav(j, :, :) = class0_wav(j, :, :) + temp;
        else
            class1_wav(j, :, :) = class1_wav(j, :, :) + temp;
        end
    end
end
class0_wav = class0_wav/length(find(y_train==0));
class0_wav = class0_wav/length(find(y_train==1));
wav_diff = abs(class0_wav-class1_wav);

w = sum(wav_diff, 3);
w = sum(w, 3);
figure
%%
plottopomap(locsX,locsY,name,w);
[~, desired_channels] = maxk(w, 15);

%train dataset wavelet transform

cnn_input = zeros([37 1000 length(desired_channels) 208]);
for i = 1:size(data_train, 3)
    trial = data_train(:, :, i);
    for j = 1:length(desired_channels)
        wav = abs(cwt(trial(desired_channels(j), :), fs));
        cnn_input(:, :, j, i) = wav(35:71,:);
    end
end


%%
% test dataset wavelet transform

cnn_input_test = zeros([37 1000 length(desired_channels) 53]);
for i = 1:size(data_test, 3)
    trial = data_train(:, :, i);
    for j = 1:length(desired_channels)
        wav = abs(cwt(trial(desired_channels(j), :), fs));
        cnn_input_test(:, :, j, i) = wav(35:71,:);
    end
end

y_cat = categorical(y_train);
y_test_cat = categorical(y_test);



%%
% layers
layers = [
    imageInputLayer([37 1000 length(desired_channels)])
    
    convolution2dLayer([10 25],4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer([5 5],8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer([3 3],8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer
        
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];



%%
%training options

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'ValidationData',{cnn_input_test,y_test_cat}, ...
    'Plots','training-progress');

%%
net = trainNetwork(cnn_input,y_cat,layers,options);

%%
analyzeNetwork(net)
%%

YPred = classify(net,cnn_input_test);
acc = sum(YPred == y_test_cat)./numel(y_test_cat)





%% ensemble method with concatenated features

opts.fs = 250;
features = {'bpd','bpt', 'bpa', 'bpb', 'bpg', 'rba', 'min', 'max', 'md', 'var',...
    'am', 'sd', 'le', 'sh', 'me', 'n2d', 'n1d', 'skew', 'ha', 'hc'};


%%
%training data feature extraction

fc = 71;
concatenated_input = zeros([208 (1000*fc+length(features))*length(desired_channels)]);
for i = 1:size(data_train, 3)
    trial = data_train(:, :, i);
    for j = 1:length(desired_channels)
        wav = abs(cwt(trial(desired_channels(j), :), fs));
        concatenated_input(i,(j-1)*1000*fc+1:(j)*1000*fc) = reshape(wav.',1,[]);
        
        for k = 1:length(features)
            concatenated_input(i,1000*fc*length(desired_channels)+length(features)*(j-1)+k)= ...
            jfeeg(char(features(k)), trial(desired_channels(j), :), opts);
        end
    end
end


%%
%testing data feature extraction

concatenated_input_test = zeros([53 (1000*fc+length(features))*length(desired_channels)]);
for i = 1:size(data_test, 3)
    trial = data_test(:, :, i);
    for j = 1:length(desired_channels)
        wav = abs(cwt(trial(desired_channels(j), :), fs));
        concatenated_input_test(i,(j-1)*1000*fc+1:(j)*1000*fc) = reshape(wav.',1,[]);
        
        for k = 1:length(features)
            concatenated_input_test(i,1000*fc*length(desired_channels)+length(features)*(j-1)+k)= ...
            jfeeg(char(features(k)), trial(desired_channels(j), :), opts);
        end
    end
end

%%
[coeff,scoreTrain,~,~,explained,mu] = pca(concatenated_input);



%%
components_number = 200;
Mdl1 = fitcensemble(scoreTrain(:,1:components_number),y_train);


x_test = (concatenated_input_test-mu)*coeff(:,1:components_number);
YTest_predicted = predict(Mdl1,x_test);

acc = sum(YTest_predicted == y_test)./numel(y_test)
cm = plotconfusion(y_test', YTest_predicted');
cm = plotconfusion(y_test', YTest_predicted');









