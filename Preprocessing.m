%%%% Run here
%% make sure you run this section in this file
fs = 250;
trials = 14;
channels = {'C4', 'Cp4', 'F4', 'Fc4', 'Cz', 'Fp2', 'Fz', 'Fcz', 'O2', 'Tp8', ...
    'P8', 'Ft8', 'T8', 'P4', 'F8', 'P3', 'Cp3', 'C3', 'Fc3', 'F3', 'Fp1', ...
    'Pz', 'Cpz', 'Oz', 'O1', 'P7', 'Tp7', 'T7', 'Ft7', 'F7','F8'};

trial_length = 3;
experiment_length = 85;

%% event extraction and reading files Arshak
data = zeros(trials*2, length(channels), experiment_length*fs);

for i=1:trials
    
    %read data
    filepath = 'Dara_org/Arshak/Arshak%d/Raw/Arshak%d.tdms';
    struct = TDMS_getStruct(sprintf(filepath,i,i));
    cell = struct2cell(struct);
    mat = cell2mat(cell(2));
    
    %find start of the data
    trig_diff = diff(mat.Trig.data);
    rising_edge_indices = find(trig_diff);
    
    %second trial has one extra trigger in the beginning
    if i ~= 2
        start = rising_edge_indices(1)+1;
    else
        start = rising_edge_indices(2)+1;
    end
    
    %find second start time this is for the second time in the subject
    meta_filepath = 'Dara_org/Amirreza_Hatami/data/data/Arshak_%d';
    metadata = load(sprintf(meta_filepath,i));
    second_start = ceil(metadata.Start_time_second*fs) + start;
    
    %metadata response time index
    j = 1;
   
    response_time1 = zeros(1,60);
    response_key1 = zeros(1,60);
    
    response_time2 = zeros(1,60);
    response_key2 = zeros(1,60);
    %stored response index
    k = 1;
    current_response_time = 0;
    second_trial = 0;
    
    while j<length(metadata.ResponseTime) && metadata.ResponseTime(j) ~= 0
       
    %if the key is pressed at least (trial length) seconds after the
    %last time, store the response time
        
        %if first run
        if metadata.ResponseTime(j)<metadata.Start_time_second
            
            if metadata.ResponseTime(j)>current_response_time+trial_length
                current_response_time = metadata.ResponseTime(j);
                %store indices of the samples of the starts of the trials
                response_time1(k) = ceil(current_response_time*fs) - trial_length*fs;
                response_key1(k) = metadata.ResponseKey(j-1);
                k = k+1;
            end
        
        

        else
            if (second_trial == 0)
                current_response_time = metadata.Start_time_second;
                k = 1;    
                second_trial = 1;
            end
            
            
            if metadata.ResponseTime(j)>current_response_time+trial_length
                current_response_time = metadata.ResponseTime(j);
                %store indices of the samples of the starts of the trials
                response_time2(k) = ceil((current_response_time- metadata.Start_time_second)*fs)...
                - trial_length*fs;
                response_key2(k) = metadata.ResponseKey(j-1);
                k = k+1;
            end
            
            
        end
        
        j = j+1;
    end
    
    response_time1 = response_time1(response_time1~=0);
    response_time2 = response_time2(response_time2~=0);
    response_key1 = response_key1(response_key1~=0);
    response_key2 = response_key2(response_key2~=0);
    
    %response times in your time
    file = fopen(sprintf('events%d.txt', i*2-1),'w');
    fprintf(file,'%s %s\n','latency','type');
    fprintf(file,'%deee %d\n',[response_time1; response_key1]);
    fclose(file);
    
    file = fopen(sprintf('events%d.txt', i*2),'w');
    fprintf(file,'%s %s\n','latency','type');
    fprintf(file,'%d %d\n',[response_time2; response_key2]);
    fclose(file);
    
end

%% Makoto

%% note. Steps 1 through 6 of Makoto's pipeline were applied by the interactive GUI of EEGLAB
%% Step 7: Apply clean_rawdata() 
% to reject bad channels and correct continuous data using Artifact Subspace Reconstruction (ASR). 
% Note 'availableRAM_GB' is for clean_rawdata1.10. For any newer version, it will cause error.
    
for i = 1:28
    originalEEG = EEG(i);
    EEG(i) = clean_rawdata(EEG(i), 'FlatlineCriterion',5,'ChannelCriterion',0.85,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','off','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7]);
    % Step 8: Interpolate all the removed channels
    EEG(i) = pop_interp(EEG(i), originalEEG.chanlocs, 'spherical');
    
end

%% Step 9: Re-reference the data to average on the collected data in the dataset
for i = 1:28
    EEG(i).nbchan = EEG(i).nbchan+1;
    EEG(i).data(end+1,:) = zeros(1, EEG(i).pnts);
    EEG(i).chanlocs(1,EEG(i).nbchan).labels = 'initialReference';
    EEG(i) = pop_reref(EEG(i), []);
    EEG(i) = pop_select( EEG(i),'nochannel',{'initialReference'}); 
    
end

%% Step 10: ICA

for i = 1:28
   dataRank = sum(eig(cov(double(EEG(i).data'))) > 1E-6); % 1E-6 follows pop_runica() line 531, changed from 1E-7.
    runamica15(EEG(i).data, 'num_chans', EEG(i).nbchan,...
        'outdir', sprintf('amicaResults%d', i),...
        'pcakeep', dataRank, 'num_models', 1,...
        'do_reject', 1, 'numrej', 15, 'rejsig', 3, 'rejint', 1);
    EEG(i).etc.amica  = loadmodout15(sprintf('amicaResults%d', i));
    EEG(i).etc.amica.S = EEG(i).etc.amica.S(1:EEG(i).etc.amica.num_pcs, :); % Weirdly, I saw size(S,1) be larger than rank. This process does not hurt anyway.
    EEG(i).icaweights = EEG(i).etc.amica.W;
    EEG(i).icasphere  = EEG(i).etc.amica.S;
    EEG(i) = eeg_checkset(EEG(i), 'ica');    
end

%% Step 11:	Estimate single equivalent current dipoles
for i = 1:28
    [~,coordinateTransformParameters] = coregister(EEG(i).chanlocs, 'plugins/dipfit/standard_BEM/elec/standard_1005.elc', 'warp', 'auto', 'manual', 'off');
    templateChannelFilePath = 'plugins/dipfit/standard_BEM/elec/standard_1005.elc';
    hdmFilePath             = 'plugins/dipfit/standard_BEM/standard_vol.mat';
    EEG(i) = pop_dipfit_settings( EEG(i), 'hdmfile', hdmFilePath, 'coordformat', 'MNI',...
        'mrifile', 'plugins/dipfit/standard_BEM/standard_mri.mat',...
        'chanfile', templateChannelFilePath, 'coord_transform', coordinateTransformParameters,...
        'chansel', 1:EEG(i).nbchan);
    EEG(i) = pop_multifit(EEG(i), 1:EEG(i).nbchan,'threshold', 100, 'dipplot','off','plotopt',{'normlen' 'on'}); 
    
end

%% Step 12: Search for and estimate symmetrically constrained bilateral dipoles

for i = 1:28
    EEG(i) = fitTwoDipoles(EEG(i), 'LRR', 35);
end

%% Step 13: Run ICLabel (Pion-Tonachini et al., 2019)
for i = 1:28
    EEG(i) = iclabel(EEG(i), 'default'); 
end


%% save
for i = 1:28
    % Save the dataset
    EEG(i) = pop_saveset(EEG(i), 'filename', sprintf('s%d',i), 'filepath', 'data06');
end








%% epoch extraction
for i = 1:28
    filepath = 'D:\\School\\S7\\Foundations__of__Neuroscience\\Project\\data06\\';
    EEG = pop_loadset('filename',sprintf('s%d.set',i),'filepath',filepath);
    eventpath = sprintf('D:\\School\\S7\\Foundations__of__Neuroscience\\Project\\events%d.txt',i);
    EEG = pop_importevent( EEG, 'event',eventpath,'fields',{'latency','type'},'skipline',1,'timeunit',NaN,'align',NaN);
    EEG = pop_epoch( EEG, {  }, [-1  3], 'newname', sprintf('s%d epochs',i), 'epochinfo', 'yes');
    epochs_path = sprintf('D:\\School\\S7\\Foundations__of__Neuroscience\\Project\\epochs\\epochs%d.set',i);
    EEG = pop_newset(EEG, EEG, 2,'savenew',epochs_path,'gui','off');     
    EEG = pop_delset( EEG, [1] );
end


%write into file
for i = 1:28
    if (i ~= 17)
        filepath = 'D:\\School\\S7\\Foundations__of__Neuroscience\\Project\\epochs\\';
        EEG = pop_loadset('filename',sprintf('epochs%d.set',i),'filepath',filepath);
        events = vertcat(EEG.epoch.eventtype);
        filename = sprintf('D:\\School\\S7\\Foundations__of__Neuroscience\\Project\\epochs\\mat\\epochs_labels%d',i);
        save(filename, 'events');    
    end

end




