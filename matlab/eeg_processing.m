clc
clearvars
close all

%% ===================== CONFIGURATION =====================
% File path:
fpath = 'C:\Users\elisa\...'; %% add your own path
addpath(genpath(fpath));

fs          = 256;      % Sampling frequency
imag_offset = 4;        % Imagery starts 4 s after trial onset
imag_dur    = 3;        % Imagery duration (s)
thr         = 60;       % Threshold for artifact removal

% Bandpass filter (EEG)
order   = 5;
fband   = [2 45];       % Hz
[b, a]  = butter(order, fband/(fs/2));

% Frequency bands for bandpower features
fb_width    = [2, 4, 6];     % width of frequency bands in Hz
fb_overlap  = 1;             % overlap of frequency bands in Hz
f1          = 8;             % lower limit (Hz)
f2          = 30;            % upper limit (Hz)

freq_bands  = build_freq_bands(f1, f2, fb_width, fb_overlap);
n_channels  = 3;

% ---- Plotting flags per block ----
PLOT.train1 = true;
PLOT.train2 = false;
PLOT.train3 = false;
PLOT.test1  = false;
PLOT.test2  = false;

%% ===================== LOAD DATA ==========================
data_train = load('B04T.mat');
data_test  = load('B04E.mat');

%% ===================== PREPROCESS TRAINING BLOCKS =========

[train_cell_1, labels_1] = preprocess_block( ...
    data_train.data{1,1}, fs, b, a, thr, imag_offset, imag_dur, PLOT.train1);

[train_cell_2, labels_2] = preprocess_block( ...
    data_train.data{1,2}, fs, b, a, thr, imag_offset, imag_dur, PLOT.train2);

[train_cell_3, labels_3] = preprocess_block( ...
    data_train.data{1,3}, fs, b, a, thr, imag_offset, imag_dur, PLOT.train3);

%% ===================== PREPROCESS TEST BLOCKS =============

[test_cell_1, labels_1t] = preprocess_block( ...
    data_test.data{1,1}, fs, b, a, thr, imag_offset, imag_dur, PLOT.test1);

[test_cell_2, labels_2t] = preprocess_block( ...
    data_test.data{1,2}, fs, b, a, thr, imag_offset, imag_dur, PLOT.test2);

%% ===================== CONCATENATE SEQUENCES ==============

train_array   = [train_cell_1, train_cell_2, train_cell_3];
test_array    = [test_cell_1,  test_cell_2];

labels_train  = [labels_1; labels_2; labels_3];
labels_test   = [labels_1t; labels_2t];

training_set  = struct('eeg_sequences', train_array, 'label', labels_train);
test_set      = struct('eeg_sequences', test_array,  'label', labels_test);

save('training_set_new', 'training_set');
save('test_set_new',    'test_set');

%% ===================== FEATURE EXTRACTION: TRAIN ===========
[train_bp, col_names] = extract_bp_features( ...
    train_array, fs, freq_bands, n_channels);

Training_features = array2table(train_bp, 'VariableNames', col_names);

% put label after the last feature column
last_feat_name = col_names{end};
train_final = addvars(Training_features, labels_train, ...
    'After', last_feat_name, 'NewVariableName', 'label');

writetable(train_final, 'trainset_feat_new.csv');

%% ===================== FEATURE EXTRACTION: TEST ============
% Reuse the same freq_bands and column order
[test_bp, ~] = extract_bp_features( ...
    test_array, fs, freq_bands, n_channels);

Test_features = array2table(test_bp, 'VariableNames', col_names);

test_final = addvars(Test_features, labels_test, ...
    'After', last_feat_name, 'NewVariableName', 'label');

writetable(test_final, 'testset_feat_new.csv');



%% ==========================================================
% =============== LOCAL FUNCTIONS BELOW =====================
%% ==========================================================

function [seq_cell, labels] = preprocess_block(block, fs, b, a, thr, imag_offset, imag_dur, do_plot)
%PREPROCESS_BLOCK  Clean EOG artifacts, bandpass filter, epoch trials.
%
%   block: struct with fields X, trial, y, artifacts
%   Returns:
%       seq_cell : 1xN cell array, each cell = [samples x 3 channels]
%       labels   : Nx1 vector (0/1) after removing artifact trials

    signals    = block.X;          % [N x 6] (3 EEG + 3 EOG)
    eeg_data   = signals(:,1:3);
    eog_data   = signals(:,4:6);

    % --- Estimate regression coefficients on eye-artifact period (3-4 min) ---
    t_artifacts = (3*60*fs):(4*60*fs);  % indices for 1 minute with eye artifacts
    beta        = compute_regression_coeffs(eeg_data, eog_data, t_artifacts);

    % --- Remove EOG artifacts ---
    eeg_clean   = eeg_data - eog_data * beta(2:end,:);

    % --- Bandpass filtering ---
    eeg_filt    = filtfilt(b, a, eeg_clean);

    % --- Trial starts and labels (remove artifact trials) ---
    start_trial = block.trial;
    labels      = block.y;
    idx_art     = find(block.artifacts);

    start_trial(idx_art) = [];
    labels(idx_art)      = [];

    % Convert labels from [1,2] to [0,1]
    labels = labels(:) - 1;

    % --- Imagery window indices (start after imag_offset seconds) ---
    im_on  = start_trial + imag_offset*fs;             % start indices
    im_len = imag_dur * fs;                            % #samples in imagery window

    % --- Threshold-based artifact cleaning within imagery periods ---
    eeg_filt = remove_threshold_artifacts(eeg_filt, im_on, fs, imag_dur, thr);

    % --- Imagery end indices (keep im_len samples) ---
    im_off = im_on + im_len - 1;

    % --- Build cell array of sequences ---
    seq_cell = create_sequences(eeg_filt, im_on, im_off);

    % --- Optional plotting: example epoch and PSD ---
    if do_plot && ~isempty(seq_cell)
        eeg_epoch = seq_cell{1};                % first epoch
        t_epoch   = (0:size(eeg_epoch,1)-1)/fs; % time axis

        figure;
        subplot(2,1,1);
        plot(t_epoch, eeg_epoch);
        xlabel('Time (s)');
        ylabel('Amplitude (\muV)');
        title('Example epoch after preprocessing');
        legend({'C3','Cz','C4'}, 'Location','best');

        % PSD of channel 1
        [pxx,f] = pwelch(eeg_epoch(:,1),[],[],[],fs);
        subplot(2,1,2);
        plot(f, pxx);
        xlim([0 50]);
        xlabel('Frequency (Hz)');
        ylabel('Power / Hz (\muV^2/Hz)');
        title('PSD of channel 1 (epoch 1)');
    end
end


function beta = compute_regression_coeffs(eeg_data, eog_data, t_artifacts)
%COMPUTE_REGRESSION_COEFFS  Linear regression of EEG on EOG during artifacts.
%
%   eeg_data : [N x 3]
%   eog_data : [N x 3]
%   t_artifacts : indices to use for regression
%
%   Returns beta: [4 x 3] (intercept + 3 EOG channels -> 3 EEG channels)

    X = [ones(numel(t_artifacts),1), eog_data(t_artifacts,:)];  % [N_art x 4]
    Y = eeg_data(t_artifacts,:);                                % [N_art x 3]

    beta = (X' * X) \ (X' * Y);  % [4 x 3]
end


function eeg_data = remove_threshold_artifacts(eeg_data, im_on, fs, imag_dur, thr)
%REMOVE_THRESHOLD_ARTIFACTS
%   Sets values above `thr` in each imagery period to zero, and keeps the
%   same logic of iterating over channels and imagery windows as in the
%   original script. (It does NOT change im_on length, just zeros out.)

    n_channels  = size(eeg_data, 2);
    im_len      = imag_dur * fs;  % samples in imagery window

    for ch = 1:n_channels
        for j = 1:numel(im_on)
            idx = im_on(j):(im_on(j) + im_len);   % imagery interval
            high_idx = find(eeg_data(idx, ch) > thr);

            if ~isempty(high_idx)
                % Zero out all channels at those time points
                eeg_data(idx(high_idx), :) = 0;
            end
        end
    end
end


function seq_cell = create_sequences(eeg_data, im_on, im_off)
%CREATE_SEQUENCES  Cut continuous EEG into trial-wise segments.

    n_trials = numel(im_on);
    seq_cell = cell(1, n_trials);

    for k = 1:n_trials
        seq_cell{k} = eeg_data(im_on(k):im_off(k), :);
    end
end


function freq_bands = build_freq_bands(f1, f2, fb_width, fb_overlap)
%BUILD_FREQ_BANDS  Construct [low high] frequency bands for bandpower.

    freq_bands = [];

    for i = 1:length(fb_width)
        edges = f1:fb_overlap:(f2 - fb_width(i));
        freq_bands = [freq_bands;
                      [edges', edges' + fb_width(i)] ];
    end
end


function [bp_features, col_names] = extract_bp_features(seq_cell, fs, freq_bands, n_channels)
%EXTRACT_BP_FEATURES  Bandpower features for each epoch, channel, band.
%
%   seq_cell   : 1xN cell array, each cell is [samples x n_channels]
%   fs         : sampling rate
%   freq_bands : [nBands x 2] matrix of [low high] limits
%   n_channels : number of EEG channels (e.g. 3)
%
%   Returns:
%       bp_features : [N x (n_channels*nBands)]
%       col_names   : cell array of names "f1-f2_Hz_chX"

    n_samples     = numel(seq_cell);
    n_freq_bands  = size(freq_bands, 1);
    n_features    = n_channels * n_freq_bands;

    bp_features   = zeros(n_samples, n_features);
    col_names     = cell(1, n_features);

    for eeg_idx = 1:n_samples
        eeg_sample = seq_cell{eeg_idx};    % [T x n_channels]

        for ch = 1:n_channels
            for fb = 1:n_freq_bands
                feat_idx = (ch-1)*n_freq_bands + fb;

                power_signal = bandpower(eeg_sample(:, ch), fs, freq_bands(fb,:));
                bp_features(eeg_idx, feat_idx) = power_signal;

                % Build feature names once (first epoch)
                if eeg_idx == 1
                    col_names{feat_idx} = sprintf('%d-%d_Hz_ch%d', ...
                        freq_bands(fb,1), freq_bands(fb,2), ch);
                end
            end
        end
    end
end
