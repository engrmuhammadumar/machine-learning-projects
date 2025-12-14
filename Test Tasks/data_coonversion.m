% Load the dataset
load('E:\CP Dataset\[20200428] PUMP DATA\VIbration_ALL\impeller_crack_vibration_ALL.mat');

% Assuming your data is stored in the variable 'concatenated_rows'
[num_signals, signal_length] = size(concatenated_rows); % Get dimensions of the dataset

% Sampling parameters
Fs = 25600; % Sampling rate 25.6 kHz
nfft = 1024; % Number of FFT points

% Add the EWT toolbox to your MATLAB path (correct path to the EWT toolbox)
addpath(genpath('C:\Users\Muhammad Umar\Documents\MATLAB\EWT'));  % Adjust the path if necessary

% Parameters for EWT1D (Empirical Wavelet Transform)
params.globtrend = 'none';          % No global trend removal
params.degree = 6;                  % Degree of polynomial subtraction (set as needed)
params.reg = 'none';                % No regularization
params.detect = 'locmax';           % Detection method: local maxima
params.log = 0;                     % Set log to 0 (no logarithmic scaling)
params.typeDetect = 'locmax';       % Method for boundary detection (local maxima)
params.N = 5;                       % Number of frequency bands (5 bands)

% Loop through each signal (row)
for signal_idx = 1:num_signals
    % Get the current signal
    signal = concatenated_rows(signal_idx, :);
    
    % Apply Welch Power Spectrum Estimation
    window = blackman(256); % Blackman window for Welch's method
    noverlap = length(window)/2; % 50% overlap
    [pxx, f] = pwelch(signal, window, noverlap, nfft, Fs);

    % Plot Welch Power Spectrum and save as an image
    figure;
    plot(f, 10*log10(pxx));
    title(['Welch Power Spectral Density for Signal ', num2str(signal_idx)]);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    grid on;

    % Create folder for saving results if it doesn't exist
    folder_name = ['Signal_', num2str(signal_idx)];
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    saveas(gcf, fullfile(folder_name, ['Welch_PSD_Signal_', num2str(signal_idx), '.png']));
    close(gcf);

    % Apply EWT to decompose the signal
    [ewt_imfs, Boundaries] = EWT1D(signal, params); % Perform EWT decomposition with parameters
    
    % Plot and save each IMF
    for i = 1:length(ewt_imfs)
        figure;
        plot(ewt_imfs{i});
        title(['IMF ', num2str(i), ' for Signal ', num2str(signal_idx)]);
        saveas(gcf, fullfile(folder_name, ['IMF_', num2str(i), '_Signal_', num2str(signal_idx), '.png']));
        close(gcf);
    end

    % Apply SVD for denoising
    for i = 1:length(ewt_imfs)
        imf = ewt_imfs{i};
        L = floor(length(imf)/2); % Hankel matrix dimension
        H = hankel(imf(1:L), imf(L:end)); % Hankel matrix
        
        % SVD decomposition
        [U, S, V] = svd(H);
        
        % Reconstruct using dominant singular values
        S_denoised = S;
        S_denoised(S<max(S(:))*0.1) = 0; % Thresholding
        H_denoised = U * S_denoised * V';
        ewt_imfs{i} = H_denoised(:,1); % Update denoised IMF
    end

    % Apply Hilbert Transform to extract time-frequency features
    for i = 1:length(ewt_imfs)
        analytic_signal = hilbert(ewt_imfs{i});
        inst_amplitude = abs(analytic_signal);
        inst_phase = unwrap(angle(analytic_signal));
        inst_freq = diff(inst_phase)/(2*pi)*Fs; % Instantaneous frequency
        
        % Plot Instantaneous Frequency and save as an image
        figure;
        plot(inst_freq);
        title(['Instantaneous Frequency of IMF ', num2str(i), ' for Signal ', num2str(signal_idx)]);
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        saveas(gcf, fullfile(folder_name, ['Inst_Frequency_IMF_', num2str(i), '_Signal_', num2str(signal_idx), '.png']));
        close(gcf);
    end
end
