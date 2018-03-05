load rat3_all.mat
eeg = EEGandEMG;
[b, a] = butter(7, 0.3, 'high');
eeg = filter(b, a, eeg);
eeg = fft(eeg);

EEGandEMG = eeg;