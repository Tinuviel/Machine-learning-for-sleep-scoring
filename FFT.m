clear all, close all;
r =  matfile('rat3_all.mat', 'Writable', true);
eeg = r.EEGandEMG;
[b, a] = butter(7, 0.3, 'high');
eeg = filter(b, a, eeg);
eeg = fft(eeg);

r.EEGandEMG = eeg;
