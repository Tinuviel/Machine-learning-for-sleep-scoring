r =  matfile('rat3_allextraSamp.mat', 'Writable', true);
EEGandEMG = r.EEGandEMG;
EEG = EEGandEMG(1:2000, :);
EMG = EEGandEMG(2001:4000, :);
fftEEG= fft(EEG);
fftEMG = fft(EMG);
fftEEGandEMG = [fftEEG; fftEMG];
r.fftEEGandEMG = fftEEGandEMG
