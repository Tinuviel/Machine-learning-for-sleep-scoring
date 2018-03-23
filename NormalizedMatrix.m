r =  matfile('rat3_all.mat', 'Writable', true);
EEGandEMG = r.EEGandEMG;
mean(EEGandEMG(:))
size(EEGandEMG)
EEG = EEGandEMG(1:2000, :);
EMG = EEGandEMG(2001:4000, :);

EEG = EEG - mean(EEG(:));
EEG = EEG./std(EEG(:));
mean(EEG(:))
std(EEG(:))

EMG = EMG - mean(EMG(:));
EMG = EMG./std(EMG(:));
mean(EMG(:))
std(EMG(:))

EEGandEMG = [EEG; EMG];
size(EEGandEMG)
r.EEGandEMG = EEGandEMG;