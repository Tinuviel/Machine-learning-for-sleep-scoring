%Changes the data to standard deviation 1 and zero-mean
%Doesn't help, still has to multiply the data to get anything

r =  matfile('rat3_all.mat', 'Writable', true);

eeg = r.EEGandEMG;
eeg = eeg - mean(eeg);
mean(eeg);
std(eeg);
eeg2 = eeg./std(eeg);
std(eeg2);
r.EEGandEMG = eeg2;


%r.EEGandEMG = eeg;
% matrixA = [0 1 2; 2 7 4; 5 1 7]
% mean(matrixA)
% matrixB = matrixA-mean(matrixA)
% mean(matrixB)
% std(matrixA)
% matrixB = matrixA./std(matrixA)
% std(matrixB)
