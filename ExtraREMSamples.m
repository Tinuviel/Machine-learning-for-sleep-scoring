r =  matfile('rat3_all.mat', 'Writable', true);
EEGandEMG = r.EEGandEMG;
EEG = EEGandEMG(1:2000, :);
EMG = EEGandEMG(2001:4000, :);

sleep = [0 0 1 0 0 0].';
REM = [0 0 0 0 1 0].';
u = 20*2000;
%Allocate matixes
test5 = zeros(4000, 4000);
test9099EEG = zeros(2000, 20);

test9099EMG = zeros(2000, 20);
sleepLabels9099 = zeros(6, 4000);

%Initializing from EEGandEMG
 for i = 1:20
     test9099EEG(:, i) = EEG(:, 9098+i);
     test9099EMG(:, i) = EMG(:, 9098+i);
 end
size(test9099EEG)
size(test9099EMG)

% 
test9099EEG = reshape(test9099EEG, [u, 1]);
test9099EMG = reshape(test9099EMG, [u, 1]);
size(test9099EEG)
size(test9099EMG)
 for c = 1:2000
     test5(:, c) = [test9099EEG(c*19:c*19+1999, 1); test9099EMG(c*19:c*19+1999, 1)];
     sleepLabels9099(:, c) = REM;   
 end
 size(test5)
 r.EEGandEMG = [r.EEGandEMG test5];
 r.labels = [r.labels sleepLabels9099];