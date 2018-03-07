%Uncomment the section you want to run, makes rat3_all.mat 39800 samples instead of 19800.
%Trashed the idea of just moving 1ms, matlab lagged to much when I tried to input more samples
%Each consecutive time of S/P gives 4k samples now, moving the timeframe n-1 ms for each new sample,
%where n = amount of consecutive samples
clear all, close all;
r =  matfile('rat3_all.mat', 'Writable', true);
eeg = r.EEGandEMG;
sleep = [0 0 1 0 0 0].';
REM = [0 0 0 0 1 0].';
%% WORKING
% test2 = zeros(4000, 4000);
% test122 = zeros(4000, 2);
% sleepLabels122 = zeros(6, 4000);
% size(sleepLabels122)
% test122(:, 1) = eeg(:, 122);
% test122(:, 2) = eeg(:, 123);
% 
% 
% test122 = reshape(test122, [8000,1]);
% for c = 1:4000
%    test2(:, c) = test122(c:c+3999, 1);
%    sleepLabels122(:, c) = sleep;    
% end
% r.EEGandEMG = [r.EEGandEMG test2];
% r.labels = [r.labels sleepLabels122];
%% WORKING
% u = 41*4000;
% %Allocate matixes
% test3 = zeros(4000, 4000);
% test19274 = zeros(4000, 41);
% sleepLabels19274 = zeros(6, 4000);
% 
% %Initializing from EEGandEMG
% for i = 1:41
%     test19274(:, i) = eeg(:, 19273+i);
% end
% 
% test19274 = reshape(test19274, [u, 1]);
% size(test19274)
% for c = 1:4000
%     test3(:, c) = test19274(c*16:c*16+3999, 1);
%     sleepLabels19274(:, c) = sleep;   
% end
% r.EEGandEMG = [r.EEGandEMG test3];
% r.labels = [r.labels sleepLabels19274];
%% WORKING
% test4 = zeros(4000, 4000);
% test2799 = zeros(4000, 2);
% sleepLabels2799 = zeros(6, 4000);
% size(sleepLabels2799)
% test2799(:, 1) = eeg(:, 2799);
% test2799(:, 2) = eeg(:, 2800);
% 
% 
% test2799 = reshape(test2799, [8000,1]);
% for c = 1:4000
%     test4(:, c) = test2799(c:c+3999, 1);
%     sleepLabels2799(:, c) = REM;    
% end
% r.EEGandEMG = [r.EEGandEMG test4];
% r.labels = [r.labels sleepLabels2799];
%% WORKING
% 
% u = 20*4000;
% %Allocate matixes
% test5 = zeros(4000, 4000);
% test9099 = zeros(4000, 20);
% sleepLabels9099 = zeros(6, 4000);

%Initializing from EEGandEMG
% for i = 1:20
%     test9099(:, i) = eeg(:, 9098+i);
% end
% 
% test9099 = reshape(test9099, [u, 1]);
% size(test9099)
% for c = 1:4000
%     test5(:, c) = test9099(c*19:c*19+3999, 1);
%     sleepLabels9099(:, c) = REM;   
% end
% r.EEGandEMG = [r.EEGandEMG test5];
% r.labels = [r.labels sleepLabels9099];

%% WORKING
% 
% u = 8*4000;
% %Allocate matixes
% test6 = zeros(4000, 4000);
% test6671 = zeros(4000, 8);
% sleepLabels6671 = zeros(6, 4000);
% 
% %Initializing from EEGandEMG
% for i = 1:8
%     test6671(:, i) = eeg(:, 6670+i);
% end
% 
% test6671 = reshape(test6671, [u, 1]);
% size(test6671)
% for c = 1:4000
%     test6(:, c) = test6671(c*7:c*7+3999, 1);
%     sleepLabels6671(:, c) = REM;   
% end
% r.EEGandEMG = [r.EEGandEMG test6];
% r.labels = [r.labels sleepLabels6671];

%% SAMPLES

%2799-2800 P
%3004-3006 P
%5574-5576 P
%5578-5579 P
%6671-6679 P
%9099-9119 P

%19274-19315 S
%19358-19362 S
%19364-19464 S
%19525-19527 S
%19689-19591 S
%19595-19597 S
%19602-19604 S
%19654-19657 S
%19696-19779 S

%eeg = [eeg newSamples];
%size(eeg)
%% TEST
%test2 = [zeros(4, 17)]
%test3 = [1,2,3,4;5,6,7,8;9, 10, 11, 12; 13, 14, 15, 16; 17, 18, 19, 20]
%test3 = reshape(test3, [20,1])
%for c = 1:17
%    test2(:, c) = test3(c:c+3, 1)
%end
