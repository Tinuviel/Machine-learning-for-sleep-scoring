%Making extra samples. Goes extreamly slow right now, should optimize by initializing the matrixes first and THEN changing the .mat-file
%But I already compiled so I'll make that change another day. Maybe we should have larger steps than 1 ms, 
%since the samples will be very much alike now
clear all, close all;
r =  matfile('rat3_all.mat', 'Writable', true);
eeg = r.EEGandEMG;
sleep = [0 0 1 0 0 0].';
REM = [0 0 0 0 1 0].';

test122 = zeros(4000, 2);
test122(:, 1) = eeg(:, 122);
test122(:, 2) = eeg(:, 123);
test19274 = zeros(4000, 41);
for i = 1:41
    test19274(:, i) = eeg(:, 19273+i);
end
test19274 = reshape(test19274, [4000*41, 1]);
test122 = reshape(test122, [8000,1]);

for c = 1:4000
    r.EEGandEMG = [r.EEGandEMG test122(c:c+3999, 1)];
    r.labels = [r.labels sleep];
    
end

for c = 1:41*4000
    r.EEGandEMG = [r.EEGandEMG test19274(c:c+3999, 1)];
    r.labels = [r.labels sleep];
end

test9099 = zeros(4000, 20);
for i = 1:20
    test9099(:, i) = eeg(:, 9099+i);
end
test9099 = reshape(test9099, [4000*20, 1]);
for c = 1:20*4000
    r.EEGandEMG = [r.EEGandEMG test9099(c:c+3999, 1)];
    r.labels = [r.labels REM];
end

test6671 = zeros(4000, 8);
for i = 1:8
    test6671(:, i) = eeg(:, 6671+i);
end
test6671 = reshape(test6671, [4000*8, 1]);
for c = 1:8*4000
    r.EEGandEMG = [r.EEGandEMG test6671(c:c+3999, 1)];
    r.labels = [r.labels REM];
end


%Examples of continuous samples to make new samples from
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


%Test data to make sure I got the matrix dimensions right
%test2 = [zeros(4, 4)]
%test3 = [1,2,3,4;5,6,7,8]
%test3 = reshape(test3, [8,1])
%test2(1, :) = test3(1, :)
%for c = 1:4
%    test2(:, c) = test3(c:c+3, 1)
%end
