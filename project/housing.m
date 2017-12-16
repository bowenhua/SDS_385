clear;
%% Read data
load('data.mat')

% Add a column of ones
trainX(:, 125) = 1;
testX(:, 125) = 1;
trainY = trainY';

M = size(trainX, 1);
N = size(trainX, 2);

K = 20;

%% Obtain big-M's

% Specify an upper bound on the obj function of the best subset problem
UB = 6.5; 
bigM = zeros(N, 1);
for idx = 1 : N
    bigM(idx) = generateBigM(trainX,trainY, UB, idx);
end

bigM_sorted = sort(bigM(1:N-1), 'descend');
bigM_L1 = sum(bigM_sorted(1:K));

%% Compute best subset problem

[beta, obj] = bestSubset(trainX,trainY, bigM, bigM_L1, K);
Y_pred_BSS = testX * beta;


