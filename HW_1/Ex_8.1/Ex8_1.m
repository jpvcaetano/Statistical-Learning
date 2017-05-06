clear; close all; clc
%% Load Data

load('spamData.mat')

%% Preprocess features

Xtrain = [ones(size(Xtrain,1),1), Xtrain];

%% Train classifier (Newton LogReg Specific Algorithm)

costF = @(w, lambda, X, y) sum(log(1 + exp(X * w)) - y .* (X * w)) + lambda * norm(w,2)^2;

lambda = 0;
threshold = 1e-15;

[ w_opt, mean_error, n_iters ] = TrainLogRegClassifier( Xtrain, ytrain, lambda, threshold, costF );

%% Classification

Xtest = [ones(size(Xtest,1),1), Xtest];
[ y ] = LogRegClassify( w, Xtest );

errors = sum(abs(y - ytest));
