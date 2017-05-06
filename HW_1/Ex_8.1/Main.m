clear; close all; clc
%% Load Data
load('spamData.mat')

%% Preprocess Features


Xtrain = [ones(size(Xtrain,1),1), Xtrain];

%% Set randomly the Test and Validation sets according to var proportion
proportion = 4/5;
k = randperm(size(Xtrain,1));
Xvalidate = Xtrain;
yvalidate = ytrain;
Xvalidate(k(1:size(Xtrain,1)*proportion),:) = [];
yvalidate(k(1:size(ytrain,1)*proportion),:) = [];
Xtrain = Xtrain(k(1:size(Xtrain,1)*proportion),:);
ytrain = ytrain(k(1:size(ytrain,1)*proportion),:);
Xtrain = Standertize(Xtrain);
%% Plot the error obtained in the validation after training set in acordance to the value of lambda used for training 
costF = @(w, lambda, X, y) sum(log(1 + exp(X * w)) - y .* (X * w)) + lambda * norm(w,2)^2;
threshold = 1e-10;
lambdas = [];
errors = [];
imin = -10;
imax = 10;
for i=imin:imax
    lambda = 10^i;
    [ w_opt, ~, ~ ] = TrainLogRegClassifier( Xtrain, ytrain, lambda, threshold, costF );
    [ y ] = LogRegClassify( w_opt, Xvalidate );
    %error = sum(abs(y - yvalidate));
    error = costF(w_opt, lambda, Xvalidate, yvalidate) / size(Xvalidate,1);
    
    lambdas = [lambdas; lambda];
    errors = [errors; error]; 
end

lower_error = min(errors);

figure 
plot(imin:imax, errors)
xlabel('lambda = 10^x')
ylabel('error')
