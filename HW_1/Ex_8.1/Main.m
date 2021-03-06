clear; close all; clc
%% Load Data
load('spamData.mat')

%% Preprocess Features
choice = 'Which type of preprocessing should be done?\n (1) 0 mean and unit variance\n (2) log(x+0.1)\n (3) binarization\n (else) none\n';
x = input(choice);
if x == 1
    Xtrain = Standertize(Xtrain);
    Xtest = Standertize(Xtest);
elseif x == 2
     Xtrain = LogTransform(Xtrain);   
     Xtest = LogTransform(Xtest); 
elseif x == 3
     Xtrain = Binarize(Xtrain);
     Xtest = Binarize(Xtest);
end
    
Xtrain = [ones(size(Xtrain,1),1), Xtrain];
Xtest = [ones(size(Xtest,1),1), Xtest];
%% Cross-Validation to retrieve the best lambda

train_set_partition = 10;
k = randperm(size(Xtrain,1));
Xtrain = Xtrain(k,:);
ytrain = ytrain(k,:);

threshold = 1e-10;
lambdas = [];
imin = -3;
imax = 3;
progression = 1;

errors = [];
flambda = @(x) 10^x;
for i=imin:progression:imax
    
    initial_pos = 1;
    final_pos = size(Xtrain,1)/train_set_partition;
    lambda = flambda(i);
    error = 0;
    for j=1:train_set_partition
        
        % set validation and train sets
        Xvalidation_aux = Xtrain(initial_pos:floor(j*final_pos),:);
        yvalidation_aux = ytrain(initial_pos:floor(j*final_pos),:);
        Xtrain_aux = Xtrain([1:initial_pos-1,floor(j*final_pos)+1:size(Xtrain,1)],:);
        ytrain_aux = ytrain([1:initial_pos-1,floor(j*final_pos)+1:size(ytrain,1)],:);
        initial_pos = floor(j*final_pos) + 1;
        
        % train classifier and get train error
        [ w_hat, ~ ] = TrainLogRegClassifier( Xtrain_aux, ytrain_aux, lambda, threshold);
        [ y ] = LogRegClassify( w_hat, Xvalidation_aux );
        error = error + sum(abs(double(y) - yvalidation_aux)) / length(y);
        
    end
    
    errors = [errors; error / train_set_partition];
    lambdas = [lambdas; lambda];
    
end
        
figure 
plot(imin:imax, errors)
xlabel('lambda = 10^x')
ylabel('mean error rate in validation')

[min_error_validation, indice] = min(errors);
best_lambda = flambda(imin+(indice-1)*progression);

%% Get error in Train and Test Set



[ w_hat, ~ ] = TrainLogRegClassifier( Xtrain, ytrain, best_lambda, threshold);
[ y ] = LogRegClassify( w_hat, Xtrain );
training_error = sum(abs(double(y) - ytrain)) / length(y);

[ y ] = LogRegClassify( w_hat, Xtest );
test_error = sum(abs(double(y) - ytest)) / length(y);








