function [ w_opt, mean_error, n_iters ] = TrainLogRegClassifier( X, y, lambda, threshold, costF )

[ w_opt, n_iters ] = NewtonAlg4LogReg( X, y, lambda, threshold );

mean_error = costF(w_opt, lambda, X, y) / size(X,1);

end

