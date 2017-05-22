function [ w_opt, n_iters ] = TrainLogRegClassifier( X, y, lambda, threshold )

[ w_opt, n_iters ] = NewtonAlg4LogReg( X, y, lambda, threshold );

end

