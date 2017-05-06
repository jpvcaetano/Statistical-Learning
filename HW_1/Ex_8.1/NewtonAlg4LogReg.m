function [ w, n_iters ] = NewtonAlg4LogReg( X, y, lambda, threshold )

% Gradient
u = @(w) exp(X*w) ./ (1 + exp(X*w));
g = @(w) X' * (u(w) - y);

% Hessian
S = @(w) diag(u(w) .* (1 - u(w)));
H = @(w) X' * S(w) * X;

prev_w = zeros(size(X,2),1);
n_iters = 0;
lambdaM = lambda * eye(size(X,2), size(X,2));
while(true)
    w = prev_w - (H(prev_w) + lambdaM)  \ (g(prev_w) + lambda * prev_w);
    n_iters = n_iters + 1;
    
    if( abs(w - prev_w) <= threshold)
        return
    end
    
    prev_w = w;
end

end

