function [ y ] = LogRegClassify( w, X )

y = (X * w >= 0);

end

