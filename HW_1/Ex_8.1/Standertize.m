function [ outM ] = Standertize( inM )

outM_aux = inM - ones(size(inM,1),1) * mean(inM);

outM = outM_aux ./ (ones(size(inM,1),1) * std(inM));


% outM = (inM - repmat(mean(inM), size(inM,1), 1)) ./ repmat(var(inM), size(inM,1), 1);
end

