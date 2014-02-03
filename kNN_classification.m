function [ pred_Y ] = kNN_classification( X, Y, k, valid_x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

size_valid = size(valid_x,1);
pred_Y = zeros(size_valid, 1);

for i = 1:size_valid
    tmp = abs(X -valid_x(i));
    tmp2 = cat(2, tmp, Y);
    [tmp3, ~] = sortrows(tmp2);
    result = tmp3(1:k,2);
    final = sum(result)/k;
    pred_Y(i) = round(final);
end

end

