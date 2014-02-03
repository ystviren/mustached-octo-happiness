function [ norm_x, norm_y ] = norm_x_y( x, y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

mean_x = mean(x);
std_x = std(x);
x = bsxfun(@minus, x, mean(x));
x = bsxfun(@rdivide, x, std(x));
y = bsxfun(@minus, y, mean_x);
y = bsxfun(@rdivide, y, std_x);

norm_x = x;
norm_y = y;



end

