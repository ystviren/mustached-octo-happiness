function [ pred, error ] = nn_eval ( x, y, w, H, M )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

t = size(x,1);
I = size(x,2) + 1;
xin = zeros(t, I+H+M);
xin(:,1) = 1;
xin(:,2:I) = x;

g = zeros(t, I+H+M);
g(:,1) = 1;
g(:,2:I) = x;

for j =I+1:I+H
    xin(:,j) = g*w(:,j);
    g(:,j) = 1./(1+exp(-xin(:,j)));
end

for j = I+H+1:I+H+M
    xin(:,j) = g*w(:,j);
    g(:,j) = xin(:,j);
end

pred = g(:, I+H+1:I+H+M);

error = sum((y-pred).^2)/t;

end

