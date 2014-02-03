function [ w ] = nn_init(x, H, M)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

I = size(x,2) + 1;

w = zeros(I + H + M, I + H + M);

w(1, I+1:I+H+M) = -0.1 + (0.2).*rand(1,H+M);
w(2:I, I+1:I+H) = -0.1 + (0.2).*rand((I-1),H);
w(I+1:I+H, I+H+1:I+H+M) =  -0.1 + (0.2).*rand(H, M);

end

