function [ y_out] = nn_train2( x, y, lr, its)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
t = size(x,1);

w = zeros(6, 6);

w(1, 3:6) = rand(1, 4);

w(2, 3:5) = rand(1, 3);

w(3:5, 6) = rand(3, 1);


xx = zeros(t, 6);
 
xx(:, 1) = 1;

xx(:, 2) = x;


g = zeros(t, 6);

g(:, 1) = 1;

g(:, 2) = x;


dedx = zeros(t, 6);


for i = 1:its
    
    for j = 3:5
        
        xx(:, j) = g * w(:, j);
        
        g(:, j) = 1./(1 + exp(-xx(:, j)));
        
    end
    xx(:, 6) = g * w(:, 6);
    
    g(:, 6) = xx(:, 6);
    
    dedx(:, 6) = 2 * (g(:,6) - y);
    
    for m = 5:-1:3
        
        dedx(:, m) = dedx(:, m+1:6) * w(m, m+1:6)' .* g(:, m) .* (1 - g(:,m));
        
    end
    y_out = g(:,6);
    del = g' * dedx;
    
    w = w - lr * del .* (w ~= 0);
    
    
end

end