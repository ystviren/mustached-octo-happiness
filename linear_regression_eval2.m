function [MSE_train, pred_out, x_out] = linear_regression_eval2( x, xv, yv, w, b )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

k = size(w,1);
n_train = size(x,1);
n_valid = size(xv,1);
x_train = cat(2, x, ones(n_train, k -1));
xv_train = cat(2, xv, ones(n_valid, k - 1));
w_out = cat(1, w, b);

if k > 1
    for i = 2:k
        x_train(:,i) = x.*x_train(:,i-1);
        xv_train(:,i) = xv.*xv_train(:,i-1);
    end
end
mean_x = mean(x_train);
std_x = std(x_train);


xv_train = bsxfun(@minus, xv_train, mean_x);
xv_train = bsxfun(@rdivide, xv_train, std_x);
xv_train = cat(2, xv_train, ones(n_valid, 1));

x_out = xv_train(:,1);
w_in = w_out;

y_model = xv_train*w_in;
pred_out = y_model;
%y  - y_hat then squared
tmp = yv - y_model;
tmp = tmp.^2;


%calculate MSE
MSE_train = sum(tmp)/n_valid;



end

