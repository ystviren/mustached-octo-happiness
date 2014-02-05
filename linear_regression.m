function [w_out, b_out, MSE_train, MSE_valid ] = linear_regression( x, y, valx, valy, w, b, alpha, its, lr )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

k = size(w,1);
n_train = size(x,1);
n_valid = size(valx,1);
x_train = cat(2, x, ones(n_train, k -1));
x_valid = cat(2, valx, ones(n_valid, k - 1));
w_out = cat(1, w, b);

if k > 1
    for i = 2:k
        x_train(:,i) = x.*x_train(:,i-1);
        x_valid(:,i) = valx.*x_valid(:,i-1);
    end
end
mean_x = mean(x_train, 1);
std_x = std(x_train, 1);
x_train = bsxfun(@minus, x_train, mean_x);
x_train = bsxfun(@rdivide, x_train, std_x);
x_valid = bsxfun(@minus, x_valid, mean_x);
x_valid = bsxfun(@rdivide, x_valid, std_x);
x_train = cat(2, x_train, ones(n_train, 1));
x_valid = cat(2, x_valid, ones(n_valid, 1));


w_in = w_out;
for i = 1:its
y_train = x_train*w_in;
y_valid = x_valid*w_in;
%y  - y_hat then squared
tmp = y - y_train;
tmp = tmp.^2;


tmp2 = valy - y_valid;
tmp2 = tmp2.^2;

%calculate MSE
MSE_train = sum(tmp)/n_train;
MSE_valid = sum(tmp2)/n_valid;

tmp = y - y_train;
w_out = w_in(1:k,1);
b_out = w_in(k+1,1);
dE_dw = (tmp'*x_train).*2./n_train;
w_in = w_in.*alpha + dE_dw'.*lr;
%display(MSE_valid);

end


end

