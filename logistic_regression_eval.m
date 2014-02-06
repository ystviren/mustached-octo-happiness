function [predtest, logtest, errort ] = logistic_regression_eval( x, valx, valy, w, b)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

k = size(w,1);
ntrain = size(x,1);
nvalid = size(valx,1);
xtrain = cat(2, x, ones(ntrain, k -1));
xvalid = cat(2, valx, ones(nvalid, k - 1));
w_out = cat(1, w, b);

if k > 1
    for i = 2:k
        xtrain(:,i) = x.*xtrain(:,i-1);
        xvalid(:,i) = valx.*xvalid(:,i-1);
    end
end
mean_x = mean(xtrain, 1);
std_x = std(xtrain, 1);
xtrain = bsxfun(@minus, xtrain, mean_x);
xtrain = bsxfun(@rdivide, xtrain, std_x);
xvalid = bsxfun(@minus, xvalid, mean_x);
xvalid = bsxfun(@rdivide, xvalid, std_x);
xtrain = cat(2, xtrain, ones(ntrain, 1));
xvalid = cat(2, xvalid, ones(nvalid, 1));

w_in = w_out;

ytrain = xtrain *w_in;
yvalid = xvalid *w_in;

ytrain = 1./(1 + exp(-ytrain));
yvalid = 1./(1 + exp(-yvalid));


predtest = yvalid;
%calculate MSE

logvalid = valy'*log(yvalid) + (1 - valy)'*log(1-yvalid);
logvalid = logvalid/nvalid;
logtest = logvalid;

errort = abs(valy - round(yvalid));
errort = sum(errort)/nvalid;


end

