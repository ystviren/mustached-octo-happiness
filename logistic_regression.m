function [w_out, b_out, logtrain, logvalid, predx, predxv ] = logistic_regression( x, y, valx, valy, w, b, alpha, its, lr )
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


%calculate win*xin pass the value into sigmoid fucntion
%calcualte log error for both valid and training.

w_in = w_out;
for i = 1:its
ytrain = xtrain *w_in;
yvalid = xvalid *w_in;

ytrain = 1./(1 + exp(-ytrain));
yvalid = 1./(1 + exp(-yvalid));


predx = ytrain;
predxv = yvalid;
%calculate MSE
logtrain = y'*log(ytrain) + (1 - y)'*log(1-ytrain);
logtrain = logtrain/ntrain;


logvalid = valy'*log(yvalid) + (1 - valy)'*log(1-yvalid);
logvalid = logvalid/nvalid;

tmp = y - ytrain;

w_out = w_in(1:k,1);
b_out = w_in(k+1,1);
dE_dw = (tmp'*xtrain)./ntrain;
w_in = w_in.*alpha + dE_dw'.*lr;
%display(MSE_valid);

end

end