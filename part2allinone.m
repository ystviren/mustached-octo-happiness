%% Ensemble Part 2 ALL IN ONE

% set up all required variables for Ensemble method and graphing
clear all;
close all;

A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
xvalid = A(51:100);
xtest = A(101:200);

ytrain = B(1:50);
yvalid = B(51:100);
ytest = B(101:200);


[xtrnorm, xvnorm] = norm_x_y(xtrain, xvalid);
[~, xtestnorm] = norm_x_y(xtrain, xtest);

xpred = linspace(min(A), max(A), 100)';
[~, xpred_norm] = norm_x_y(xtrain, xpred);

num_models = 34;
test_pred = zeros(num_models, size(ytest,1));
valid_MSE_log = zeros(num_models, 1);
valid_classification_error = zeros(num_models,1);

index = 1;
%kNN hard
for i = 1:2:11
    test_pred(index,:) = kNN_classification(xtrain,ytrain,i,xtest);
    valid_pred = kNN_classification(xtrain,ytrain,i,xvalid);
    valid_MSE_log(index) = sum(abs(yvalid - valid_pred).^2)/size(valid_pred,1);
    valid_classification_error(index) = sum(abs(yvalid - valid_pred))/size(valid_pred,1);
    index = index + 1;
end

%kNN soft
for i = 1:2:11
    test_pred(index,:) = kNN_classification_soft(xtrain,ytrain,i,xtest);
    valid_pred = kNN_classification_soft(xtrain,ytrain,i,xvalid);
    logresult = yvalid'*log(valid_pred) + (1-yvalid)'*log(1-valid_pred);
    logresult = logresult/size(valid_pred,1);
    valid_MSE_log(index) = logresult;
    valid_classification_error(index) = sum(abs(yvalid - valid_pred))/size(valid_pred,1);
    index = index + 1;
end

%polynomial regression

for i = 1:10
    W =  -0.1 + (0.2).*rand(i,1);
    W0 =  -0.1 + (0.2).*rand(1,1);
    [W_out, B_out, MSE_TRAIN, MSE_VALID] = linear_regression(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 6000, 0.01);
    [MSE_TEST, y_out, x_out] = linear_regression_eval(xtrain, xtest, ytest, W_out,B_out);
    [~, yv_out, xv_out] = linear_regression_eval(xtrain, xvalid, yvalid, W_out,B_out);
    valid_MSE_log(index) = MSE_VALID;
    test_pred(index,:) = round(y_out);
    valid_classification_error(index) = sum(abs(yvalid - round(yv_out)))/size(yvalid,1);
    index = index + 1;
end


%logistic regression

for i = 1:10
    W =  -0.1 + (0.2).*rand(i,1);
    W0 =  -0.1 + (0.2).*rand(1,1);
    [W_out, B_out, logtrain, logvalid, predx, predxv] = logistic_regression(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 100000, 0.1);
    [predt, logtest, errort] = logistic_regression_eval(xtrain,xtest,ytest,W_out, B_out);
    test_pred(index,:) = predt;
    valid_MSE_log(index) = logvalid;
    valid_classification_error(index) = sum(abs(yvalid - round(predxv)))/size(yvalid,1);
    index = index+1;
end


%neural net linear output 
%only including early stopping model as it is the most consistent one and
%from part 1 it was seen that too many neural nets in the ensemble caused
%it to converage to teh neural net's accuracy

hidden = 10;

w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv, errmin, i_min] = nn_train_es(xtrnorm, ytrain, w, hidden, 1, 0.005, 1000000, xvnorm, yvalid);
[ytpred, erroryt] = nn_eval(xtestnorm, ytest, w_out, hidden, 1);
test_pred(index,:) = round(ytpred);
valid_MSE_log(index) = erroryv(i_min);
valid_classification_error(index) = sum(abs(yvalid-round(yvpred)))/size(yvpred,1);
index = index + 1;

%nerual net simoid output

hidden = 10;

w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv, errmin, i_min] = nn_train_prob_es(xtrnorm, ytrain, w, hidden, 1, 1, 0.007, 1000000, xvnorm, yvalid);
[ytpred, erroryt] = nn_eval_prob(xtestnorm, ytest, w_out, hidden, 1);
test_pred(index,:) = ytpred;
valid_MSE_log(index) = erroryv(i_min);
valid_classification_error(index) = sum(abs(yvalid-round(yvpred)))/size(yvpred,1);
index = index + 1;


%% Avg of all ensembles

tmp = round(mean(test_pred)');
Err_ensemble1 = sum(abs(ytest-tmp))/size(ytest,1);

%% Best of 3 ensembles
[~, tmp2index] = sort(valid_classification_error);
tmp2 = round(mean(test_pred(tmp2index(1:3),:)))';
Err_ensemble2 = sum(abs(ytest-tmp2))/size(ytest,1);

%% Weighted Enesmbles
weights = valid_classification_error./sum(valid_classification_error);
weights = 1./weights;

pred_weighted = bsxfun(@times, test_pred, weights);
pred_weighted = sum(pred_weighted, 1)/sum(weights);

Err_ensemble3 = sum(abs(ytest - pred_weighted'))/size(ytest,1);




