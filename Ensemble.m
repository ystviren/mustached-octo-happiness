%% Ensemble 

clear all;

A = importdata('RegressionX.txt');
B = importdata('RegressionY.txt');

xtrain = A(1:50);
xvalid = A(51:100);
xtest = A(101:200);

ytrain = B(1:50);
yvalid = B(51:100);
ytest = B(101:200);


[xtrnorm, xvnorm] = norm_x_y(xtrain, xvalid);
[~, xtestnorm] = norm_x_y(xtrain, xtest);

num_models = 221;

test_prediction = zeros(num_models, 100); 
validation_mse = zeros(num_models,1);

% Nearest Neighbour

N = size(yvalid,1);
for i = 1:10
test_prediction(i,:) = kNN_regression(xtrain,ytrain,i,xtest);
pred = kNN_regression(xtrain,ytrain,i,xvalid);
error = yvalid - pred;
error = error.^2;
validation_mse(i) = sum(error)/N;
end


% polynomial regression

for i = 1:5

W =  -0.1 + (0.2).*rand(i,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, MSE_TRAIN, MSE_VALID] = linear_regression(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 6000, 0.01);
[MSE_TEST, y_out, x_out] = linear_regression_eval(xtrain, xtest, ytest, W_out,B_out);
test_prediction(10+i,:) = y_out;
validation_mse(10+i,:) = MSE_VALID;
end
for i = 6:10
W =  -0.1 + (0.2).*rand(i,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, MSE_TRAIN, MSE_VALID] = linear_regression(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 6000, 0.01);
[MSE_TEST, y_out, x_out] = linear_regression_eval(xtrain, xtest, ytest, W_out,B_out);
test_prediction(10+i,:) = y_out;
validation_mse(10+i,:) = MSE_VALID;
end

% Neural Nets with random restart
for k = 1:10
hidden = k;
epochs = 10000;
for i = 1:20
w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.002, epochs, xvnorm, yvalid);
[ytestpred, test_error] = nn_eval(xtestnorm, ytest, w_out, hidden, 1);
validation_mse(20+i+(k-1)*20,:) = erroryv(epochs);
test_prediction(20+i+(k-1)*20,:) = ytestpred;
end
end

% Neural Net with early stop
hidden = 10;
w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv, errmin, i_min] = nn_train_es(xtrnorm, ytrain, w, hidden, 1, 0.002, 1000000, xvnorm, yvalid);
[ytpred, erroryt] = nn_eval(xtestnorm, ytest, w_out, hidden, 1);
validation_mse(221, :) = erroryv(i_min);
test_prediction(221, :) = ytpred;

% Calculate all ensembles
pred_all_ensemble = mean(test_prediction);
error_all_ensemble = (ytest - pred_all_ensemble');
error_all_ensemble = sum(error_all_ensemble.^2)/100;

% Calculate best 3 ensembles

[best_model_valid, best_valid_index] = sort(validation_mse);
pred_best_3 = test_prediction(best_valid_index(1:3), :);
pred_best_3 = mean(pred_best_3);

error_best_3 = (ytest - pred_best_3');
error_best_3 = sum(error_best_3.^2)/100;

% Weighted best 3 ensembles

pred_w_best_3 = test_prediction(best_valid_index(1:3), :);
weight_for_3 = best_model_valid(1:3);
weight_for_3 = weight_for_3./(sum(weight_for_3));
weight_for_3 = 1./weight_for_3;

pred_w_best_3_weighted = bsxfun(@times, pred_w_best_3, weight_for_3);
pred_w_best_3_weighted = sum(pred_w_best_3_weighted, 1)/sum(weight_for_3);

error_w_best_3 = (ytest - pred_w_best_3_weighted');
error_w_best_3 = sum(error_w_best_3.^2)/100;

figure
hold on
plot(xtrain,ytrain, 'ro');
plot(xtest,ytest, 'b+');
plot(xtest, pred_w_best_3_weighted, 'go');


