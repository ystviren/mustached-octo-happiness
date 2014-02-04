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

num_models = 50;

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
[ypred, yvpred, w_out, error, erroryv] = nn_train(xtrnorm, ytrain, w, hidden, k, 0.002, epochs, xvnorm, yvalid);
[ytestpred, test_error] = nn_eval(xtestnorm, ytest, w_out, hidden, 1);
validation_mse(20+i+k-1,:) = erroryv(epochs);
test_prediction(20+i+k-1,:) = ytestpred;
end
end

% Neural Net with early stop
