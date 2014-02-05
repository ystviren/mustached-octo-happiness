%% Plot all graphs

clear all;
close all;

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

xpred = linspace(min(A), max(A), 100)';
[~, xpred_norm] = norm_x_y(xtrain, xpred);
num_models = 9;

test_prediction = zeros(num_models, 100); 

% Nearest Neighbour
test_prediction(1,:) = kNN_regression(xtrain,ytrain,1,xpred);

% kNN for k = 3
test_prediction(2,:) = kNN_regression(xtrain,ytrain,3,xpred);


% linear regression

W =  -0.1 + (0.2).*rand(1,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, ~, ~] = linear_regression(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 6000, 0.01);
[~, y_out, ~] = linear_regression_eval(xtrain, xpred, ytest, W_out,B_out);
test_prediction(3,:) = y_out;


W =  -0.1 + (0.2).*rand(9,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, ~, ~] = linear_regression(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 100000, 0.07);
[~, y_out, ~] = linear_regression_eval(xtrain, xpred, ytest, W_out,B_out);
test_prediction(4,:) = y_out;

% neural net
epochs = 10000;
hidden = 5;

w = nn_init(xtrnorm,hidden,1);
[~, ~, w_out, error, erroryv] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.002, epochs, xvnorm, yvalid);
[test_prediction(5,:), ~] = nn_eval(xpred_norm, ytest, w_out, hidden, 1);

% Neural Nets with random restart
min_errorv = zeros(10,1);
min_errorv(:) = 1000;

test_error = zeros(10,1);
min_error = 1000;

for i = 1:20
w = nn_init(xtrnorm,hidden,1);
[~, ~, w_out, error, erroryv] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.002, epochs, xvnorm, yvalid);
if min_error > error(epochs)
    w_final = w_out;
    min_error = error(epochs);
end
end

[test_prediction(6,:), ~] = nn_eval(xpred_norm, ytest, w_final, hidden, 1);

% Nueral Net with k = 10
epochs = 10000;
hidden = 10;

w = nn_init(xtrnorm,hidden,1);
[~, ~, w_out, error, erroryv] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.002, epochs, xvnorm, yvalid);
[test_prediction(7,:), ~] = nn_eval(xpred_norm, ytest, w_out, hidden, 1);

% Neural Net with early stop
hidden = 10;
w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv, errmin, i_min] = nn_train_es(xtrnorm, ytrain, w, hidden, 1, 0.002, 1000000, xvnorm, yvalid);
[ytpred, erroryt] = nn_eval(xpred_norm, ytest, w_out, hidden, 1);
test_prediction(8, :) = ytpred;



% Weighted best 3 ensembles (Neural Net early stopping, 3-NN and 2-NN

pred_w_best3 = zeros(3, 100);
valid_w_best3 = zeros(3,1);

pred_w_best3(1,:) = kNN_regression(xtrain,ytrain,3,xpred);
pred = kNN_regression(xtrain,ytrain,3,xvalid);
error = yvalid - pred;
error = error.^2;
valid_w_best3(1) = sum(error)/50;

pred_w_best3(2,:) = kNN_regression(xtrain,ytrain,2,xpred);
pred = kNN_regression(xtrain,ytrain,2,xvalid);
error = yvalid - pred;
error = error.^2;
valid_w_best3(2) = sum(error)/50;

pred_w_best3(3,:) = ytpred;
valid_w_best3(3) = erroryv(i_min);

weight_for_3 = valid_w_best3;
weight_for_3 = weight_for_3./(sum(weight_for_3));
weight_for_3 = 1./weight_for_3;

pred_w_best_3_weighted = bsxfun(@times, pred_w_best3, weight_for_3);
pred_w_best_3_weighted = sum(pred_w_best_3_weighted, 1)./sum(weight_for_3);

test_prediction(9,:) = pred_w_best_3_weighted;

figure
hold on
plot(xtrain,ytrain, 'b+');
plot(xtest, ytest, 'ro');
plot(xpred, test_prediction(1,:), 'Color', [0 0.5 0.5]);
plot(xpred, test_prediction(2,:), 'Color', [0.5 0.5 0.5]);
plot(xpred, test_prediction(3,:), 'Color', [0.5 0 0]);
plot(xpred, test_prediction(4,:), 'Color', [0.5 0 0.5]);
plot(xpred, test_prediction(5,:), 'Color', [0 0.5 0]);
plot(xpred, test_prediction(6,:), 'Color', [0 0 0.5]);
plot(xpred, test_prediction(7,:), 'Color', [0.5 0.5 1]);
plot(xpred, test_prediction(8,:), 'Color', [0.5 1 0.5]);
plot(xpred, test_prediction(9,:), 'Color', [1 0.5 0.5]);
legend('training', 'test set', 'Nearest Neigbour', '3NN', 'linear', 'polynomial', 'Neural Net', 'Neural Net Random Restart', 'Neural Net k = 10', 'Early Stopping', 'Ensemble');
xlabel('x');
ylabel('y');



