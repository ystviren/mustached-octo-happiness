%% Part 1a
clear all
A = importdata('RegressionX.txt');
B = importdata('RegressionY.txt');

train_X = A(1:50);
valid_X = A(51:100);
test_X = A(101:200);

train_Y = B(1:50);
valid_Y = B(51:100);
test_Y = B(101:200);


l = size(test_X, 1);
N = size(test_X, 1);

pred = kNN_regression(train_X,train_Y,1,test_X);


error = test_Y - pred;
error = error.^2;
MSE = sum(error)/N;


%%% 
% The MSE error is 
display (MSE);
%% Part 1b
clear all
A = importdata('RegressionX.txt');
B = importdata('RegressionY.txt');

train_X = A(1:50);
valid_X = A(51:100);
test_X = A(101:200);

train_Y = B(1:50);
valid_Y = B(51:100);
test_Y = B(101:200);


N_valid = size(valid_X, 1);
MSE_valid = zeros (10,1);


for i = 1:10
    pred = kNN_regression(train_X,train_Y,i,valid_X);
    error = valid_Y - pred;
    error = error.^2;
    MSE_valid(i) = sum(error)/N_valid;
end 

display(MSE_valid);
%%%
% choose k to be 3, calculating MSE for test with K = 3
N_test = size(test_X, 1);
pred = kNN_regression(train_X,train_Y,3,test_X);

error = test_Y - pred;
error = error.^2;
MSE_test = sum(error)/N_test;

display (MSE_test);
%% Part 1c

clear all
A = importdata('RegressionX.txt');
B = importdata('RegressionY.txt');

train_X = A(1:50);
valid_X = A(51:100);
test_X = A(101:200);

train_Y = B(1:50);
valid_Y = B(51:100);
test_Y = B(101:200);

W =  -0.1 + (0.2).*rand(1,1);
B =  -0.1 + (0.2).*rand(1,1);

[W_out, B_out, MSE_TRAIN, MSE_VALID] = linear_regression(train_X, train_Y, valid_X, valid_Y, W, B, 1, 600, 0.01);
train_X = A(1:50);
[MSE_TEST, test_out, x_out] = linear_regression_eval(train_X, test_X, test_Y, W_out,B_out);

display(MSE_TRAIN);
display(MSE_TEST);
tmp = cat(2,x_out, test_out);
tmp = sortrows(tmp);
figure
hold on
scatter(x_out, test_Y);
plot(tmp(:,1), tmp(:,2));
hold off

%% Part 1d

clear all

A = importdata('RegressionX.txt');
B = importdata('RegressionY.txt');
train_X = A(1:50);
valid_X = A(51:100);
test_X = A(101:200);

train_Y = B(1:50);
valid_Y = B(51:100);
test_Y = B(101:200);
lowest.mse = 1000000;
lowest.index = 0;
for i = 1:5

train_X = A(1:50);
valid_X = A(51:100);

train_Y = B(1:50);
valid_Y = B(51:100);

W =  -0.1 + (0.2).*rand(i,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, MSE_TRAIN, MSE_VALID] = linear_regression(train_X, train_Y, valid_X, valid_Y, W, W0, 1, 6000, 0.01);
if lowest.mse > MSE_VALID
    lowest.index = i;
    lowest.mse  = MSE_VALID;
end
display(MSE_VALID);
end
for i = 6:10
train_X = A(1:50);
valid_X = A(51:100);

train_Y = B(1:50);
valid_Y = B(51:100);

W =  -0.1 + (0.2).*rand(i,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, MSE_TRAIN, MSE_VALID] = linear_regression(train_X, train_Y, valid_X, valid_Y, W, W0, 1, 70000, 0.01);
display(MSE_VALID);
if lowest.mse > MSE_VALID
    lowest.index = i;
    lowest.mse  = MSE_VALID;
end
end
%lowest MSE is index 10
train_X = A(1:50);
[MSE_TEST, test_out, x_out] = linear_regression_eval(train_X, test_X, test_Y, W_out,B_out);
display(lowest.mse);
display(MSE_TEST);
tmp = cat(2, x_out, test_out);
tmp = sortrows(tmp);
figure
hold on
scatter(x_out, test_Y);
plot(tmp(:,1), tmp(:,2));
hold off