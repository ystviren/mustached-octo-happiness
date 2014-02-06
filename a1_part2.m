%% Part 2a
clear all
A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
xvalid = A(51:100);
xtest = A(101:200);

ytrain = B(1:50);
yvalid = B(51:100);
ytest = B(101:200);


l = size(xtest, 1);
N = size(xtest, 1);

pred = kNN_classification(xtrain,ytrain,1,xtest);


error = abs(ytest - pred);
MSE_test = sum(error)/N;

confusion = zeros(2,2);
for i = 1:100
        confusion(1 + ytest(i), 1 + pred(i)) =  confusion(1 + ytest(i), 1 + pred(i))  + 1;
end

%% Part 2b
clear all
A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
xvalid = A(51:100);
xtest = A(101:200);

ytrain = B(1:50);
yvalid = B(51:100);
ytest = B(101:200);


l = size(xvalid, 1);
N = size(xvalid, 1);
min_err = 1;
k = 0;
MSE_valid = zeros(6, 1);
count  = 1;
for i = 1:2:11
    pred = kNN_classification(xtrain,ytrain,i,xvalid);
    error = abs(yvalid - pred);
    MSE_valid(count,1) = sum(error)/N;
    MSE = sum(error)/N;
    if min_err > MSE
        k = i;
        min_err = MSE;
    end
    count = count + 1;
end

pred = kNN_classification(xtrain,ytrain,k,xtest);
test_error = sum(abs(ytest-pred))/size(xtest,1);

confusion = zeros(2,2);
for i = 1:100
        confusion(1 + ytest(i), 1 + pred(i)) =  confusion(1 + ytest(i), 1 + pred(i))  + 1;
end

%% Part 2c

clear all
A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
xvalid = A(51:100);
xtest = A(101:200);

ytrain = B(1:50);
yvalid = B(51:100);
ytest = B(101:200);


l = size(xvalid, 1);
N = size(xvalid, 1);
min_err = 1;
k = 0;
loglikehood_valid = zeros(6, 1);
count  = 1;
for i = 1:2:11
    pred = kNN_classification_soft(xtrain,ytrain,i,xvalid);
    error = abs(yvalid - round(pred));
    loglikehood_valid(count) = yvalid'*log(pred) + (1 - yvalid)'*log(1-pred);
    loglikehood_valid(count) = loglikehood_valid(count)/N;
    MSE = sum(error)/N;
    if min_err > MSE
        k = i;
        min_err = MSE;
    end
    count = count + 1;
end

pred = kNN_classification_soft(xtrain,ytrain,k,xtest);
test_error = sum(abs(ytest-pred))/size(xtest,1);
test_logerror = ytest'*log(pred) + (1 - ytest)'*log(1-pred);
test_pred = pred;
pred = round(pred);
confusion = zeros(2,2);
for i = 1:100
        confusion(1 + ytest(i), 1 + pred(i)) =  confusion(1 + ytest(i), 1 + pred(i))  + 1;
end

%% Part 2d
clear all
A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
xvalid = A(51:100);
xtest = A(101:200);

ytrain = B(1:50);
yvalid = B(51:100);
ytest = B(101:200);

W =  -0.1 + (0.2).*rand(1,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, MSE_TRAIN, MSE_VALID] = linear_regression(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 6000, 0.01);
[MSE_TEST, y_out, x_out] = linear_regression_eval(xtrain, xtest, ytest, W_out,B_out);

y_out = round(y_out);
test_error = sum(abs(ytest-y_out))/size(xtest,1);

confusion = zeros(2,2);
for i = 1:100
        confusion(1 + ytest(i), 1 + y_out(i)) =  confusion(1 + ytest(i), 1 + y_out(i))  + 1;
end
figure
hold on
plot(xtrain,ytrain, 'b*');
plot(xvalid,yvalid, 'g+');
plot(xtest,y_out, 'ro');

%% Part 2e

lowest_valid = 10000;
lowest_index = 0;
MSE_VALID = zeros(10,1);
MSE_TRAIN = zeros(10,1);
for i = 1:5
W =  -0.1 + (0.2).*rand(i,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, MSE_TRAIN(i), MSE_VALID(i)] = linear_regression_class(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 6000, 0.01);
if lowest_valid> MSE_VALID(i)
    lowest_index = i;
    lowest_valid = MSE_VALID(i);
	W_lowest = W_out;
    B_lowest = B_out;
end
end

for i = 6:10
W =  -0.1 + (0.2).*rand(i,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, MSE_TRAIN(i), MSE_VALID(i)] = linear_regression_class(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 6000, 0.01);
if lowest_valid> MSE_VALID(i)
	lowest_index = i;
	lowest_valid = MSE_VALID(i);
	W_lowest = W_out;
    B_lowest = B_out;
end
end

[MSE_TEST, y_out, x_out] = linear_regression_eval(xtrain, xtest, ytest, W_lowest,B_lowest);

y_out = round(y_out);
test_error = sum(abs(ytest-y_out))/size(xtest,1);

confusion = zeros(2,2);
for i = 1:100
        confusion(1 + ytest(i), 1 + y_out(i)) =  confusion(1 + ytest(i), 1 + y_out(i))  + 1;
end
figure
hold on
plot(xtrain,ytrain, 'b*');
plot(xvalid,yvalid, 'g+');
plot(xtest,y_out, 'ro');

