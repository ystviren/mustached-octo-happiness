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
for i = 1:N
        confusion(1 + ytest(i), 1 + pred(i)) =  confusion(1 + ytest(i), 1 + pred(i))  + 1;
end
display(confusion);

display (MSE_test);

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
for i = 1:N
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

pred = kNN_classification(xtrain,ytrain,k,xtest);
test_error = sum(abs(ytest-pred))/size(xtest,1);
test_logerror = ytest'*log(pred) + (1 - ytest)'*log(1-pred);

confusion = zeros(2,2);
for i = 1:N
        confusion(1 + ytest(i), 1 + pred(i)) =  confusion(1 + ytest(i), 1 + pred(i))  + 1;
end

