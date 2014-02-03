%% Part 2a
clear all
A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

train_X = A(1:50);
valid_X = A(51:100);
test_X = A(101:200);

train_Y = B(1:50);
valid_Y = B(51:100);
test_Y = B(101:200);


l = size(test_X, 1);
N = size(test_X, 1);

pred = kNN_classification(train_X,train_Y,1,test_X);


error = abs(test_Y - pred);
MSE = sum(error)/N;

confusion = zeros(2,2);
for i = 1:N
        confusion(1 + test_Y(i), 1 + pred(i)) =  confusion(1 + test_Y(i), 1 + pred(i))  + 1;
end
display(confusion);

display (MSE);

%% Part 2b
clear all
A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

train_X = A(1:50);
valid_X = A(51:100);
test_X = A(101:200);

train_Y = B(1:50);
valid_Y = B(51:100);
test_Y = B(101:200);


l = size(valid_X, 1);
N = size(valid_X, 1);
min_err = 1;
k = 0;
for i = 1:2:11
    pred = kNN_classification(train_X,train_Y,i,valid_X);
    error = abs(valid_Y - pred);
    MSE = sum(error)/N;
    if min_err > MSE
        k = i;
        min_err = MSE;
    end
    display(MSE);
end

pred = kNN_classification(train_X,train_Y,k,test_X);
test_error = sum(abs(test_Y-pred))/size(test_X,1);

confusion = zeros(2,2);
for i = 1:N
        confusion(1 + test_Y(i), 1 + pred(i)) =  confusion(1 + test_Y(i), 1 + pred(i))  + 1;
end
display(confusion);
display (test_error);

