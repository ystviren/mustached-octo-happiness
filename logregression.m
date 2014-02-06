%% Logistic Regression
clear all
A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
xvalid = A(51:100);
xtest = A(101:200);

ytrain = B(1:50);
yvalid = B(51:100);
ytest = B(101:200);

logvalid = zeros(10,1);

lowest_log = -10000;
lowest_index = 0;

for i = 1:10
W =  -0.1 + (0.2).*rand(i,1);
W0 =  -0.1 + (0.2).*rand(1,1);
[W_out, B_out, logtrain, logvalid(i), predx, predxv] = logistic_regression(xtrain, ytrain, xvalid, yvalid, W, W0, 1, 100000, 0.1);
if lowest_log < logvalid(i)
    lowest_log = logvalid(i);
    lowest_index = i;
    W_lowest = W_out;
    B_lowest = B_out;
end

end

[pred, logtest, error] = logistic_regression_eval(xtrain,xtest,ytest,W_lowest, B_lowest);

figure
hold on
plot(xtest, ytest, 'b+');
plot(xtest, round(pred), 'ro');

