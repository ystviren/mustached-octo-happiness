%% NN test

clear all
close all

A = importdata('RegressionX.txt');
B = importdata('RegressionY.txt');

xtrain = A(1:50);
ytrain = B(1:50);

xvalid = A(51:100);
yvalid = B(51:100);

xtest = A(101:200);
ytest = B(101:200);

[test_a, test_b] = norm_x_y(xtrain, xvalid);

hidden = 5;

w = nn_init(test_a,hidden,1);
[ypred, yvpred, w_out, error, erroryv] = nn_train(test_a, ytrain, w, hidden, 1, 0.001, 50000, test_b, yvalid);
%[yvpred, erroryv] = nn_eval(test_b, yvalid, w_out, hidden, 1);

[x_plot, index] = sort(test_a);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, ypred(index), 'r-');

figure
semilogx(error, 'b-');
hold on
semilogx(erroryv, 'r-');

%add test set prediction
