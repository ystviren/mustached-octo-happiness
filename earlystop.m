%% NN test early stopping

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

hidden = 10;

% should be around 212235

w = nn_init(test_a,hidden,1);
[ypred, yvpred, w_out, error, erroryv, errmin, i_min] = nn_train_es(test_a, ytrain, w, hidden, 1, 0.002, 1000000, test_b, yvalid);

[test_a, test_b] = norm_x_y(xtrain, xtest);
[ytpred, erroryt] = nn_eval(test_b, ytest, w_out, hidden, 1);

[x_plot, index] = sort(test_a);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, ypred(index), 'r-');
hold off

figure
semilogx(error(error~=0), 'b-');
hold on
semilogx(erroryv(erroryv~=0), 'r-');
legend('training error', 'validation error');
xlabel('log epochs');
ylabel('MSE');

%add test set prediction