%% NN for finding a good learning rate

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
error = zeros(5,10000);
erroryv = zeros(5,10000);
hidden = 5;

w = nn_init(test_a,hidden,1);
[ypred, yvpred, w_out, error(1,:), erroryv(1,:)] = nn_train(test_a, ytrain, w, hidden, 1, 0.001, 10000, test_b, yvalid);
[ypred, yvpred, w_out, error(2,:), erroryv(2,:)] = nn_train(test_a, ytrain, w, hidden, 1, 0.002, 10000, test_b, yvalid);
[ypred, yvpred, w_out, error(3,:), erroryv(3,:)] = nn_train(test_a, ytrain, w, hidden, 1, 0.003, 10000, test_b, yvalid);
[ypred, yvpred, w_out, error(4,:), erroryv(4,:)] = nn_train(test_a, ytrain, w, hidden, 1, 0.004, 10000, test_b, yvalid);
[ypred, yvpred, w_out, error(5,:), erroryv(5,:)] = nn_train(test_a, ytrain, w, hidden, 1, 0.008, 10000, test_b, yvalid);


[ypred, yvpred, w_out, error(2,:), erroryv(2,:)] = nn_train(test_a, ytrain, w, hidden, 1, 0.002, 10000, test_b, yvalid);
[test_a, test_b] = norm_x_y(xtrain, xtest);

[yvpred, erroryv] = nn_eval(test_b, ytest, w_out, hidden, 1);

[x_plot, index] = sort(test_a);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, ypred(index), 'r-');
hold off

figure
semilogx(error(1,:), 'b-');
hold on
semilogx(error(2,:), 'r-');
semilogx(error(3,:), 'g-');
semilogx(error(4,:), 'k-');
semilogx(error(5,:), 'c-');
legend('lr = 0.001', 'lr = 0.002','lr = 0.003','lr = 0.004','lr = 0.008');
xlabel('log epoch');
ylabel('Training MSE');
%add test set prediction
