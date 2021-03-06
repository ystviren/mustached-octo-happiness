%% NN for random restart

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
hist_data = zeros(20,1);
min_error = 1000;
for i = 1:20
w = nn_init(test_a,hidden,1);
[ypred, yvpred, w_out, error, erroryv] = nn_train(test_a, ytrain, w, hidden, 1, 0.002, 10000, test_b, yvalid);
hist_data(i) = error(10000);
if min_error > error(10000)
    w_final = w_out;
    min_error = error(10000);
end
end
[x_plot, index] = sort(test_a);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, ypred(index), 'r-');

[test_a, test_b] = norm_x_y(xtrain, xtest);
[ytpred, erroryt] = nn_eval(test_b, ytest, w_final, hidden, 1);

figure
semilogx(error, 'b-');
hold on
semilogx(erroryv, 'r-');
hold off

figure
hist(hist_data);
xlabel('MSE');
ylabel('Frequency')
title('MSE spread');

tmp = mean(hist_data);
tmp2 = std(hist_data);
