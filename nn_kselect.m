%% NN selecting k with random restart

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



min_errorv = zeros(10,1);
min_errorv(:) = 1000;

epochs = 10000;

test_error = zeros(10,1);
for k = 1:10
    min_error = 1000;
    [test_a, test_b] = norm_x_y(xtrain, xvalid);
    hidden = k;
for i = 1:20
w = nn_init(test_a,hidden,1);
[ypred, yvpred, w_out, error, erroryv] = nn_train(test_a, ytrain, w, hidden, 1, 0.002, epochs, test_b, yvalid);
if min_error > error(epochs)
    w_final = w_out;
    min_error = error(epochs);
    min_errorv(k) = erroryv(epochs);
end
end
[test_a, test_b] = norm_x_y(xtrain, xtest);
[ytpred, test_error(k)] = nn_eval(test_b, ytest, w_final, hidden, 1);

end
[x_plot, index] = sort(test_a);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, ypred(index), 'r-');



