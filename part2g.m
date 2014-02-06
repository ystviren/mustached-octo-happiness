%% NN classification select learning rate

clear all
close all

A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
ytrain = B(1:50);

xvalid = A(51:100);
yvalid = B(51:100);

xtest = A(101:200);
ytest = B(101:200);

[xtrnorm, xvnorm] = norm_x_y(xtrain, xvalid);
[~, xtestnorm] = norm_x_y(xtrain, xtest);
epochs = 10000;
error = zeros(6, epochs);
erroryv = zeros(6, epochs);
hidden = 5;

w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error(1,:), erroryv(1,:)] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.01, epochs, xvnorm, yvalid);
[ypred, yvpred, w_out, error(2,:), erroryv(2,:)] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.007, epochs, xvnorm, yvalid);
[ypred, yvpred, w_out, error(3,:), erroryv(3,:)] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.005, epochs, xvnorm, yvalid);
[ypred, yvpred, w_out, error(4,:), erroryv(4,:)] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.003, epochs, xvnorm, yvalid);
[ypred, yvpred, w_out, error(5,:), erroryv(5,:)] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.001, epochs, xvnorm, yvalid);



[ypred, yvpred, w_out, error(6,:), erroryv(6,:)] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.005, epochs, xvnorm, yvalid);
[yvpred, erroryv] = nn_eval(xtestnorm, ytest, w_out, hidden, 1);
% [x_plot, index] = sort(xtrnorm);

% figure
% hold on
% plot(x_plot, B(index), 'yo');
% plot(x_plot, round(ypred(index)), 'r+');
% hold off
test_error = sum(abs(ytest-round(yvpred)))/100;

figure
semilogx(error(1,:), 'b-');
hold on
semilogx(error(2,:), 'r-');
semilogx(error(3,:), 'g-');
semilogx(error(4,:), 'k-');
semilogx(error(5,:), 'c-');
legend('lr = 0.01', 'lr = 0.007','lr = 0.005','lr = 0.003','lr = 0.001');
xlabel('log epoch');
ylabel('Training Error(MSE)');

%choose learning rate 0.005
%test MSE = 0.1075
%test classification error = 13%
%% Random restart

clear all
close all

A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
ytrain = B(1:50);

xvalid = A(51:100);
yvalid = B(51:100);

xtest = A(101:200);
ytest = B(101:200);

[xtrnorm, xvnorm] = norm_x_y(xtrain, xvalid);

hidden = 5;
hist_data = zeros(20,1);
min_error = 1000;

for i = 1:20
w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.005, 10000, xvnorm, yvalid);
hist_data(i) = error(10000);
if min_error > error(10000)
    w_final = w_out;
    min_error = error(10000);
    min_errorv_class = sum(abs(yvalid-round(yvpred)))/size(yvalid,1);
end
end

[~, xtestnorm] = norm_x_y(xtrain, xtest);
[ytpred, erroryt] = nn_eval(xtestnorm, ytest, w_final, hidden, 1);

test_mse = erroryt;
test_error = sum(abs(ytest - round(ytpred)))/size(ytest,1);

figure
hist(hist_data);
xlabel('MSE');
ylabel('Frequency')
title('MSE spread');

tmp = mean(hist_data);
tmp2 = std(hist_data);

%% Neural net selecting k

clear all
close all

A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
ytrain = B(1:50);

xvalid = A(51:100);
yvalid = B(51:100);

xtest = A(101:200);
ytest = B(101:200);



min_errorv = zeros(10,1);
min_errorv(:) = 1000;
min_errorv_class = zeros(10,1);

epochs = 10000;

test_error = zeros(10,1);
test_error_class = zeros(10,1);
min_index = 0;
for k = 1:10
    min_error = 1000;
    [xtrnorm, xvnorm] = norm_x_y(xtrain, xvalid);
    hidden = k;
for i = 1:20
w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv] = nn_train(xtrnorm, ytrain, w, hidden, 1, 0.005, epochs, xvnorm, yvalid);
if min_error > error(epochs)
    w_final = w_out;
    min_index = k;
    min_error = error(epochs);
    min_errorv(k) = erroryv(epochs);
    min_errorv_class(k) = sum(abs(yvalid-round(yvpred)))/size(yvalid,1);
end
end
[~, xtestnorm] = norm_x_y(xtrain, xtest);
[ytpred, test_error(k)] = nn_eval(xtestnorm, ytest, w_final, hidden, 1);
test_error_class(k) = sum(abs(ytest - round(ytpred)))/size(ytest,1);

end
[x_plot, index] = sort(xtrnorm);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, round(ypred(index)), 'r-');
% choose k = 6
% mse is 0.1059
% classification rate 13%

%% Early Stopping

clear all
close all

A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

xtrain = A(1:50);
ytrain = B(1:50);

xvalid = A(51:100);
yvalid = B(51:100);

xtest = A(101:200);
ytest = B(101:200);

[xtrnorm, xvnorm] = norm_x_y(xtrain, xvalid);

hidden = 10;


w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv, errmin, i_min] = nn_train_es(xtrnorm, ytrain, w, hidden, 1, 0.005, 1000000, xvnorm, yvalid);

classv_error = sum(abs(yvalid-round(yvpred)))/size(yvpred,1);

[~, xtestnorm] = norm_x_y(xtrain, xtest);
[ytpred, erroryt] = nn_eval(xtestnorm, ytest, w_out, hidden, 1);
classtest_error = sum(abs(ytest-round(ytpred)))/size(ytpred,1);
[x_plot, index] = sort(xtrnorm);

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