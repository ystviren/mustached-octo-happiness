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
logtrain = zeros(6, epochs);
logvalid = zeros(6, epochs);
hidden = 5;

w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, logtrain(1,:), logvalid(1,:)] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.01, epochs, xvnorm, yvalid);
[ypred, yvpred, w_out, logtrain(2,:), logvalid(2,:)] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.007, epochs, xvnorm, yvalid);
[ypred, yvpred, w_out, logtrain(3,:), logvalid(3,:)] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.005, epochs, xvnorm, yvalid);
[ypred, yvpred, w_out, logtrain(4,:), logvalid(4,:)] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.003, epochs, xvnorm, yvalid);
[ypred, yvpred, w_out, logtrain(5,:), logvalid(5,:)] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.001, epochs, xvnorm, yvalid);



[ypred, yvpred, w_out, logtrain(6,:), logvalid(6,:)] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.007, epochs, xvnorm, yvalid);
[yvpred, logtest] = nn_eval_prob(xtestnorm, ytest, w_out, hidden, 1);

test_error = sum(abs(ytest-round(yvpred)))/100;
[x_plot, index] = sort(xtrnorm);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, round(ypred(index)), 'r+');
hold off

figure
semilogx(logtrain(1,:), 'b-');
hold on
semilogx(logtrain(2,:), 'r-');
semilogx(logtrain(3,:), 'g-');
semilogx(logtrain(4,:), 'k-');
semilogx(logtrain(5,:), 'c-');
legend('lr = 0.01', 'lr = 0.007','lr = 0.005','lr = 0.003','lr = 0.001');
xlabel('log epoch');
ylabel('Log likelihood');
%MSE  -0.3437
%test error 13%
%select lr = 0.07

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
[ypred, yvpred, w_out, logtrain, logvalid] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.007, 10000, xvnorm, yvalid);
hist_data(i) = logtrain(10000);
if min_error > logtrain(10000)
    w_final = w_out;
    min_error = logtrain(10000);
    min_errorv_class = sum(abs(yvalid-round(yvpred)))/size(yvalid,1);
end
end

[~, xtestnorm] = norm_x_y(xtrain, xtest);
[ytpred, erroryt] = nn_eval_prob(xtestnorm, ytest, w_final, hidden, 1);

test_log = erroryt;
test_error = sum(abs(ytest - round(ytpred)))/size(ytest,1);

figure
hist(hist_data);
xlabel('MSE');
ylabel('Frequency')
title('MSE spread');

tmp = mean(hist_data);
tmp2 = std(hist_data);

%mean -0.2184
%std 0.0027

%error 13%
%test log -0.3454

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
for k = 1:10
    min_error = 1000;
    [xtrnorm, xvnorm] = norm_x_y(xtrain, xvalid);
    hidden = k;
for i = 1:20
w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, logtrain, logvalid] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.007, epochs, xvnorm, yvalid);
if min_error > logtrain(epochs)
    w_final = w_out;
    min_error = logtrain(epochs);
    min_errorv(k) = logvalid(epochs);
    min_errorv_class(k) = sum(abs(yvalid-round(yvpred)))/size(yvalid,1);
end
end
[~, xtestnorm] = norm_x_y(xtrain, xtest);
[ytpred, test_error(k)] = nn_eval_prob(xtestnorm, ytest, w_final, hidden, 1);
test_error_class(k) = sum(abs(ytest - round(ytpred)))/size(ytest,1);

end
[x_plot, index] = sort(xtrnorm);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, round(ypred(index)), 'r-');
% choose k = 2
% logprobabilty is -0.3686
% classification 13%

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
[ypred, yvpred, w_out, logtrain, logvalid, errmin, i_min] = nn_train_prob_es(xtrnorm, ytrain, w, hidden, 1, 1, 0.007, 1000000, xvnorm, yvalid);

classv_error = sum(abs(yvalid-round(yvpred)))/size(yvpred,1);

[~, xtestnorm] = norm_x_y(xtrain, xtest);
[ytpred, erroryt] = nn_eval_prob(xtestnorm, ytest, w_out, hidden, 1);
classtest_error = sum(abs(ytest-round(ytpred)))/size(ytpred,1);
[x_plot, index] = sort(xtrnorm);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, ypred(index), 'r-');
hold off

figure
semilogx(logtrain(logtrain~=0), 'b-');
hold on
semilogx(logvalid(logvalid~=0), 'r-');
legend('training error', 'validation error');
xlabel('log epochs');
ylabel('Log Likelihood');

%iteration 2131
%classification errror 12%
%loglikehood -0.3658
