%% NN test probablistic output

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
epochs = 10000;
w = nn_init(xtrnorm,hidden,1);
[ypred, yvpred, w_out, error, erroryv, logtrain, logvalid] = nn_train_prob(xtrnorm, ytrain, w, hidden, 1, 1, 0.01, epochs, xvnorm, yvalid);
%[yvpred, erroryv] = nn_eval(test_b, yvalid, w_out, hidden, 1);

[x_plot, index] = sort(xtrnorm);

figure
hold on
plot(x_plot, B(index), 'yo');
plot(x_plot, round(ypred(index)), 'r+');
hold off



%add test set prediction
