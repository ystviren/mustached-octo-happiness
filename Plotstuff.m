%% Plot Classfication
clear all
close all

A = importdata('ClassificationX.txt');
B = importdata('ClassificationY.txt');

figure
plot(A,B,'bo');