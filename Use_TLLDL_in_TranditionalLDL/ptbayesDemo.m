%PTBAYESDEMO	The example of PTBayes algorithm.
%
%	Description  
%   A demo of using PTBayes algorithm.
% 
% See also
%       PTBAYESTRAIN, RESAMPLE, BAYES, PTBAYESPREDICT
%   
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

clear;
clc;
datasetnames = {'Yeast_alpha'};
loops = 5;

%datasetnames = {'Yeast_alpha','Yeast_cdc','Yeast_cold','Yeast_dtt','Yeast_heat','Yeast_spo','Yeast_spo5','Natural_Scene','SJAFFE','SBU_3DFE'};
for loop = 1:loops
result_loss = [];
result_O = [];
for jj = 1:size(datasetnames,2)
name = string(datasetnames(jj));
load(name)

% Load the trainData and testData.
%load yeastcoldDataSet;
total_num = size(Distribution,1);
index = randperm(total_num);
sep = round(total_num/5);
testDistribution = Distribution(index(1:sep),:);
testFeature = Feature(index(1:sep),:);
trainDistribution = Distribution(index(sep+1:end),:);
trainFeature = Feature(index(sep+1:end),:);
trainNum=size(trainDistribution,1);
testNum=size(testDistribution,1);

D_gt = trainDistribution;
use = 1;
if use==1
s = ones(1,size(trainDistribution,2));%sum(trainDistribution,1)+sum(testDistribution,1);
rate =10;
g = Generator(s,rate);
startend_gt = g.startend;
matrix = g.matrix;
disp(startend_gt);
D_gt = g.genld(trainDistribution);
L = trainDistribution;
Ostartend = 1:rate:g.level*g.rate+1;
O = zeros(g.rate*g.level,g.level);
for i=1:g.level
    O(Ostartend(i):Ostartend(i+1)-1,i)=1;
end
end
tic;

% Training of PTBayes
model = ptbayesTrain(trainFeature,D_gt);
fprintf('Training time of PT-Bayes: %8.7f \n', toc);
if use == 1
for epoch = 1:0
    D_predict = ptbayesPredict(model,trainFeature);
    [O, Ostartend] = markov(D_predict,O,L,Ostartend,@kldist);
    %disp(Ostartend)
end
end
%Prediction of PTBayes
preDistribution = ptbayesPredict(model, testFeature);
if use == 1
    preDistribution = preDistribution*matrix';
end
fprintf('Finish prediction of PT-Bayes. \n');

total_distance = [];
% To visualize two distribution and display some selected metrics of distance
for i=1:testNum
    % Show the comparisons between the predicted distribution
	[disName, distance] = computeMeasures(testDistribution(i,:), preDistribution(i,:));
    total_distance = [total_distance;distance];
    % Draw the picture of the real and prediced distribution.
    
    %drawDistribution(testDistribution(i,:),preDistribution(i,:),disName, distance);
    %sign=input('Press any key to continue:');
end
result_loss = [result_loss;mean(total_distance,1)];
disp(mean(total_distance,1))

if use==1
%result_O = [result_O;Ostartend];
disp(Ostartend)
end
end
xlswrite(strcat(int2str(loop),'.xls'), result_loss);

%result_loss = reshape(result_loss,1,size(result_loss,1),size(result_loss,2));
%total_result_loss = cat(1,total_result_loss,result_loss);
end
%meannum = mean(total_result_loss,1);
%meannum = reshape(meannum,size(meannum,2),size(meannum,3));
%maxnum = max(total_result_loss,1);
%maxnum = reshape(maxnum,size(maxnum,2),size(maxnum,3));
%minnum = min(total_result_loss,1);
%minnum = reshape(minnum,size(minnum,2),size(minnum,3));





