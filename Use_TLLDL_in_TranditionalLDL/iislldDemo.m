%IISLLDDEMO	The example of IISLLD algorithm.
%
%	Description
%   We establish a maximum entropy model and use IIS algorithm to estimate
%   the parameters. In this way, we can get our LDL model. Then a new 
%   distribution can be predicted based on this model.
% 
%	See also
%	LLDPREDICT, IISLLDTRAIN
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
rate = 10;
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

% Initialize the model parameters.
para.minValue = 1e-7; % the feature value to replace 0, default: 1e-7
para.iter = 10; % learning iterations, default: 50 / 200 
para.minDiff = 1e-4; % minimum log-likelihood difference for convergence, default: 1e-7
para.regfactor = 0; % regularization factor, default: 0

tic;
% The training part of IISLLD algorithm.
[weights] = iislldTrain(para, trainFeature, D_gt);
fprintf('Training time of IIS-LLD: %8.7f \n', toc);
if use == 1
for epoch = 1:1
    D_predict = lldPredict(weights,trainFeature);
    [O, Ostartend] = markov(D_predict,O,L,Ostartend,@kldist);
    disp(Ostartend)
end

end
% Prediction
preDistribution = lldPredict(weights,testFeature);
if use == 1
    preDistribution = preDistribution*matrix';
end
fprintf('Finish prediction of IIS-LLD. \n');

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

end

