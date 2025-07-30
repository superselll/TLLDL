%BFGSLLDDEMO	The example of BFGSLLD algorithm.
%
%	Description
%   In order to optimize the IIS-LLD algorithm, we follow the idea of an
%   effective quasi-Newton method BFGS to further improve IIS-LLD. 
%   Here is an example of BFGSLLD algorithm.
%	
%	See also
%	LLDPREDICT, BFGSLLDTRAIN
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
clear;
clc;
res = [];
global name
global index
datasetnames = {'Yeast_alpha'};

for loop = 1:5
result_loss = [];
result_O = [];
for jj = 1:size(datasetnames,2)
name = string(datasetnames(jj));

load(name)
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
rate = 4;
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
item=eye(size(trainFeature,2),size(D_gt,2));
disp('     Iteration  Func-count   Grad-count         f(x)         Step-size')
% The training part of BFGSLLD algorithm.
tic;
weights = item;
% The function of bfgsprocess provides a target function and the gradient.
for epoch=1:1
[weights,fval] = bfgslldTrain(@bfgsProcess,weights);  %item就是现在的权重
%if use == 1
%D_predict = lldPredict(weights,trainFeature);
%[O, Ostartend] = markov(D_predict,O,L,Ostartend,@chebyshev);
%disp(Ostartend)
%end
%fprintf('Training time of BFGS-LLD: %8.7f \n', toc);
end
if use == 1
for epoch = 1:0
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
fprintf('Finish prediction of BFGS-LLD. \n');
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

