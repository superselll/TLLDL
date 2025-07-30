function [target,gradient] = bfgsProcess(weights)
%BFGSPROCESS	Provide the target function and the gradient.
%
%	Description
%   [TARGET,GRADIENT] = BFGSPROCESS(WEIGHTS) provides the target function and the gradient.
%   They will be used in the optimization of BFGSLLD -- BFGSLLDTRAIN.
%   
%   Inputs,
%       WEIGHTS: the weights which will be optimized in BFGSLLDTRAIN
%   Outputs,
%       TARGET:  the target function which will be used in BFGSLLDTRAIN
%       GRADIENT: the gradient which will be used in BFGSLLDTRAIN
% 
%	See also
%	BFGSLLDTRAIN, LLDPREDICT, FMINLBFGS 
%	
%   Copyright: Xin Geng (xgeng@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

% Load the data set.

global name
global index
load(name)
total_num = size(Distribution,1);
sep = round(total_num/5);
testDistribution = Distribution(index(1:sep),:);
testFeature = Feature(index(1:sep),:);
trainDistribution = Distribution(index(sep+1:end),:);
trainFeature = Feature(index(sep+1:end),:);
trainNum=size(trainDistribution,1);
testNum=size(testDistribution,1);
use = 1;
if use == 1
s = ones(1,size(trainDistribution,2));%sum(trainDistribution,1)+sum(testDistribution,1);
rate = 4;
g = Generator(s,rate);
trainDistribution = g.genld(trainDistribution);
matrix = g.matrix;
end
% lambda=0.5;
modProb = exp(trainFeature * weights);  % size_sam * size_Y
sumProb = sum(modProb, 2);
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));

% Target function.

target = -sum(sum(trainDistribution.*log(modProb)));

% The gradient.
gradient = trainFeature'*(modProb - trainDistribution);

end
