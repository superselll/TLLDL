clear;
clc;

s = [1003.711652, 511.380149, 1057.974047, 677.058356, 783.200313, 874.675486];
rate = 10;
g = Generator(s,rate);
L = [[0.080000,0.000000,0.280000,0.000000,0.640000,0.000000];
                             [0.068966,0.000000,0.344828,0.068966,0.482759,0.034483]];
D = g.genld(L);
startend_gt = g.startend;
startend = 1:rate:g.level*g.rate+1;
O = zeros(g.rate*g.level,g.level);
for i=1:g.level
    O(startend(i):startend(i+1)-1,i)=1;
end

markov(D,O,L,startend,@chebyshev)