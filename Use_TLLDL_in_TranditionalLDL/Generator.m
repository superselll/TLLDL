classdef Generator
    properties
        startend
        level
        rate
        length
        matrix
        base_distribution
    end
    methods
        function obj = Generator(s,rate)
            % 构造函数
            obj.startend = s;
            obj.level = size(obj.startend,2);
            obj.rate = rate;
            obj.startend = obj.startend / sum(obj.startend)*obj.rate*obj.level;
            for i=2:obj.level
                obj.startend(i) = obj.startend(i) + obj.startend(i-1);
            end
            obj.startend = [0,floor(obj.startend)]+1;
            obj.length = obj.startend(1,2:end)-obj.startend(1,1:end-1);
            obj.matrix = zeros(obj.level,obj.level*obj.rate);
            for i = 1:obj.level
                obj.matrix(i,obj.startend(i):obj.startend(i+1)-1) = 1;
            end
            obj.base_distribution = zeros(1,obj.level*obj.rate);
            for i=1:obj.level
                begin = obj.startend(i);
                last = obj.startend(i+1)-1;
                %生成【begin-last】闭区间上的高斯分布
                distribition = genld(begin+(last-begin)/2,3,obj.level*obj.rate);
                obj.base_distribution(begin:last) = distribition(begin:last)/sum(distribition(begin:last));

            end
            obj.base_distribution = obj.base_distribution/sum(obj.base_distribution);
        end

        function weighted_distribution = genld(obj, distribution)  
            weight = distribution * obj.matrix;
            weighted_distribution = weight .* obj.base_distribution;
            weighted_distribution = weighted_distribution./sum(weighted_distribution,2);
        end  
    end  
end


