

function ld = genld(label, sigma,class_num)
label_set = (1:class_num);
ld_num = size(label_set,2);
dif_age = repmat(label_set',1,size(label,2)) - repmat(label,ld_num,1);
ld = 1.0 ./ repmat(sqrt(2*pi)*sigma,ld_num,1) .* exp(-1*dif_age.*dif_age ./ repmat(2*sigma.*sigma,ld_num,1));
ld = ld ./ sum(ld,1) ;
ld = ld';
end




