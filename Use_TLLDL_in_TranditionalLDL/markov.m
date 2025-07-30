function [O, Ostartend] = markov(D,O,L,Ostartend,loss)
addpath 'D:\Personal\Desktop\ldl\LDL-master\LDLPackage_v1.2\measures'
levels = size(O,2);
len = size(O,1);
Olist = [];
for i=1:levels-1
    Olist = reshape(O,1,len,[]);
    startendlist = Ostartend;
    start = Ostartend(i);
    mid = Ostartend(i+1)-1;
    mid2 = Ostartend(i+1);
    last = Ostartend(i+2)-1;
    if mid>start+1
        [Onew,Ostartendnew] = transfer(O(:,i:i+1),Ostartend,i,'left');
        startendlist = [startendlist;Ostartendnew];
        temp = O;
        temp(:,i:i+1)=Onew;
        Olist = [Olist;reshape(temp,1,len,[])];
    end
    if mid2+1<last
        [Onew,Ostartendnew] = transfer(O(:,i:i+1),Ostartend,i,'right');
        startendlist = [startendlist;Ostartendnew];
        temp = O;
        temp(:,i:i+1)=Onew;
        Olist = [Olist;reshape(temp,1,len,[])];
    end
    probs = zeros(1,size(startendlist,1));
    for j=1:size(startendlist,1)
        predict = D*reshape(Olist(j,:,:),len,[]);
        gt = L;
        probs(j) = sum(sum(abs(gt-predict)));
        %probs(j) = 1.0/kldiv(normalize(D*reshape(Olist(j,:,:),len,2)),normalize(L(:,j:j+1)));
    end
    index = find(probs==min(probs));
    O = reshape(Olist(index,:,:),len,[]);
    Ostartend = startendlist(index,:);

end
end


function [O,Ostartend]=transfer(O,Ostartend,i,dirction)
    mid1 = Ostartend(i + 1)-1;
    mid2 = mid1+1;
    if  strcmp(dirction,'left')
        O(mid1, 1) = 0;
        O(mid1, 2) = 1;
        Ostartend(i+1) = Ostartend(i+1)-1;
    end
    if strcmp(dirction,'right')
        O(mid2, 1) = 1;
        O(mid2, 2) = 0;
        Ostartend(i+1) = Ostartend(i+1)+1;
    end
end