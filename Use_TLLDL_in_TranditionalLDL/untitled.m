load('D:\Personal\Desktop\ldl\LDL-master2\code\dataset\a.mat')
x = [5,14,23.5,33,43.5,55];
xstart = x-1;
points = [0.068966,0,0.344828,0.068966,0.482759,0.034483];
for i=1:6
rectangle('Position',[xstart(i) ,0 ,2 ,points(i)/1],'FaceColor','r')
end
hold on
plot(a*5)