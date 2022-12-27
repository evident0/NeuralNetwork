clear all, fclose all

args = argv();
printf("args = %s\n", args{1});


fig = figure();
fid = fopen('dataSet.txt');
data = fscanf(fid, '%f ,%f', [2 1200]);
fclose(fid);
xAxis =[];
yAxis = [];
for i=1:2:2400
  xAxis = [xAxis; data(i)];
  yAxis = [yAxis; data(i+1)];
endfor
scatter(xAxis,yAxis,'+');
hold on;

fid = fopen('centers.txt');
centerNum = fscanf(fid, '%d',[1]);
centerNum;
centers =  fscanf(fid, '%f ,%f', [2 centerNum]);
fclose(fid);
xCenters = [];
yCenters = [];
for j=1:2:centerNum*2
  xCenters = [xCenters; centers(j)];
  yCenters = [yCenters; centers(j+1)];
endfor
scatter(xCenters,yCenters,'r',"filled");


saveas(fig,args{1});
