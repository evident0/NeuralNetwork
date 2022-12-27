#javac -d ./bin ./src/*.java

cd ./bin
java DataSet ../dataSet.txt

for i in 3 6 9 12
do
    java KMeans -centers $i -iterations 15 -input ../dataSet.txt -output ../centers.txt
    cd ..
    octave myPlot.m ./results/centers$i.jpg
    cd ./bin
done
