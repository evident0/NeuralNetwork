import java.util.List;

public class Driver {

    static double [][] X= {
            {0,0},
            {1,0},
            {0,1},
            {1,1}
    };
    static double [][] Y= {
            {0},{1},{1},{0}
    };

    public static void main(String[] args) {

        NeuralNetwork nn = new NeuralNetwork(2,100,3);


        double testX = -0.6;
        double testY = 0.7;

        double testX2 = 0.9;
        double testY2 = -0.87;

        double testX3 = 0.5;
        double testY3 = -0.7;

        double testX4 = -0.5;
        double testY4 = -0.7;

       // double testX2 = -0.5;
       // double testY2 = 0.8;

        DataSet dataSet = new DataSet();
        dataSet.createExamples(8000, 8000);
        System.out.println(dataSet.categorizeExample(testX,testY));
        System.out.println(dataSet.categorizeExample(testX2,testY2));
        System.out.println(dataSet.categorizeExample(testX3,testY3));
        System.out.println(dataSet.categorizeExample(testX4,testY4));
        double [][] IN = new double[dataSet.learningExamples.size()][2];
        double [][] OUT = new double[dataSet.learningExamples.size()][3];
        for(int i = 0; i < dataSet.learningExamples.size(); i++){
            IN[i][0] = dataSet.learningExamples.get(i).x1;
            IN[i][1] = dataSet.learningExamples.get(i).x2;
            OUT[i][0] = dataSet.learningExamples.get(i).category == Category.C1 ? 1 : 0;
            OUT[i][1] = dataSet.learningExamples.get(i).category == Category.C2 ? 1 : 0;
            OUT[i][2] = dataSet.learningExamples.get(i).category == Category.C3 ? 1 : 0;
        }



        List<Double>output;

        nn.fit(IN, OUT, 1000000);
        double [][] input = {
                {testX,testY}, {testX2,testY2}, {testX3,testY3}, {testX4,testY4}
        };
        for(double d[]:input)
        {
            output = nn.predict(d);
            System.out.println(output.toString());
        }

    }

}