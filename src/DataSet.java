import java.util.ArrayList;
import java.util.Random;


public class DataSet {

    //create an array list of tuples
    ArrayList<Example> learningExamples = new ArrayList<>();
    ArrayList<Example> testExamples = new ArrayList<>();

    public void createExamples(int learningExamplesNumber, int testExamplesNumber){
        Random random = new Random(348758934);

        createExamples(learningExamplesNumber, random, learningExamples);
        createExamples(testExamplesNumber, random, testExamples);

    }

    private void createExamples(int learningExamplesNumber, Random random, ArrayList<Example> examples) {
        for(int i = 0; i < learningExamplesNumber; i++){

            double max = 1.0;
            double min = -1.0;
            double x1 = min + (max - min) * random.nextDouble();
            double x2 = min + (max - min) * random.nextDouble();
            examples.add(new Example(x1, x2, categorizeExample(x1, x2)));

        }
    }

    public Category categorizeExample(double x1, double x2)
    {

            if((Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2)) < 0.2 && x2 > 0.5){
                return Category.C1;
            }
            else if((Math.pow(x1 - 0.5, 2) + Math.pow(x2 - 0.5, 2)) < 0.2 && x2 < 0.5){
                return Category.C2;
            }
            else if((Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2)) < 0.2 && x2 > -0.5){
                return Category.C1;
            }
            else if((Math.pow(x1 + 0.5, 2) + Math.pow(x2 + 0.5, 2)) < 0.2 && x2 < -0.5){
                return Category.C2;
            }
            else if((Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2)) < 0.2 && x2 > -0.5) {
                return Category.C1;
            }
            else if((Math.pow(x1 - 0.5, 2) + Math.pow(x2 + 0.5, 2)) < 0.2 && x2 < -0.5){
                return Category.C2;
            }
            else if((Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2)) < 0.2 && x2 > 0.5){
                return Category.C1;
            }
            else if((Math.pow(x1 + 0.5, 2) + Math.pow(x2 - 0.5, 2)) < 0.2 && x2 < 0.5){
                return Category.C2;
            }
            else{
                return Category.C3;
            }

    }


    public static void main(String[] args) {

        DataSet dataSet = new DataSet();
        dataSet.createExamples(4000, 4000);
    }
}
