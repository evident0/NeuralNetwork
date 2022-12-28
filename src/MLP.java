import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class MLP {
    String activationFunction;
    double lowerBound;
    double learningRate;

    int numberOfInputs;
    int numberOfOutputs;

    int batchSize;

    double[][][] layerWeights;
    double[][] layerBiases;

    double[][] layerOutputs;//TODO: this should probably not be a class variable

    double[][][] layerWeightsHistory;
    double[][] layerBiasesHistory;

    //create a constructor that takes in the number of inputs, number of hidden nodes , and number of outputs
    public MLP(int numberOfInputs, int[] layers, String activationFunction, double learningRate, double lowerBound){//activation function is relu or tanH
/*
        if(activationFunction.equals("relu")){
            this.activationFunction = "relu";
        }
        else if(activationFunction.equals("tanH")){
            this.activationFunction = "tanH";
        }
        else{
            System.out.println("Invalid activation function");
        }
*/
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        this.lowerBound = lowerBound;
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputs = layers[layers.length-1];


        //create and initialize the weights and biases to a random number between -1 and 1
        Random random = new Random();
        double max = 1;
        double min = -1;

        //create the weights and biases for the hidden layers
        double[][][] weights = new double[layers.length][][];
        double[][] biases = new double[layers.length][];

        for(int i = 0; i < layers.length; i++){
            weights[i] = new double[layers[i]][];
            biases[i] = new double[layers[i]];
            for(int j = 0; j < layers[i]; j++){
                weights[i][j] = new double[i == 0 ? numberOfInputs : layers[i - 1]];
                for(int k = 0; k < weights[i][j].length; k++){
                    weights[i][j][k] = min + (max - min) * random.nextDouble();
                }
                biases[i][j] = 0.0;//min + (max - min) * random.nextDouble();
            }
        }
        this.layerWeights = weights;
        this.layerBiases = biases;
        //create deep copies of the weights and biases for the history


        this.layerWeightsHistory = Arrays.copyOf(weights, weights.length);
        this.layerBiasesHistory = Arrays.copyOf(biases, biases.length);

    }

    //create a method for forward propagation
    public double[] forwardPropagation(double[] input){

        //check if input size matches the number of inputs
        if(input.length != numberOfInputs){
            System.out.println("Invalid input size");
            return null;
        }

        //create the output array
        double[] output = new double[numberOfOutputs];

        //create the hidden layer outputs
        double[][] layerOutputs = new double[layerWeights.length][];

        for(int layer = 0; layer < layerWeights.length; layer++){
            layerOutputs[layer] = new double[layerWeights[layer].length];
            for(int node = 0; node < layerWeights[layer].length; node++){
                double sum = 0;
                for(int weight = 0; weight < layerWeights[layer][node].length; weight++){
                    sum += layerWeights[layer][node][weight] * (layer == 0 ? input[weight] : layerOutputs[layer - 1][weight]);
                }
                sum += layerBiases[layer][node];
                if(layer == layerWeights.length-1){//output layer

                    output[node] = activationFunction("sigmoid", sum);//sigmoid(sum);
                    layerOutputs[layer][node] = output[node];

                }
                else{

                    layerOutputs[layer][node] = activationFunction(activationFunction, sum);

                }
                /*
                else if(activationFunction.equals("relu")){
                    layerOutputs[layer][node] = relu(sum);
                }
                else if(activationFunction.equals("tanH")){
                    layerOutputs[layer][node] = tanH(sum);
                }*/
            }
        }
        this.layerOutputs = layerOutputs;
        return output;

    }

    public void backPropagation(double[] input, double[] expectedOutput)
    {
        if(input.length != numberOfInputs){
            System.out.println("Invalid input size");
            return;
        }
        if(expectedOutput.length != numberOfOutputs){
            System.out.println("Invalid output size");
            return;
        }

        forwardPropagation(input);

        //calculate the error for the output layer
        double[][] outputLayerErrors = new double[layerWeights.length][];
        for(int i = 0; i < outputLayerErrors.length; i++){
            outputLayerErrors[i] = new double[layerWeights[i].length];
        }

        for(int out = 0; out < outputLayerErrors[outputLayerErrors.length-1].length; out++){
            // the below calculates delta = (y - t) * f'(z(i))
            outputLayerErrors[outputLayerErrors.length-1][out] = squaredErrorDerivative(layerOutputs[layerOutputs.length-1][out], expectedOutput[out])
                    * activationFunctionDerivative("sigmoid", layerOutputs[layerOutputs.length-1][out]);
            //sigmoidDerivative(layerOutputs[layerOutputs.length-1][out]);

        }

        //calculate the error for the hidden layers
        for(int i = outputLayerErrors.length-2; i >= 0; i--){
            //the below calculates delta(i) = delta(i+1) * w(i+1) * f'(z(i))
            for(int j = 0; j < outputLayerErrors[i].length; j++){
                double sum = 0;
                for(int k = 0; k < outputLayerErrors[i+1].length; k++){
                    sum += outputLayerErrors[i+1][k] * layerWeights[i+1][k][j];
                }
                outputLayerErrors[i][j] = sum * activationFunctionDerivative(activationFunction, layerOutputs[i][j]);
                /*
                if(activationFunction.equals("relu")){
                    outputLayerErrors[i][j] = sum * reluDerivative(layerOutputs[i][j]);
                }
                else if(activationFunction.equals("tanH")){
                    outputLayerErrors[i][j] = sum * tanHDerivative(layerOutputs[i][j]);
                }*/
            }
        }

        //save the weights and biases to update them at the end of the epoch
        for(int i = 0; i < layerWeightsHistory.length; i++){
            for(int j = 0; j < layerWeightsHistory[i].length; j++){
                for(int k = 0; k < layerWeightsHistory[i][j].length; k++){
                    layerWeightsHistory[i][j][k] -= learningRate * outputLayerErrors[i][j] * (i == 0 ? input[k] : layerOutputs[i-1][k]); // h * delta * output(i-1)
                }
                layerBiasesHistory[i][j] -= learningRate * outputLayerErrors[i][j];
            }
        }

    }

    private void updateWeightsAndBiases() {
        for(int i = 0; i < layerWeights.length; i++){
            for(int j = 0; j < layerWeights[i].length; j++){
                for(int k = 0; k < layerWeights[i][j].length; k++){
                    layerWeights[i][j][k] = layerWeightsHistory[i][j][k]; /// numberOfExamplesInBatch;
                }
                layerBiases[i][j] = layerBiasesHistory[i][j]; /// numberOfExamplesInBatch;
            }
        }
    }

    private double activationFunction(String function, double x){
        if (function.equals("sigmoid")){
            return sigmoid(x);
        }
        else if (function.equals("relu")){
            return relu(x);
        }
        else if (function.equals("tanH")){
            return tanH(x);
        }
        System.out.println("Invalid activation function defaulting to sigmoid");
        return  sigmoid(x);
    }

    private double activationFunctionDerivative(String function, double x){
        if (function.equals("sigmoid")){
            return sigmoidDerivative(x);
        }
        else if (function.equals("relu")){
            return reluDerivative(x);
        }
        else if (function.equals("tanH")){
            return tanHDerivative(x);
        }
        System.out.println("Invalid activation function defaulting to sigmoid derivative");
        return  sigmoidDerivative(x);
    }

    private ArrayList<ArrayList<Example>> createBatches(ArrayList<Example> examples, int batchSize){
        //suffle the examples list
        Collections.shuffle(examples);
        //split the examples into batches
        ArrayList<ArrayList<Example>> batches = new ArrayList<>();
        for(int i = 0; i < examples.size(); i += batchSize){
            batches.add(new ArrayList<>(examples.subList(i, Math.min(i + batchSize, examples.size()))));
        }
        return batches;
    }

    private double[] oneHotEncode(Category category){

        if(category.equals(Category.C1)){
            return new double[] {1, 0, 0};
        }
        else if(category.equals(Category.C2)){
            return new double[]  {0, 1, 0};
        }
        else if(category.equals(Category.C3)){
            return new double[]  {0, 0, 1};
        }
        else{

            System.err.println("Invalid category");
            return new double[]  {0, 0, 0};

        }
    }

    //TODO one epoch == examples/batch_size number of iterations
    public ArrayList<Double> trainBatch(ArrayList<Example> exampleList, int epochs, int batchCount){

        ArrayList<Double> errorList = new ArrayList<>();
        int batchSize = exampleList.size()/batchCount;

        //for number of epochs
        for(int i = 0; i < epochs; i++){

            ArrayList<ArrayList<Example>> batches = new ArrayList<>();
            batches = createBatches(exampleList, batchCount);
            double error = 0;

            for(ArrayList<Example> batch: batches){
                for(Example example: batch){

                    Category category = example.category;
                    double[] input = {example.x1, example.x2};
                    double[] expectedOutput = oneHotEncode(category);
                    backPropagation(input, expectedOutput);

                }

                updateWeightsAndBiases();

                for(Example example: batch){

                    Category category = example.category;
                    double[] input = {example.x1, example.x2};
                    double[] expectedOutput = oneHotEncode(category);
                    forwardPropagation(input);
                    error += calculateMeanSquaredErrorForOutputs(expectedOutput);
                }

                error = error/batchSize;
                errorList.add(error);
                System.out.println("Epoch: " + i + " with Error: "+ error);

            }
        }
        return errorList;
        /*
        ArrayList<Double> errorList = new ArrayList<Double>();
        int batchSize = exampleList.size()/batchCount;
        this.batchSize = batchSize;
        int step = 0;
        for(int i = 0;; i++){
            if(step == exampleList.size()){
                step = 0;
            }

            for(int j = step; j < step + batchSize && j < exampleList.size(); j++){
                Example example = exampleList.get(j);

                Category category = example.category;
                double[] input = {example.x1, example.x2};

                if(category.equals(Category.C1)){
                    double[] expectedOutput = new double[] {1, 0, 0};
                    backPropagation(input, expectedOutput);
                }
                else if(category.equals(Category.C2)){
                    double[] expectedOutput = new double[]  {0, 1, 0};
                    backPropagation(input, expectedOutput);
                }
                else if(category.equals(Category.C3)){
                    double[] expectedOutput = new double[]  {0, 0, 1};
                    backPropagation(input, expectedOutput);
                }
            }

            updateWeightsAndBiases();

            double sum = 0;
            for(int j = step; j < step + batchSize && j < exampleList.size(); j++){

                Example example = exampleList.get(j);

                Category category = example.category;
                double[] input = {example.x1, example.x2};

                if(category.equals(Category.C1)){
                    double[] expectedOutput = new double[] {1, 0, 0};
                    forwardPropagation(input);
                    sum += calculateMeanSquaredErrorForOutputs(expectedOutput);
                }
                else if(category.equals(Category.C2)){
                    double[] expectedOutput = new double[]  {0, 1, 0};
                    forwardPropagation(input);
                    sum += calculateMeanSquaredErrorForOutputs(expectedOutput);
                }
                else if(category.equals(Category.C3)){
                    double[] expectedOutput = new double[]  {0, 0, 1};
                    forwardPropagation(input);
                    sum += calculateMeanSquaredErrorForOutputs(expectedOutput);
                }
            }

            double err = sum/batchSize;
            errorList.add(err);
            System.out.println("Epoch: " + i + " with Error: "+ err);

            if (i >= epochs-1 && err <= lowerBound){

                break;

            }

            step += batchSize;

        }
        return errorList;*/
    }

    public void testMLP(ArrayList<Example> testExampleList)
    {
        int correct = 0;
        for(int i = 0; i < testExampleList.size(); i++){
            Example example = testExampleList.get(i);
            Category category = example.category;
            double[] input = {example.x1, example.x2};
            forwardPropagation(input);
            double[] output = layerOutputs[layerOutputs.length-1];
            double max = 0;
            int maxIndex = 0;
            // we choose the maximum regardless of the confidence level
            // although we could choose the maximum only if it is above a certain threshold
            // the accuracy would not change much
            for(int j = 0; j < output.length; j++){
                if(output[j] > max ){//testing only --> && output[j] >= 0.9){
                    max = output[j];
                    maxIndex = j;
                }
            }
            if(maxIndex == 0 && category.equals(Category.C1)){
                correct++;
                example.setIsCorrect(true);
                System.out.println("(+) " + example.x1 + " " + example.x2 + " " + category);
            }
            else if(maxIndex == 1 && category.equals(Category.C2)){
                correct++;
                example.setIsCorrect(true);
                System.out.println("(+) " + example.x1 + " " + example.x2 + " " + category);
            }
            else if(maxIndex == 2 && category.equals(Category.C3)){
                correct++;
                example.setIsCorrect(true);
                System.out.println("(+) " + example.x1 + " " + example.x2 + " " + category);
            }
            else{
                example.setIsCorrect(false);
                System.out.println("(-) " + example.x1 + " " + example.x2 + " guess: " +
                        (((maxIndex == 0) ? Category.C1 : (maxIndex == 1) ? Category.C2 : Category.C3) + " actual: " + category));
            }
        }
        System.out.println("Accuracy: " + (double)correct/testExampleList.size());
    }

    public double calculateMeanSquaredErrorForOutputs(double[] expectedOutput)
    {
        double sum = 0;
        for(int i = 0; i < layerOutputs[layerOutputs.length-1].length; i++){
            sum += Math.pow(expectedOutput[i] - layerOutputs[layerOutputs.length-1][i], 2);
        }
        return 1.0/2.0 * sum/layerOutputs[layerOutputs.length-1].length;
    }



    //relu function
    public static double relu(double x){
        return Math.max(0, x);
    }
    //tanH function
    public static double tanH(double x){
        return Math.tanh(x);
    }

    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    double reluDerivative(double x){
        if(x > 0){
            return 1;
        }
        else{
            return 0;
        }
    }

    double tanHDerivative(double x){
        return 1 - Math.pow(Math.tanh(x), 2);
    }

    double sigmoidDerivative(double x){
        return x * (1 - x);// out * (1 - out)
    }

    double squaredErrorDerivative(double actual, double expected){
        return (actual - expected);
    }


    public static void main(String[] args) {
        //outputs don't really have weights therefore they are not included in the layerWeights array
        //note tanh is slower than relu
        //the output layer uses the sigmoid activation function
        MLP mlp = new MLP( 2, new int[]{10,10,10,3}, "relu", 0.001, 0.01);

        DataSet dataSet = new DataSet();
        dataSet.createExamples(4000, 4000);

        mlp.testMLP(dataSet.testExamples);
        ArrayList<Double> errorList = mlp.trainBatch(dataSet.learningExamples, 700, 100);
        FileManager.writeArrayToFile(errorList, "errorList.txt");
        mlp.testMLP(dataSet.testExamples);
        FileManager.writeArrayToFile(dataSet.testExamples, "testDatasetResults.txt");

        //mlp.testMLP(dataSet.learningExamples);
        FileManager.writeArrayToFile(dataSet.learningExamples, "trainDatasetResults.txt");

    }
}
