import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class MLP {
    String activationFunction;
    double lowerBound;
    double learningRate;

    int numberOfInputs;
    int numberOfOutputs;

    double[][][] layerWeights;
    double[][] layerBiases;

    double[][] layerOutputs;

    double[][][] layerWeightsHistory;
    double[][] layerBiasesHistory;

    //relu function
    public static double relu(double x){
        return Math.max(0, x);
    }
    //tanH function
    public static double tanH(double x){
        return Math.tanh(x);
    }

    //create a constructor that takes in the number of inputs, number of hidden nodes , and number of outputs
    public MLP(int numberOfInputs, int[] layers, String activationFunction, double learningRate, double lowerBound){//activation function is relu or tanH

        if(activationFunction.equals("relu")){
            this.activationFunction = "relu";
        }
        else if(activationFunction.equals("tanH")){
            this.activationFunction = "tanH";
        }
        else{
            System.out.println("Invalid activation function");
        }

        this.learningRate = learningRate;
        this.lowerBound = lowerBound;
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputs = layers[layers.length-1];


        //create and initialize the weights and biases to a random number between -1 and 1
        Random random = new Random(348758934);
        double max = 1.0;
        double min = -1.0;

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
                biases[i][j] = min + (max - min) * random.nextDouble();
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

        for(int layer = 0; layer < layerWeights.length; layer++){//how many layers is hiddenlayerweights
            layerOutputs[layer] = new double[layerWeights[layer].length];
            for(int node = 0; node < layerWeights[layer].length; node++){//how many nodes in each layer
                double sum = 0;
                for(int weight = 0; weight < layerWeights[layer][node].length; weight++){//how many weights in each node
                    sum += layerWeights[layer][node][weight] * (layer == 0 ? input[weight] : layerOutputs[layer - 1][weight]);//go to the previous layer and get the output for node k
                }
                sum += layerBiases[layer][node];
                if(layer == layerWeights.length-1){

                    output[node] = sigmoid(sum);
                    layerOutputs[layer][node] = output[node];

                }
                else if(activationFunction.equals("relu")){
                    layerOutputs[layer][node] = relu(sum);
                }
                else if(activationFunction.equals("tanH")){
                    layerOutputs[layer][node] = tanH(sum);
                }
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
            outputLayerErrors[outputLayerErrors.length-1][out] = squaredErrorDerivative(layerOutputs[layerOutputs.length-1][out], expectedOutput[out])
                    * sigmoidDerivative(layerOutputs[layerOutputs.length-1][out]); //* layerOutputs[layerOutputs.length-2][out];
            //thetaE/thetaw = (expected - actual) * sigmoidDerivative(actual) where sigmoidDerivative = (actual(1-actual))
        }

        //calculate the error for the hidden layers
        for(int i = outputLayerErrors.length-2; i >= 0; i--){
            for(int j = 0; j < outputLayerErrors[i].length; j++){
                double sum = 0;
                for(int k = 0; k < outputLayerErrors[i+1].length; k++){
                    sum += outputLayerErrors[i+1][k] * layerWeights[i+1][k][j];
                }
                if(activationFunction.equals("relu")){
                    outputLayerErrors[i][j] = sum * reluDerivative(layerOutputs[i][j]);
                }
                else if(activationFunction.equals("tanH")){
                    outputLayerErrors[i][j] = sum * tanHDerivative(layerOutputs[i][j]);
                }
            }
        }

        //save the weights and biases to update them at the end of the epoch
        for(int i = 0; i < layerWeightsHistory.length; i++){
            for(int j = 0; j < layerWeightsHistory[i].length; j++){
                for(int k = 0; k < layerWeightsHistory[i][j].length; k++){
                    layerWeightsHistory[i][j][k] += learningRate * outputLayerErrors[i][j] * (i == 0 ? input[k] : layerOutputs[i-1][k]);
                }
                layerBiasesHistory[i][j] += learningRate * outputLayerErrors[i][j];
            }
        }

/*
        //update the weights and biases
        for(int i = 0; i < layerWeights.length; i++){
            for(int j = 0; j < layerWeights[i].length; j++){
                for(int k = 0; k < layerWeights[i][j].length; k++){
                    layerWeights[i][j][k] += learningRate * outputLayerErrors[i][j] * (i == 0 ? input[k] : layerOutputs[i-1][k]);
                }
                layerBiases[i][j] += learningRate * outputLayerErrors[i][j];
            }
        }
*/
    }
    //train the network
    public void train(ArrayList<Example> exampleList, int epochs){
        for(int i = 0; i < epochs; i++){
            for(int j = 0; j < exampleList.size(); j++){
                Example example = exampleList.get(j);
                Category category = example.category;
                double[] input = {example.x1, example.x2};

                if(category.equals(Category.C1)){
                    double[] expectedOutput = {1, 0, 0};
                    backPropagation(input, expectedOutput);
                }
                else if(category.equals(Category.C2)){
                    double[] expectedOutput = {0, 1, 0};
                    backPropagation(input, expectedOutput);
                }
                else if(category.equals(Category.C3)){
                    double[] expectedOutput = {0, 0, 1};
                    backPropagation(input, expectedOutput);
                }
            }
        }
    }

    public void trainBatch(ArrayList<Example> exampleList, int epochs, int batchSize){


       // int batchSize = exampleList.size()/batchNumber;
        int step = 0;
        for(int i = 0; i < epochs; i++){
            //cheat
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

            updateWeightsAndBiases(exampleList.size()/batchSize);

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
            System.out.println("Epoch: " + i + " with Error: "+ err);


            step += batchSize;

        }
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
            for(int j = 0; j < output.length; j++){
                if(output[j] > max ){//testing only && output[j] >= 0.9){
                    max = output[j];
                    maxIndex = j;
                }
            }
            if(maxIndex == 0 && category.equals(Category.C1)){
                correct++;
            }
            else if(maxIndex == 1 && category.equals(Category.C2)){
                correct++;
            }
            else if(maxIndex == 2 && category.equals(Category.C3)){
                correct++;
            }
        }
        System.out.println("Accuracy: " + (double)correct/testExampleList.size());
    }

    public double calculateMeanSquaredErrorForOutputs(double[] expectedOutput)
    {
        double sum = 0;
        for(int i = 0; i < layerOutputs[layerOutputs.length-1].length; i++){
            sum += Math.pow(expectedOutput[i] - layerOutputs[layerOutputs.length-1][i], 2);
            //System.out.println("diff:  " + (layerOutputs[layerOutputs.length-1][i] - expectedOutput[i]));
        }
        return 1.0/2.0 * sum/layerOutputs[layerOutputs.length-1].length;
    }


    private void updateWeightsAndBiases(int numberOfExamplesInBatch) {
        for(int i = 0; i < layerWeights.length; i++){
            for(int j = 0; j < layerWeights[i].length; j++){
                for(int k = 0; k < layerWeights[i][j].length; k++){
                    layerWeights[i][j][k] = layerWeightsHistory[i][j][k]; /// numberOfExamplesInBatch;
                }
                layerBiases[i][j] = layerBiasesHistory[i][j]; /// numberOfExamplesInBatch;
            }
        }
    }


    //train with batches

    double squaredErrorDerivative(double actual, double expected){
        return (expected - actual);
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

    // sigmoid function
    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public static void main(String[] args) {
        MLP mlp = new MLP( 2, new int[]{20,20,20,3}, "relu", 0.01, 0.1);

        //this is not permanent, just for testing
        double testX = 0.5;
        double testY = 0.7;

        double testX1 = -0.6;
        double testY1 = 0.7;

        double testX2 = 0.9;
        double testY2 = -0.87;

        double testX3 = 0.5;
        double testY3 = -0.7;

        double testX4 = -0.5;
        double testY4 = -0.7;

        DataSet dataSet = new DataSet();
        dataSet.createExamples(4000, 4000);
        System.out.println(dataSet.categorizeExample(testX,testY));
        System.out.println(dataSet.categorizeExample(testX1,testY1));
        System.out.println(dataSet.categorizeExample(testX2,testY2));
        System.out.println(dataSet.categorizeExample(testX3,testY3));
        System.out.println(dataSet.categorizeExample(testX4,testY4));
        mlp.testMLP(dataSet.testExamples);
        mlp.trainBatch(dataSet.learningExamples, 1000, 400);
        mlp.testMLP(dataSet.testExamples);
        mlp.testMLP(dataSet.learningExamples);

        double[] output2 = mlp.forwardPropagation(new double[]{testX, testY});

        for(int i = 0; i < output2.length; i++){
            System.out.println(output2[i]);
        }
        System.out.println();
        double[] output3 = mlp.forwardPropagation(new double[]{testX1, testY1});

        for(int i = 0; i < output3.length; i++){
            System.out.println(output3[i]);
        }
        System.out.println();
        double[] output4 = mlp.forwardPropagation(new double[]{testX2, testY2});

        for(int i = 0; i < output4.length; i++){
            System.out.println(output4[i]);
        }
        System.out.println();
        double[] output5 = mlp.forwardPropagation(new double[]{testX3, testY3});

        for(int i = 0; i < output5.length; i++){
            System.out.println(output5[i]);
        }
        System.out.println();
        double[] output6 = mlp.forwardPropagation(new double[]{testX4, testY4});

        for(int i = 0; i < output6.length; i++){
            System.out.println(output6[i]);
        }

    }
}
