import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileNotFoundException;


public class DataSet {

    //create an array list of tuples
    private ArrayList<Example> examples = new ArrayList<>();
    private double[][] examplePoints= {
    //   num  minX maxX minY maxY
        {150, 0.8, 1.2, 0.8, 1.2},
        {150, 0.0, 0.5, 0.0, 0.5},
        {150, 0.0, 0.5, 1.5, 2.0},
        {150, 1.5, 2.0, 0.0, 0.5},
        {150, 1.5, 2.0, 1.5, 2.0},
        {75,  0.8, 1.2, 0.0, 0.4},
        {75,  0.8, 1.2, 1.6, 2.0},
        {75,  0.3, 0.7, 0.8, 1.2},
        {75,  1.3, 1.7, 0.8, 1.2},
        {150, 0.0, 2.0, 0.0, 2.0}
    };


    public DataSet(){
        Random random = new Random(348758934);
        for(int i=0; i<examplePoints.length; i++){
            createExamples((int)examplePoints[i][0], random, examples, examplePoints[i][1], examplePoints[i][2], examplePoints[i][3], examplePoints[i][4]);
        }
    }

    public DataSet(ArrayList<Example> examples){
        this.examples = examples;
    }

    public DataSet(String filepath){
        try {
            File input = new File(filepath);
            Scanner reader = new Scanner(input);
            while (reader.hasNextLine()) {
              String data = reader.nextLine();
              String[] split = data.split(",");
              examples.add(new Example(Double.parseDouble(split[0]),Double.parseDouble(split[1])));
            }
            reader.close();
        } catch (FileNotFoundException e) {
            System.out.println("File not found");
            e.printStackTrace();
        }
    }

    private void createExamples(int examplesNumber, Random random, ArrayList<Example> examples, double minX, double maxX, double minY, double maxY) {
        for(int i = 0; i < examplesNumber; i++){

            double x1 = minX + (maxX - minX) * random.nextDouble();
            double x2 = minY + (maxY - minY) * random.nextDouble();
            examples.add(new Example(x1, x2));
        }
    }
    
    public void exportDataSet(String filename){
        try {
            File newFile = new File(filename);
            newFile.createNewFile();
        } catch (IOException e) {
            System.out.println("Error while creating file");
            e.printStackTrace();
        }

        try {
            FileWriter writer = new FileWriter(filename);
            for(int i=0; i<examples.size(); i++){
                writer.write(examples.get(i).x1+","+ examples.get(i).x2+"\n");
            }
            writer.close();
          } catch (IOException e) {
            System.out.println("Error while writing to file");
            e.printStackTrace();
          }
    }

    public ArrayList<Example> getExamples(){
        return examples;
    }

    public static void main(String[] args) {
        DataSet dataSet = new DataSet();
        dataSet.exportDataSet(args[0]);
    }
}
