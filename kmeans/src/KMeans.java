import java.util.Random;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class KMeans {

    private DataSet dataSet;
    private int centers;
    private double[][] currentCenters;
    private Random random = new Random();
    private double euclDistance = 0;

    public KMeans(int centers, String dataset) {
        this.centers = centers;
        dataSet = new DataSet(dataset);
        currentCenters = new double[centers][2];
        for(int i=0;i<centers;i++){
            int randomExampleIndex = random.nextInt(1200);
            Example randomExample = dataSet.getExamples().get(randomExampleIndex);
            currentCenters[i][0] = randomExample.x1;
            currentCenters[i][1] = randomExample.x2;
        }
    }

    public double[][] getCenters(){
        return currentCenters;
    }

    public double getEuclDistance(){
        return euclDistance;
    }

    public double[] updateCenters(){
        ArrayList<ArrayList<Example>> groups = new ArrayList<ArrayList<Example>>();
        double[] returnDistances = new double[centers];
        for(int i=0; i<centers; i++){ //setup groups
            groups.add(new ArrayList<Example>());
        }

        for(Example element : dataSet.getExamples()){ //for each point
            double minDistance = 1000000.0;
            int distIndex = -1;
            double[] point = {element.x1,element.x2};
            for(int i=0;i<centers;i++){ //find the closest center to point
                double curDistance = distance(point,currentCenters[i]);
                if(minDistance>=curDistance){
                    minDistance = curDistance;
                    distIndex = i;
                }
            }
            groups.get(distIndex).add(element);//add point to closest center group
        }

        //calculate new centers per group
        for(int i=0;i<centers;i++){
            double[] newCenter = {0,0};
            for(Example element: groups.get(i)){
                newCenter[0] += element.x1;
                newCenter[1] += element.x2;
            }
            if(groups.get(i).size()!=0){//algorithm edge case: triggers if center gets no points assigned to it
                newCenter[0] = newCenter[0]/groups.get(i).size();//avg x
                newCenter[1] = newCenter[1]/groups.get(i).size();//avg y
                
            } else {
                newCenter[0] = currentCenters[i][0];
                newCenter[1] = currentCenters[i][1];
            } //usually happens when bad random centers get first assigned
            //in edge case, keep last center
                returnDistances[i] = distance(newCenter,currentCenters[i]);
                currentCenters[i][0] = newCenter[0];
                currentCenters[i][1] = newCenter[1];
        }

        //calculate euclidian distance
        euclDistance = 0;
        for(int i=0;i<centers;i++){
            for(Example element: groups.get(i)){
                double[] point = {element.x1, element.x2};
                euclDistance+=distance(point,currentCenters[i]);
            }
        }
        return returnDistances; //stop contition for while loop
    }

    private double distance(double[] point1, double[] point2){
        double distance =  Math.sqrt(Math.abs(Math.pow((point1[0]-point2[0]),2)+Math.pow((point1[1]-point2[1]),2)));
        return distance;
    }


    public static void main(String args[]){ 

        //presets
        int centerNum = 3;
        int maxCycles = 1000;
        int cyclesLeft = 0;
        int accuracy = 5;
        boolean prnt = false;
        int iterations = 15;
        String dataset = "dataSet.txt";
        String dest = "centers.txt";
        double[][] results;
        double[][] bestResults;
        double minEucl = 10000000;
        

        //parse inputs
        ArrayList<String> arguments = new ArrayList<String>(Arrays.asList(args));  
        try{
            if(arguments.contains("-print")){ 
                prnt = true;
            }
            if(arguments.contains("-input")){ 
                dataset = arguments.get(arguments.indexOf("-input")+1);
            }
            if(arguments.contains("-output")){ 
                dest = arguments.get(arguments.indexOf("-output")+1);
            }
            if(arguments.contains("-iterations")){ 
                iterations = Integer.parseInt(arguments.get(arguments.indexOf("-iterations")+1));
            }
            if(arguments.contains("-centers")){
                centerNum = Integer.parseInt(arguments.get(arguments.indexOf("-centers")+1));
            }
            if(arguments.contains("-maxCycles")){
                centerNum = Integer.parseInt(arguments.get(arguments.indexOf("-maxCycles")+1));
            }
            if(arguments.contains("-accuracy")){
                accuracy = Integer.parseInt(arguments.get(arguments.indexOf("-accuracy")+1));
            }
        } catch(Exception e){
            System.out.println("Wrong arguments, using presets");
        }
        cyclesLeft = maxCycles;
        results = new double[centerNum][2];
        bestResults = results;
        




        //Start k-means iterations
        for(int it=0; it<iterations;it++){ //preset 15 iterations per test
            if(prnt){
                System.out.println("Current Repetition: "+it);
            }
            cyclesLeft = maxCycles;
            KMeans kmeans = new KMeans(centerNum,dataset);
            double avgDistance = 1;
            while(avgDistance>Math.pow(10,-1*accuracy) && cyclesLeft>0){ //preset accuracy 10^-5 and maxCycles=1000
                avgDistance = 0;
                double[] distances = kmeans.updateCenters(); //run a k-means cycle
                for(int i=0;i<distances.length;i++){ //calculate avg distance (stop condition)
                    avgDistance+=distances[i];
                }
                
                avgDistance/=distances.length;
                results = kmeans.getCenters(); //get new  calculated centers
                if(prnt){ //prints progress only with appropriate argument
                    System.out.println("Average distance change: "+ avgDistance);
                    System.out.println("Iterations Done: "+ (maxCycles - cyclesLeft));
                    System.out.println("Updated Centers:");
                    for(int j=0; j<centerNum; j++){
                        System.out.println("Center "+(j+1)+":"+results[j][0]+ "," +results[j][1]);
                    }
                }
                cyclesLeft--;
            }
            double newEucl = kmeans.getEuclDistance(); //calculate distance
            if(newEucl<minEucl){ //keep best distance found from all iterations
                minEucl = newEucl;
                bestResults = results;
            }
        }


        
        //Export/print outputs
        try {
            File output = new File(dest);
            output.createNewFile();
        } catch (IOException e) {
            System.out.println("Error printing output file");
        }

        
        System.out.println("Total Euclidean Distance: "+ minEucl);
        System.out.println("Final Centers:");

        try { //centers saved on centers.txt
            FileWriter writer = new FileWriter(dest);
            writer.write(centerNum+"\n");
            for(int i=0;i<centerNum;i++){
                writer.write(bestResults[i][0]+ "," +bestResults[i][1]+"\n");
                System.out.println(bestResults[i][0]+ "," +bestResults[i][1]);
            }
            writer.close();
        } catch (IOException e) {
            System.out.println("Error printing output file");
        }
    }
}