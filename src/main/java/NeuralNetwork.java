import java.util.Arrays;
import java.util.Random;

class NeuralNetwork {
    private int nodesNumber, layersNumber;
    private double[][][] weights;
    private double[][] biases;
    private Data data;

     NeuralNetwork(int layersNumber, int nodesNumber, Data data) {
        this.setLayersNumber(layersNumber);
        this.setNodesNumber(nodesNumber);
        this.data   = data;

    }

    void runTnTby2cv(int numberOfRepeats, int numberOfLayers){
        for(int i=0;i<numberOfRepeats;i++){
            Data learnData  = new Data(data.getPatientData()[0].length,data.getPatientDiagnoses()[0].length);
            Data testData   = new Data(data.getPatientData()[0].length,data.getPatientDiagnoses()[0].length);
            try {
                divideDataSet(learnData,testData);
                learnData.print();
            } catch (Exception e) {
                e.printStackTrace();
            }
            for(int learnSetIterations=1000;learnSetIterations<5000;learnSetIterations+=1000){
                System.out.printf("\n~~~~~~~~~~~~~~~~~~~~~~~~\nLearn set number of epochs: %d\n",learnSetIterations);
                learn(learnData.getPatientData(),learnData.getPatientDiagnoses(),learnSetIterations,true);
                test(testData.getPatientData(),testData.getPatientDiagnoses());
            }

        }
    }

    private void test(double[][] patientData, double[][] patientDiagnosis) {
/*
        double[][] Results          = np.add(np.dot(weights, patientData), biases);
        double[][] ActivationFunc   = np.sigmoid(Results);
        double cost                 = np.cross_entropy(getLayersNumber(), patientDiagnosis, ActivationFunc);
        double[][] Validation=resultsComparison(ActivationFunc,patientDiagnosis);

        System.out.println("\n~~~~~~~~~~~~~~~~~~~~~~~~\nValidation phase");
        System.out.println("Cost = "+cost);
        System.out.println("Results:");
        System.out.printf("%10s\t%10s\t%15s\t%30s\t\n","Acc","Sensitivity","False alarm","Positive predictivity value");
        for(double[] i:Validation)
            System.out.printf("%10f\t%10f\t%15f\t%30f\t", i[0], i[1], i[2], i[3]);*/
     }

    private double[][] resultsComparison(double[][] NNdiagnoze, double[][] patientState) {
        double[][] results = new double[patientState[0].length][4];
        int[] TP = new int[patientState[0].length];
        int[] FP = new int[patientState[0].length];
        int[] TN = new int[patientState[0].length];
        int[] FN = new int[patientState[0].length];
         for(int i=0;i<patientState[0].length;i++)
             for (int j = 0; j < patientState.length; j++) {
                 NNdiagnoze[j][i] = Math.round(NNdiagnoze[j][i]);
                 if (NNdiagnoze[j][i] == 1.0 && patientState[j][i] == 1.0) {//TP
                     TP[i]++;
                 } else if (NNdiagnoze[j][i] == 1.0 && patientState[j][i] == 0.0) {//FP
                     FP[i]++;
                 } else if (NNdiagnoze[j][i] == 0.0 && patientState[j][i] == 1.0) {//FN
                     FN[i]++;
                 } else if (NNdiagnoze[j][i] == 0.0 && patientState[j][i] == 0.0) {//TN
                     TN[i]++;
                 }
             }
         for(int i=0;i<TP.length;i++){
             results[i][0]=(double)(TP[i]+TN[i])/(double)(TP[i]+TN[i]+FP[i]+FN[i]);//ACCURACY
             results[i][1]=(double)TP[i]/(double)(TP[i]+FN[i]);//SENSITIVITY
             results[i][2]=(double)FP[i]/(double)(TN[i]+FP[i]);//FALSE ALARM
             results[i][3]=(double)TP[i]/(double)(TP[i]+FP[i]);//POSITIVE PREDICTIVITY VALUE
         }
        return np.multiply(100,np.divide(results,patientState.length));

    }

    private void divideDataSet(Data learnData,Data testData) throws Exception {
         Random r=new Random();
         for(int i=0;i<data.getNumberOfRecords();i++){
             if(r.nextBoolean() && (learnData.getNumberOfRecords()<(data.getNumberOfRecords()/2))) {
                 learnData.appendData(data.getPatientData()[i], data.getPatientDiagnoses()[i]);
             }
             else if(testData.getNumberOfRecords()<(data.getNumberOfRecords()/2)) {
                 testData.appendData(data.getPatientData()[i], data.getPatientDiagnoses()[i]);
             }
             else
                 learnData.appendData(data.getPatientData()[i],data.getPatientDiagnoses()[i]);
         }
         if(learnData.getNumberOfRecords()!=testData.getNumberOfRecords())throw new Exception("Different sizes of learn and test data");
     }


    private int getLayersNumber() {
        return layersNumber;
    }

    private void learn(double[][] patientsData,double[][] patientDiagnosis,int epochs, final boolean momentumOccurrence){
        weights     = new double[getLayersNumber()+1][][];
        biases      = new double[getLayersNumber()+1][];
         for(int i=0;i<weights.length;i++)
            weights[i] = np.random(getNodesNumber(),patientsData[0].length);
         for(int i=0;i<biases.length;i++)
             biases[i]   = new double[getNodesNumber()];
        int batchSize=patientsData.length;

        patientsData         = np.T(patientsData);
        patientDiagnosis    = np.T(patientDiagnosis);

        double alfa=  0.01;
        double beta=(1-alfa);
        double[][][] AccumulatorOfWeights=null;
        double[][] AccumulatorOfBiases=null;

        for(int i=0;i<epochs;i++) {
            double cost;
            //Forward Propagation
            for(int j=0;j<batchSize;j++) {
                double[][] Output = forwardPropagation(weights, patientsData[j], biases);
                cost = np.cross_entropy(batchSize, patientDiagnosis[j], Output[Output.length - 1]);
            }
                /*
            //Back Propagation
            double[][] dResults = np.subtract(Output[getLayersNumber()],patientDiagnosis);
            double[][] dWeights = np.divide(np.dot(dResults,np.T(patientData)),batchSize);
            double[][] dBiases  = np.divide(dResults,batchSize);

            //Momentum service
            if(AccumulatorOfWeights == null || AccumulatorOfBiases == null){
                AccumulatorOfWeights = new double[getLayersNumber()][dWeights.length][dWeights[0].length];
                AccumulatorOfBiases  = new double[dBiases.length][dBiases[0].length];
            }

            if(momentumOccurrence) {
                AccumulatorOfWeights = np.add(np.multiply(beta,AccumulatorOfWeights),np.multiply(1-beta,dWeights));
                AccumulatorOfBiases  = np.add(np.multiply(beta,AccumulatorOfBiases),np.multiply(1-beta,dBiases));
            }
            else {
                AccumulatorOfWeights = dWeights;
                AccumulatorOfBiases  = dBiases;
            }
            System.out.println("Jestem tutaj");
            //Gradient descent
            weights = np.subtract(weights, np.multiply(alfa, AccumulatorOfWeights));
            biases  = np.subtract(biases, np.multiply(alfa, AccumulatorOfBiases));*/
            /*if(i%(epochs/10)==0){
                System.out.println("\n~~~~~~~~~~~~~~~~~~~~~~~~\nLearning phase");
                System.out.println("Cost = "+cost);
                System.out.println("Predictions = " +Arrays.deepToString());
            }*/
        }
    }

    private double[][] forwardPropagation(double[][][] weights, double[] patientData, double[][] biases) {
        double [][] act     = new double[getLayersNumber()][];
        double[] results    = computeResultsForSingleLayer(weights[0], patientData, biases[0]);

        for(int actualLayer=1;actualLayer<getLayersNumber()-1;actualLayer++){
            results             = computeResultsForSingleLayer(weights[actualLayer],results,biases[actualLayer]);
            act[actualLayer]    = np.sigmoid(results);
        }

        results                     = computeResultsForSingleLayer(weights[getLayersNumber()-1], results, biases[getLayersNumber()-1]);
        act[getLayersNumber()-1]    = np.sigmoid(results);
        return act;
    }
    private double[] computeResultsForSingleLayer(double[][] weights,double[] input, double[] biases){
         if(weights.length!=biases.length)throw new RuntimeException("Different number of weights and biases for single layer");
         double[] results=new double[biases.length];
         for(int i=0;i<results.length;i++)
            results[i] = np.multiply(weights[i], input) +biases[i];
         return results;
    }
    private void setLayersNumber(int layersNumber) {
        this.layersNumber = layersNumber;
    }

    private int getNodesNumber() {
        return nodesNumber;
    }

    private void setNodesNumber(int nodesNumber) {
        this.nodesNumber = nodesNumber;
    }

    private class EmptyDataException extends RuntimeException {
        EmptyDataException(String s) {
            super(s);
            this.printStackTrace();
        }
    }
}
