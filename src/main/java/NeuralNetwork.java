import java.util.Random;

class NeuralNetwork {
    private int nodesNumber, numberOfHiddenLayers;
    private double[][][] weights;
    private double[][] biases;
    private Data data;
    private int numberOfInputAttributes;

     NeuralNetwork(int layersNumber, int nodesNumber, Data data) {
        this.setLayersNumber(layersNumber);
        this.setNodesNumber(nodesNumber);
        this.data   = data;

    }

    void runTnTby2cv(int numberOfRepeats){
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
        double cost                 = np.cross_entropy(getNumberOfHiddenLayers(), patientDiagnosis, ActivationFunc);
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


    private int getNumberOfHiddenLayers() {
        return numberOfHiddenLayers;
    }

    private void learn(double[][] patientsData,double[][] patientsDiagnosis,int epochs, final boolean momentumOccurrence){
        weights     = new double[getNumberOfHiddenLayers()+1][][];
        biases      = new double[getNumberOfHiddenLayers()+1][];

        weights[0]=new double[getNodesNumber()][patientsData[0].length];
        for(int i=1;i<weights.length-1;i++)
            weights[i] = np.random(getNodesNumber(),weights[i-1].length);
        weights[getNumberOfHiddenLayers()]=new double[patientsDiagnosis[0].length][getNodesNumber()];
         
        for(int i=0;i<biases.length-1;i++)
             biases[i]   = new double[getNodesNumber()];
        biases[getNumberOfHiddenLayers()]=new double[patientsDiagnosis[0].length];

        int batchSize=patientsData.length;
/*
        patientsData         = np.T(patientsData);
        patientsDiagnosis    = np.T(patientsDiagnosis);*/

        double alfa=  0.01;
        double beta=(1-alfa);
        double[][][] AccumulatorOfWeights=null;
        double[][] AccumulatorOfBiases=null;

        for(int i=0;i<epochs;i++) {
            double cost;
            for(int j=0;j<batchSize;j++) {
                //Forward Propagation
                double[][] Output = forwardPropagation(weights, patientsData[j], biases);
                cost = np.cross_entropy(batchSize, patientsDiagnosis[j], Output[Output.length - 1]);
                if(i%100==0 && j%10==0)
                    System.out.println("Epoch: "+i+"\tIteration: " +j+"\tCost:"+cost);

                //Back propagation


            }
                /*
            //Back Propagation
            double[][] dResults = np.subtract(Output[getNumberOfHiddenLayers()],patientDiagnosis);
            double[][] dWeights = np.divide(np.dot(dResults,np.T(patientData)),batchSize);
            double[][] dBiases  = np.divide(dResults,batchSize);

            //Momentum service
            if(AccumulatorOfWeights == null || AccumulatorOfBiases == null){
                AccumulatorOfWeights = new double[getNumberOfHiddenLayers()][dWeights.length][dWeights[0].length];
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
        int m=getNumberOfHiddenLayers();
        double [][] results = new double[m+1][];
        results[0]          = computeResultsForSingleLayer(weights[0], patientData, biases[0]);
        for(int actualLayer=1;actualLayer<m;actualLayer++)
            results[actualLayer] = computeResultsForSingleLayer(weights[actualLayer],results[actualLayer-1],biases[actualLayer]);

        results[m] = computeResultsForSingleLayer(weights[m], results[m-1], biases[m]);

        return results;
    }

    private double[] computeResultsForSingleLayer(double[][] weightsOnSingleLayer,double[] input, double[] biases) {
         if(weightsOnSingleLayer.length!=biases.length)throw new RuntimeException("Different number of weights and biases for single layer");
         double[] results = np.add(np.dot(weightsOnSingleLayer,input),biases);
         return np.sigmoid(results);
    }

    private void setLayersNumber(int layersNumber) {
        this.numberOfHiddenLayers = layersNumber;
    }

    private int getNodesNumber() {
        return nodesNumber;
    }

    private void setNodesNumber(int nodesNumber) {
        this.nodesNumber = nodesNumber;
    }

    public int getNumberOfInputAttributes() {
        return numberOfInputAttributes;
    }

    public void setNumberOfInputAttributes(int numberOfInputAttributes) {
        this.numberOfInputAttributes = numberOfInputAttributes;
    }

    private class EmptyDataException extends RuntimeException {
        EmptyDataException(String s) {
            super(s);
            this.printStackTrace();
        }
    }
}
