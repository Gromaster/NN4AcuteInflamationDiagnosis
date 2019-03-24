import java.util.Arrays;
import java.util.Random;

class NeuralNetwork {
    private int nodesNumber, layersNumber;
    private double[][] weights,biases;
    private Data data;

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

        double[][] Results          = np.add(np.dot(weights, patientData), biases);
        double[][] ActivationFunc   = np.sigmoid(Results);
        double cost                 = np.cross_entropy(getLayersNumber(), patientDiagnosis, ActivationFunc);
        double[][] Validation=resultsComparison(ActivationFunc,patientDiagnosis);

        System.out.println("\n~~~~~~~~~~~~~~~~~~~~~~~~\nValidation phase");
        System.out.println("Cost = "+cost);
        System.out.println("Results:");
        System.out.printf("%10s\t%10s\t%15s\t%30s\t\n","Acc","Sensitivity","False alarm","Positive predictivity value");
        for(double[] i:Validation)
            System.out.printf("%10f\t%10f\t%15f\t%30f\t", i[0], i[1], i[2], i[3]);
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

    private void learn(double[][] patientData,double[][] patientDiagnosis,int epochs, final boolean momentumOccurrence){
        weights     = np.random(getNodesNumber(),patientData[0].length);
        biases      = new double[getNodesNumber()][patientData.length];
        int m=patientData.length;
        if(patientData==null || patientDiagnosis==null) {
            throw new EmptyDataException("No data given to the NN");
        }
        else {
            patientData         = np.T(patientData);
            patientDiagnosis    = np.T(patientDiagnosis);
        }

        double alfa=  0.01;
        double beta=1/(1-alfa);
        double[][] AccumulatorOfWeights=null;
        double[][] AccumulatorOfBiases=null;

        for(int i=0;i<epochs;i++) {

            //Forward Propagation
            double[][] Results          = np.add(np.dot(weights, patientData), biases);
            double[][] ActivationFunc   = np.sigmoid(Results);
            System.out.println("Dimensions: "+patientDiagnosis.length+"x"+patientDiagnosis[0].length+"  activ "+ActivationFunc.length+"x"+ActivationFunc[0].length);
            double cost                 = np.cross_entropy(m, patientDiagnosis, ActivationFunc);

            //Back Propagation
            double[][] dResults = np.subtract(ActivationFunc,patientDiagnosis);
            double[][] dWeights = np.divide(np.dot(dResults,np.T(patientData)),m);
            double[][] dBiases  = np.divide(dResults,m);

            //Momentum service
            if(AccumulatorOfWeights == null || AccumulatorOfBiases == null){
                AccumulatorOfWeights = new double[dWeights.length][dWeights[0].length];
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
            biases  = np.subtract(biases, np.multiply(alfa, AccumulatorOfBiases));
            if(i%(epochs/10)==0){
                System.out.println("\n~~~~~~~~~~~~~~~~~~~~~~~~\nLearning phase");
                System.out.println("Cost = "+cost);
                System.out.println("Predictions = " +Arrays.deepToString(ActivationFunc));
            }
        }
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
