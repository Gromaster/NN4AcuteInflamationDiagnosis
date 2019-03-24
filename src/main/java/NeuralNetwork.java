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
        weights     = np.random(getLayersNumber(),data.getPatientData()[0].length);
        biases      = new double[getLayersNumber()][data.getPatientData().length];

    }

    void runTnTby2cv(int numberOfRepeats){
        for(int i=0;i<numberOfRepeats;i++){
            Data learnData  = new Data(data.getNumberOfRecords()/2,data.getPatientData()[0].length,data.getPatientDiagnoses()[0].length);
            Data testData   = new Data(data.getNumberOfRecords()/2,data.getPatientData()[0].length,data.getPatientDiagnoses()[0].length);
            try {
                divideDataSet(learnData,testData);
            } catch (Exception e) {
                e.printStackTrace();
            }
            for(int learnSetIterations=1000;learnSetIterations<5000;learnSetIterations+=1000){
                learn(learnData.getPatientData(),learnData.getPatientDiagnoses(),learnSetIterations,true);
                test(testData.getPatientData(),testData.getPatientDiagnoses());
            }



        }
    }

    private void test(double[][] patientData, double[][] patientDiagnoses) {

    }

    private void divideDataSet(Data learnData,Data testData) throws Exception {
         Random r=new Random();
         for(int i=0;i<data.getNumberOfRecords();i++){
             if(r.nextBoolean()&&(learnData.getNumberOfRecords()<(data.getNumberOfRecords()/2)))
                 learnData.appendData(data.getPatientData()[i],data.getPatientDiagnoses()[i]);
             else if(testData.getNumberOfRecords()<(data.getNumberOfRecords()/2))
                 testData.appendData(data.getPatientData()[i],data.getPatientDiagnoses()[i]);
             else
                 learnData.appendData(data.getPatientData()[i],data.getPatientDiagnoses()[i]);
         }
         if(learnData.getNumberOfRecords()!=testData.getNumberOfRecords())throw new Exception("Different sizes of learn and test data");
     }


    private int getLayersNumber() {
        return layersNumber;
    }

    private void learn(double[][] patientData,double[][]patientDiagnosis,int iterations, final boolean MomentumOccurence){

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

        for(int i=0;i<iterations;i++) {
            //Forward Propagation
            double[][] Results          = np.add(np.dot(weights, patientData), biases);
            double[][] ActivationFunc   = np.sigmoid(Results);
            double cost                 = np.cross_entropy(getLayersNumber(), patientDiagnosis, ActivationFunc);

            //Back Propagation
            double[][] dResults = np.subtract(ActivationFunc,patientDiagnosis);
            double[][] dWeights = np.divide(np.dot(dResults,np.T(patientData)),getLayersNumber());
            double[][] dBiases  = np.divide(dResults,getLayersNumber());

            //Momentum service
            if(AccumulatorOfWeights == null || AccumulatorOfBiases == null){
                AccumulatorOfWeights = new double[dWeights.length][dWeights[0].length];
                AccumulatorOfBiases  = new double[dBiases.length][dBiases[0].length];
            }

            if(MomentumOccurence) {
                AccumulatorOfWeights = np.add(np.multiply(beta,AccumulatorOfWeights),np.multiply(1-beta,dWeights));
                AccumulatorOfBiases  = np.add(np.multiply(beta,AccumulatorOfBiases),np.multiply(1-beta,dBiases));
            }
            else {
                AccumulatorOfWeights = dWeights;
                AccumulatorOfBiases  = dBiases;
            }

            //Gradient descent
            weights = np.subtract(weights, np.multiply(alfa, AccumulatorOfWeights));
            biases  = np.subtract(biases, np.multiply(alfa, AccumulatorOfBiases));
            if(i%(iterations/10)==0){
                System.out.println("\n~~~~~~~~~~~~~~~~~~~~~~~~");
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
